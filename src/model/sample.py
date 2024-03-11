import json
from pathlib import Path
from datetime import datetime

import h5py
from torch import tensor
from torchvision.utils import save_image
from torchvision.transforms import Lambda
from tqdm import tqdm

from model.diffusion import Diffusion
from utils.device_utils import distribute_model, set_visible_devices
from utils.plot_utils import plot_image_grid
from utils.init_utils import (
    load_model, load_model_from_folder, load_diffusion_from_config,
    load_diffusion_from_config_file, load_snapshot
)
from utils.paths import GEN_DATA_PARENT, MODEL_PARENT


def sample_batch(
    model,
    diffusion,
    bsize,
    img_size=80,
    return_steps=False,
    label_batch=None,
    **sample_kwargs,
):

    model = model.eval()
    imgs = diffusion.edm_sampling(
        model,
        img_size,
        batch_size=bsize,
        label_batch=label_batch,
        **sample_kwargs,
    )

    return imgs if return_steps else imgs[-1]


def save_batch_png(imgs, out_folder, file_prefix="samples", rescale=True,):

    for i, img in tqdm(enumerate(imgs), desc="Saving..."):

        if rescale:
            img = (img + 1) / 2

        save_image(
            img,
            out_folder.joinpath(f"{file_prefix}_{i:04d}.png")
        )


def _h5_dataset_append(dset, data):
    dset.resize(dset.shape[0] + data.shape[0], axis=0)
    dset[-data.shape[0]:] = data


def save_batch_hdf5(imgs, out_file,
                    labels=None, dataset_name="samples", rescale=True, attrs={}):

    if rescale:
        imgs = (imgs + 1) / 2

    with h5py.File(out_file, "a") as f:

        # If dataset already exists, append to it
        if dataset_name in f:

            # Append images
            dset = f[dataset_name]
            _h5_dataset_append(dset, imgs.cpu().numpy())
            dset.attrs.update(attrs)

        # If dataset does not exist, create it
        else:
            # Create dataset w. images
            dset = f.create_dataset(
                f"{dataset_name}",
                data=imgs.cpu().numpy(),
                chunks=True,
                maxshape=(None, 1, *imgs.shape[2:]),
            )
            dset.attrs.update(attrs)

        if labels is not None:
            if f"{dataset_name}_labels" in f:
                _h5_dataset_append(f[f"{dataset_name}_labels"], labels)

            else:
                f.create_dataset(
                    f"{dataset_name}_labels",
                    data=labels,
                    chunks=True,
                    maxshape=(None,),
                )


def sample_and_save_batches(
    model,
    diffusion,
    out_path,
    bsize,
    n_batches,
    labels=None,
    prefix_shift=0,
    dataset_name="samples",
    h5=True,
    **sample_kwargs,
):
    if labels is not None:
        assert n_batches % len(labels) == 0, (
            "Number of labels must be divisible by number of batches."
        )
        if not h5:
            raise NotImplementedError(
                "Saving as PNG files is only implemented for label-less "
                "sampling."
            )
        labels = labels * (n_batches // len(labels))

    for i in range(n_batches):
        # Set labels
        label_batch = tensor([labels[i]] * bsize) if labels else None

        # Sample
        imgs = sample_batch(
            model,
            diffusion,
            bsize,
            label_batch=label_batch,
            **sample_kwargs
        )

        # Save as single HDF5 files
        if h5:
            save_batch_hdf5(
                imgs,
                out_path,
                labels=label_batch.numpy() if labels else None,
                dataset_name=dataset_name
            )

        # Save as PNG files
        else:
            out_folder = out_path / dataset_name
            out_path.mkdir(exist_ok=True, parents=True)
            save_batch_png(
                imgs,
                out_folder,
                file_prefix=f'{i + prefix_shift}_{dataset_name}'
            )


def sample_set_from_config_model_files(
        config_file,
        model_file,
        out_path,
        bsize,
        n_batches,
        n_devices=1,
        T=None,
        img_size=80,
        save_as_h5=True,
        **sampling_kwargs,
):

    # Load model & diffusion
    model = load_model(config_file, model_file=model_file)
    model = distribute_model(model, n_devices)
    diffusion = load_diffusion_from_config_file(config_file)
    if T:
        diffusion.timesteps = T

    # Sample & Save
    sample_and_save_batches(
        model,
        diffusion,
        out_path,
        bsize,
        n_batches,
        img_size=img_size,
        h5=save_as_h5,
        **sampling_kwargs
    )

    return


def save_sampling_info(out_path, info_dict):
    if out_path.is_dir():
        info_file = out_path / "sampling_info.json"
        with open(info_file, "w") as f:
            json.dump(info_dict, f, indent=4)

    else:
        with h5py.File(out_path, "a") as f:

            if not info_dict["dataset_name"] in f:
                # TODO: This creates problems with dataset. Fix it.
                f.create_dataset(
                    info_dict["dataset_name"],
                    (0, 0, 0, 0),
                    chunks=True,
                    maxshape=(None, 1, *(info_dict["img_size"],) * 2)
                )

            f[info_dict["dataset_name"]].attrs.update(info_dict)


def sample_set_from_model(
        model_name,
        batch_size,
        n_batches,
        n_devices=1,
        labels=None,
        T=None,
        use_ema=True,
        img_size=80,
        snapshot_iter=None,
        save_as_h5=True,
        dataset_suffix=None,
        out_parent=GEN_DATA_PARENT,
        model_parent=MODEL_PARENT,
        **sampling_kwargs,
):
    model_dir = model_parent / model_name

    # Set output paths
    dataset_name = f"samples_{model_name}"
    if dataset_suffix is not None:
        dataset_name += f"_{dataset_suffix}"

    if save_as_h5:
        out_path = out_parent / f"{model_name}.h5"
    else:
        out_path = out_parent / model_name

    # If out folder contains images, shift prefix:
    prefix_shift = 0
    if len(list(out_path.glob("*.png"))):
        prefix_shift = max(
            [int(f.name[0]) for f in list(out_path.glob("*.png"))]
        ) + 1

    # Load model & config
    model, config = load_model_from_folder(
        model_dir, use_ema=use_ema, return_config=True
    )
    if snapshot_iter is not None:
        model = load_snapshot(
            model_dir, snapshot_iter, use_ema=use_ema, model=model
        )

    # Move to GPU
    model, _ = distribute_model(model, n_devices)

    # Load diffusion
    diffusion = load_diffusion_from_config(config)
    if T:
        diffusion.timesteps = T

    # Save sampling info
    sampling_info = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "img_size": img_size,
        "sampling_timesteps": diffusion.timesteps,
        **sampling_kwargs,
    }

    # Sample
    t0 = datetime.now()
    sample_and_save_batches(
        model,
        diffusion,
        out_path,
        batch_size,
        n_batches,
        labels=labels,
        img_size=img_size,
        prefix_shift=prefix_shift,
        dataset_name=dataset_name,
        h5=save_as_h5,
        **sampling_kwargs
    )
    save_sampling_info(out_path, sampling_info)
    dt = datetime.now() - t0
    print(f"Sampling done after {dt}.")
    return out_path


if __name__ == "__main__":

    # Setup
    model_name = 'EDM_SNR5_50as'
    n_devices = 2
    # n_samples = 100
    # labels = [0, 1, 2, 3]
    batch_size = 1000
    n_batches = 10
    # n_batches = len(labels)

    sample_set_from_model(
        model_name,
        batch_size=batch_size,
        n_batches=n_batches,
        # labels=labels,
        n_devices=n_devices,
        T=25,
        use_ema=True,
        img_size=80,
    )
