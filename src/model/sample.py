import json
from pathlib import Path
from datetime import datetime

from torchvision.utils import save_image
from torchvision.transforms import Lambda
from tqdm import tqdm

from model.diffusion import Diffusion
from model.edm_diffusion import EDM_Diffusion
from utils.device_utils import distribute_model
from utils.plot_utils import plot_samples
from utils.init_utils import (
    load_model, load_model_from_folder,
    load_diffusion_from_config, load_diffusion_from_config_file
)

OUT_PARENT = Path("/storage/tmartinez/image_data/generated")
SAMPLING_METHODS = [
    # EDM diffusion
    "edm_stochastic_sampling",
    "edm_deterministic_sampling",
    # Original diffusion
    "p_sampling",
    "ddim_sampling",
]


def get_sampling_method(diffusion, sampling_method):
    assert sampling_method in SAMPLING_METHODS, (
        f"Sampling method {sampling_method} not implemented."
    )
    try:
        sampling = getattr(diffusion, sampling_method)
    except AttributeError as e:
        f"Tried to use sampling method {sampling_method} with diffusion " \
            f"object of type {type(diffusion)}, which does not have this method."
        raise e
    return sampling


def sample_batch(
    model,
    diffusion,
    bsize,
    sampling_method="edm_stochastic_sampling",
    img_size=80,
    return_steps=False,
    **sample_kwargs,
):
    sampling = get_sampling_method(diffusion, sampling_method)
    model = model.eval()
    imgs = sampling(model, img_size, batch_size=bsize, **sample_kwargs)
    return imgs if return_steps else imgs[-1]


def save_batch(imgs, out_folder, file_prefix="sample"):
    for i, img in tqdm(enumerate(imgs), desc="Saving..."):
        rescale = Lambda(lambda t: (t + 1) / 2)
        save_image(rescale(img),  # Only last time step
                   out_folder.joinpath(f"{file_prefix}_{i:04d}.png"))


def sample_and_save_batch(
    model,
    diffusion,
    bsize,
    out_folder,
    file_prefix="sample",
    **kwargs,
):
    out_folder.mkdir(exist_ok=True)
    imgs = sample_batch(model, diffusion, bsize, **kwargs)
    save_batch(imgs, out_folder, file_prefix=file_prefix)


def sample_n_batches(
    model,
    diffusion,
    out_folder,
    bsize,
    n_batches,
    prefix_shift=0,
    **kwargs,
):
    for i in range(n_batches):
        sample_and_save_batch(
            model, diffusion, bsize, out_folder,
            file_prefix=f"{i+prefix_shift}", **kwargs
        )


def sample_set_from_config_model_files(
        config_file,
        model_file,
        out_folder,
        bsize,
        n_batches,
        n_devices=1,
        T=None,
        img_size=80,
        sampling_method="edm_stochastic_sampling",
        **sampling_kwargs,):
    model = load_model(config_file, model_file=model_file)
    model = distribute_model(model, n_devices)
    diffusion = load_diffusion_from_config_file(config_file)
    if T:
        diffusion.timesteps = T
    sample_n_batches(
        model, diffusion, out_folder, bsize, n_batches,
        img_size=img_size, sampling_method=sampling_method,
        **sampling_kwargs
    )
    return


def sample_set_from_folder(
        model_dir,
        batch_size,
        n_batches,
        n_devices=1,
        T=None,
        use_ema=True,
        img_size=80,
        sampling_method="edm_stochastic_sampling",
        out_parent=OUT_PARENT,
        out_folder_suffix=None,
        **sampling_kwargs,):
    # Set paths and create output folder
    model_name = model_dir.name
    model_file = model_dir / f"parameters_{'ema' if use_ema else 'model'}_"\
                             f"{model_name}.pt"
    config_file = model_dir / f"config_{model_name}.json"
    out_folder = out_parent.joinpath(
        f"samples_{model_name}"
        + (f"_{out_folder_suffix}" if out_folder_suffix else "")
    )
    Path.mkdir(out_folder, exist_ok=True, parents=False)
    info_file = out_folder.joinpath("sampling_info.json")
    i = 0
    while info_file.exists():
        info_file = out_folder.joinpath(f"sampling_info_{i}.json")
        i += 1

    # If out folder contains images, shift prefix:
    prefix_shift = 0
    if len(list(out_folder.glob("*.png"))) > 0:
        prefix_shift = max(
            [int(f.name[0]) for f in list(out_folder.glob("*.png"))]
        ) + 1

    # Load model & config
    model, config = load_model_from_folder(model_dir, use_ema=use_ema,
                                           return_config=True)
    # Move to GPU
    model = distribute_model(model, n_devices)
    # Load diffusion
    diffusion = load_diffusion_from_config(config)
    if T:
        diffusion.timesteps = T

    # Save sampling info
    sampling_info = {
        "model_name": model_name,
        "model_file": str(model_file),
        "config_file": str(config_file),
        "batch_size": batch_size,
        "n_batches": n_batches,
        "img_size": img_size,
        "sampling_method": sampling_method,
        "sampling_timesteps": diffusion.timesteps,
        "sampling_finished": False,
        "n_gpu": n_devices,
    }
    with open(info_file, "w") as f:
        json.dump(sampling_info, f, indent=4)

    # Sample
    t0 = datetime.now()
    sample_n_batches(model, diffusion, out_folder, batch_size, n_batches,
                     img_size=img_size, sampling_method=sampling_method,
                     prefix_shift=prefix_shift, **sampling_kwargs)
    dt = datetime.now() - t0
    print(f"Sampling took {dt}.")

    # Update sampling info
    sampling_info.update(
        {
            "sampling_finished": True,
            "sampling_time": str(dt),
        }
    )
    with open(info_file, "w") as f:
        json.dump(sampling_info, f, indent=4)

    print("Sampling done.")


if __name__ == "__main__":
    # Available sampling methods:

    # EDM diffusion:
    # "edm_stochastic_sampling",
    # "edm_deterministic_sampling",

    # Original diffusion:
    # "p_sampling",
    # "ddim_sampling",

    model_dir = Path(
        "/home/bbd0953/diffusion/results/InitModel_EDM_SnapshotRun")

    sample_set_from_folder(
        model_dir,
        batch_size=1536,
        n_batches=1,
        n_devices=3,
        T=50,
        use_ema=True,
        img_size=80,
        sampling_method="edm_stochastic_sampling",
        out_parent=OUT_PARENT,
        out_folder_suffix="T=50",
    )


def sneak_peek(model_dir,
               img_size=80,
               use_ema=True,
               sampling_method="edm_stochastic_sampling",
               T=None,
               batch_size=25):
    # Load model
    model, conf = load_model_from_folder(model_dir, use_ema=use_ema,
                                         return_config=True)
    distribute_model(model)

    # Load diffusion
    assert sampling_method in SAMPLING_METHODS, (
        f"Sampling method {sampling_method} not implemented."
    )
    diff_class = EDM_Diffusion if "edm" in sampling_method else Diffusion
    diffusion = diff_class.from_config(conf)
    if T:
        diffusion.timesteps = T

    try:
        sampling = getattr(diffusion, sampling_method)
    except AttributeError as e:
        f"Tried to use sampling method {sampling_method} with diffusion " \
            f"object of type {type(diffusion)}, which does not have this method."
        raise e

    # Sample
    imgs = sampling(model, img_size, batch_size=batch_size)[-1]
    fig, axs = plot_samples(imgs, title=f"Model {model_dir.name} samples")
    return fig, axs
