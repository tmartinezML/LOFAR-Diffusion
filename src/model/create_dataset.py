import json
from pathlib import Path
from datetime import datetime

import torch
from torchvision.utils import save_image
from torchvision.transforms import Lambda
from torch.nn import DataParallel
from tqdm import tqdm

from model.unet import Unet
from model.diffusion import Diffusion
from model.edm_diffusion import EDM_Diffusion
from utils.sample_utils import load_model, load_config
from utils.train_utils import get_free_gpu

OUT_PARENT = Path("/storage/tmartinez/image_data/generated")
SAMPLING_METHODS = [
    # EDM diffusion
    "edm_stochastic_sampling",
    "edm_deterministic_sampling",
    # Original diffusion
    "p_sampling",
    "ddim_sampling",
]

def sample_batch(
        model,
        diffusion,
        bsize,
        sampling_method="edm_stochastic_sampling",
        img_size=80,
        **sample_kwargs,
    ):
    assert sampling_method in SAMPLING_METHODS, (
        f"Sampling method {sampling_method} not implemented."
    )
    try:
        sampling = getattr(diffusion, sampling_method)
    except AttributeError as e:
        f"Tried to use sampling method {sampling_method} with diffusion " \
        f"object of type {type(diffusion)}, which does not have this method."
        raise e
    
    # Only last diffusion time step (= final sample):
    imgs = sampling(model, img_size, batch_size=bsize, **sample_kwargs)[-1]

    return imgs

def sample_and_save_batch(
        model,
        diffusion,
        bsize,
        out_folder,
        file_prefix="sample",
        **kwargs,
    ):
    imgs = sample_batch(model, diffusion, bsize, **kwargs)
    for i, img in tqdm(enumerate(imgs), desc="Saving..."):
        rescale = Lambda(lambda t: (t + 1) / 2)
        save_image(rescale(img), 
                   out_folder.joinpath(f"{file_prefix}_{i:04d}.png"))

def sample_set(
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

def sample_set_from_folder(
        model_dir,
        batch_size,
        n_batches,
        n_devices=1,
        T=1000,
        use_ema=True,
        img_size=80,
        sampling_method="edm_stochastic_sampling",
        out_parent=OUT_PARENT,
        **sampling_kwargs,):
    
    model_name = model_dir.name
    out_folder = out_parent.joinpath(f"samples_{model_name}")
    Path.mkdir(out_folder, exist_ok=True, parents=False)
    prefix_shift = 0
    if len(list(out_folder.glob("*.png"))) > 0:
        prefix_shift = max(
            [int(f.name[0]) for f in list(out_folder.glob("*.png"))]
        ) + 1

    model_file = model_dir.joinpath(
        f"parameters_{'ema' if use_ema else 'model'}_{model_name}.pt"
    )
    config_file = sorted(model_dir.glob("*.json"))
    assert len(config_file) <= 1, "More than one .json file found in model dir!"
    assert len(config_file) == 1, "No .json file found in model dir!"
    config_file = config_file[0]

    # Load model
    model = load_model(config_file, model_file)
    if n_devices == 1:
        model.to(torch.device('cuda', get_free_gpu()[0]))

    else:
        device_ids = get_free_gpu()[:n_devices]
        model.to(torch.device('cuda', device_ids[0]))
        model = DataParallel(
            model,
            device_ids=device_ids
        )
    
    # Load diffusion
    assert sampling_method in SAMPLING_METHODS, (
        f"Sampling method {sampling_method} not implemented."
    )
    diff_class = EDM_Diffusion if "edm" in sampling_method else Diffusion
    diffusion = diff_class.from_config(load_config(config_file))
    diffusion.timesteps = T

    # Sample
    t0 = datetime.now()
    sample_set(model, diffusion, out_folder, batch_size, n_batches,
                img_size=img_size, sampling_method=sampling_method,
                prefix_shift=prefix_shift, **sampling_kwargs)
    dt = datetime.now() - t0
    print(f"Sampling took {dt}.")

    # Save sampling info
    sampling_info = {
        "model_name": model_name,
        "model_file": str(model_file),
        "config_file": str(config_file),
        "batch_size": batch_size,
        "n_batches": n_batches,
        "img_size": img_size,
        "sampling_method": sampling_method,
        "sampling_time": str(dt),
    }
    info_file = out_folder.joinpath("sampling_info.json")
    i = 0
    while info_file.exists():
        info_file = out_folder.joinpath(f"sampling_info_{i}.json")
        i += 1

    with open(info_file, "w") as f:
        json.dump(sampling_info, f, indent=4)

    print("Sampling done.")

               

if __name__ == "__main__":
    model_dir_parent = Path("/home/bbd0953/diffusion/results/channel_configs")
    model_dirs = sorted(model_dir_parent.iterdir())

    # Available sampling methods:

    # EDM diffusion:
    # "edm_stochastic_sampling",
    # "edm_deterministic_sampling",

    # Original diffusion:
    # "p_sampling",
    # "ddim_sampling",

    for model_dir in model_dirs:
        sample_set_from_folder(
            model_dir,
            batch_size=1000,
            n_batches=10,
            n_devices=2,
            T=250,
            use_ema=True,
            img_size=80,
            sampling_method="p_sampling",
        )

