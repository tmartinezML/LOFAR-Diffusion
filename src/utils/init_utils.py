import json
from pathlib import Path

import torch

import model.unet as unet
from model.diffusion import EDM_Diffusion
from utils.config_utils import modelConfig


def load_config(config_file):
    # Load model config
    with open(config_file, "r") as f:
        config = json.load(f)

    return modelConfig(**config)

def load_config_from_path(path):
    model_name = path.name
    config_file = Path(path) / f"config_{model_name}.json"
    return load_config(config_file)

def load_model(config_file, model_file=None):
    # Load model config
    config = load_config(config_file)
    # Load model
    model = getattr(unet, config.model_type).from_config(config)
    # Load model weights
    if model_file is not None:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))

    return model


def load_model_from_folder(path, use_ema=True, return_config=False):
    model_name = path.name
    model_file = path / f"parameters_{'ema' if use_ema else 'model'}_"\
                        f"{model_name}.pt"
    config_file = path / f"config_{model_name}.json"

    print(f"Loading model from {model_file} and {config_file}")

    if return_config:
        return load_model(config_file, model_file), load_config(config_file)
    else:
        return load_model(config_file, model_file)


def load_diffusion_from_config(config):
    if 'EDM' in config.model_type:
        diffusion = EDM_Diffusion.from_config(config)
    else:
        raise NotImplementedError(f"Diffusion for {config.model_type} not "
                                  f"implemented.")
    return diffusion


def load_diffusion_from_config_file(path):
    config = load_config(path)
    return load_diffusion_from_config(config)

def load_diffusion_from_folder(path):
    model_name = path.name
    config_file = path / f"config_{model_name}.json"
    return load_diffusion_from_config_file(config_file)

def model_name_from_file(path):
    name = path.stem.replace('parameters_', '')
    name = name.replace('model_', '').replace('ema_', '')
    return name

def rename_files(path, model_name_new, model_name_old=None):
    if model_name_old is None:
        model_name_old = path.name

    for file in path.iterdir():
        if file.is_file():
            name = file.stem.replace(model_name_old, model_name_new)
            file.rename(path / f"{name}{file.suffix}")
        elif file.is_dir():
            rename_files(file, model_name_new, model_name_old)