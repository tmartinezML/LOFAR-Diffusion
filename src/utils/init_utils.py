import json
from pathlib import Path
import warnings

import torch

import model.unet as unet
import model.unet_prev as unet_prev
from utils.config_utils import modelConfig
import utils.paths as paths


def load_config(config_file):
    # Load model config
    with open(config_file, "r") as f:
        config = json.load(f)

    return modelConfig(**config)


def load_config_from_path(path):
    model_name = path.name
    config_file = path / f"config_{model_name}.json"
    return load_config(config_file)


def load_parameters(model, path, key="ema_model"):
    # Load model weights
    state_dict = torch.load(path, map_location="cpu")[key]
    if key != "model":
        # Remove 'module.' from keys
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }
    model.load_state_dict(state_dict)
    return model


def load_model(config_file, model_file=None, key="ema_model"):
    # Load model config
    config = load_config(config_file)
    # Load model
    model = unet.EDMPrecond.from_config(config)
    # Load model weights
    if model_file is not None:
        load_parameters(model, model_file, key=key)

    return model


def load_model_from_folder(path, key="ema_model", return_config=False):
    model_name = path.name
    model_file = path / f"parameters_{model_name}.pt"
    config_file = path / f"config_{model_name}.json"

    print(f"Loading model from {model_file} and {config_file}")

    if return_config:
        out = (load_model(config_file, model_file, key=key), load_config(config_file))
    else:
        out = load_model(config_file, model_file, key=key)
    return out


def load_model_by_name(name, key='ema_model'):
    path = paths.MODEL_PARENT / name
    return load_model_from_folder(path, key=key)


def load_old_model_from_folder(path, use_ema=True, return_config=False):
    warnings.warn("Using previous UNet model.")
    model_name = path.name
    model_file = path / f"parameters_{model_name}.pt"
    config_file = path / f"config_{model_name}.json"

    print(f"Loading model from {model_file} and {config_file}")
    config = load_config(config_file)
    # Load model
    model = unet_prev.EDMPrecond.from_config(config)
    # Load model weights
    load_parameters(model, model_file, use_ema=use_ema)
    return model


def load_snapshot(path, iter, key='ema_model', model=None):
    if model is None:
        model = load_model_from_folder(path, key=key)

    if iter == 0:
        print("Snapshot iteration is 0 - returning final model.")
        return model

    snapshot_file = path / f"snapshots/snapshot_iter_{iter:08d}.pt"
    if not snapshot_file.exists():
        raise FileNotFoundError(f"Snapshot file {snapshot_file} not found.")
    print(f"Loading snapshot from {snapshot_file}")
    # model.load_state_dict(torch.load(snapshot_file, map_location="cpu")[key])
    return load_parameters(model, snapshot_file, key=key)


def model_name_from_file(path):
    name = path.stem.replace("parameters_", "")
    name = name.replace("model_", "").replace("ema_", "")
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
