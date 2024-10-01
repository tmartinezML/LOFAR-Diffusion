from pathlib import Path

import torch
import torch.nn as nn

import utils.logging
import model.unet as unet
import utils.paths as paths
from model.config import modelConfig

logger = utils.logging.get_logger(__name__)


def parse_model_specifier(source: str | Path):
    """
    Parse a model specifier to return the model name and paths, as well as path
    to config file.

    Parameters
    ----------
    source : str or Path
        The model specifier.

    Returns
    -------
    model_name : str
        The name of the model.
    model_dir : Path
        The directory containing the model files.
    model_file : Path
        The path to the model weights file.
    config_file : Path
        The path to the model configuration file.
    """
    # Set paths according to source, which can be a file, a directory, or a model name.
    match source:

        # If source is a file, it is assumed to be the config file.
        case Path() if source.is_file():
            assert (
                source.suffix == ".json"
            ), f"This does not look like a config file: {source}"
            model_dir = source.parent
            config_file = source
            model_name = model_dir.name
            model_file = model_dir / f"parameters_{model_name}.pt"

        # If source is a directory, it is assumed to be the model directory.
        case Path() if source.is_dir():
            model_name = source.name
            model_dir = source
            model_file = source / f"parameters_{model_name}.pt"
            config_file = source / f"config_{model_name}.json"

        # If source is a string, it is assumed to be the model name.
        case str():
            model_name = source
            model_dir = paths.MODEL_PARENT / model_name
            model_file = model_dir / f"parameters_{model_name}.pt"
            config_file = model_dir / f"config_{model_name}.json"

        # Anything else is invalid.
        case _:
            raise ValueError(f"Invalid model identifier: {source}")

    return model_name, model_dir, model_file, config_file


def load_model(
    source: str | Path,
    load_weights: bool = True,
    key: str = "ema_model",
    return_config: bool = False,
    snapshot_iter: int | None = None,
) -> nn.Module | tuple[nn.Module, modelConfig]:
    """
    Load a model from a given source.

    Parameters:
    ----------
    source : str or Path
        The source of the model config. It can be a file or a directory if Path,
        or a model name if str.
    load_weights : bool, optional
        Whether to load the model weights. Defaults to True.
    key : str, optional
        The dict key to use when loading the model weights. Defaults to "ema_model".
    return_config : bool, optional
        Whether to return the model configuration. Defaults to False.
    snapshot_iter : int, optional
        The iteration number of the snapshot to load. Defaults to None.

    Returns:
    ----------
    model : nn.Module
        The loaded model.
    config : modelConfig, optional
        The model configuration, if `return_config` is True.

    Raises:
    ----------
    ValueError
        If 'source' is an invalid model identifier.
    FileNotFoundError
        If the specified snapshot file is not found.

    """

    # Parse model specifier
    _, model_dir, model_file, config_file = parse_model_specifier(source)

    # Load config and construct model
    logger.info(f"Loading model from {config_file}")
    config = modelConfig.from_preset(config_file)
    model = unet.EDMPrecond.from_config(config)

    # Load model weights
    if load_weights:

        # Load snapshot, if specified
        if snapshot_iter is not None:
            snapshot_file = model_dir / f"snapshots/snapshot_iter_{iter:08d}.pt"
            if not snapshot_file.exists():
                raise FileNotFoundError(f"Snapshot file {snapshot_file} not found.")
            logger.info(f"Loading snapshot from {snapshot_file}")
            model = load_parameters(model, snapshot_file, key=key)

        # Otherwise, load final model
        else:
            model = load_parameters(model, model_file, key=key)

    # Return desired output
    if return_config:
        return model, config
    else:
        return model


def load_parameters(
    model: nn.Module, path: str | Path, key: str = "ema_model"
) -> nn.Module:
    """
    Load model parameters from a file.

    Parameters
    ----------
    model : nn.Module
        Model to load parameters into.
    path : str or Path
        Path to the file containing the model parameters.
    key : str, optional
        The key to load the state_dict from parameters file. Defaults to "ema_model".

    Returns
    -------
    model: nn.Module
        Model with loaded parameters.
    """
    logger.info(f"Loading model parameters from {path}")

    # Import model weights from file
    state_dict = torch.load(path, map_location="cpu")[key]

    # If necessary, remove 'module.' from keys (e.g. for ema model)
    if key != "model":
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }

    # Load weights into model
    model.load_state_dict(state_dict)

    return model


def isModel(model: nn.Module, modelClass: type) -> bool:
    """
    Check whether a model is an instance of a given class.

    Parameters
    ----------
    model : nn.Module
        The model to check.
    modelClass : type
        The class to check against.

    Returns
    -------
    bool
        True if model is an instance of modelClass, False otherwise.
    """
    if isinstance(model, nn.DataParallel):
        return isinstance(model.module, modelClass)
    return isinstance(model, modelClass)


def load_data_transforms(source: str | Path):
    """
    Load data transforms from a file.

    Parameters
    ----------
    source : str or Path
        The source of the transforms. It can be a file or a directory if Path,
        or a model name if str.

    Returns
    -------
    transforms : dict
        The loaded data transforms.
    """
    # Parse model specifier
    model_name, model_dir, _, _ = parse_model_specifier(source)

    # Load transforms
    transforms_file = model_dir / f"data_transforms_{model_name}.pt"
    return torch.load(transforms_file)
