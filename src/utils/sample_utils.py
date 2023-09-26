import json

import torch

import model.unet as unet
from model.diffusion import Diffusion
from utils.model_utils import modelConfig
from utils.train_utils import get_free_gpu

def load_model(config_file, model_file=None):
    # Load model config
    config = load_config(config_file)

    # Load model
    model = getattr(unet, config.model_type).from_config(config)
    
    # Load model weights
    if model_file is not None:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
    
    return model

def load_config(config_file):
    # Load model config
    with open(config_file, "r") as f:
        config = json.load(f)
    
    return modelConfig(**config)