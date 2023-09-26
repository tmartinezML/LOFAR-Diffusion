from pathlib import Path
import torch.multiprocessing as mp
import logging
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.INFO)
import itertools

import torch

from utils.train_utils import (
    DiffusionTrainer, LofarDiffusionTrainer
)
from utils.model_utils import modelConfig

def OptiModel_Initial_config():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "OptiModel_Initial",
        use_improved_unet = False,
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        # Diffusion parameters
        timesteps = 250,
        schedule = "linear",
        learn_variance = False,
        # Training parameters
        batch_size = 128,
        iterations = 10000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 100,
        write_output = True,
        override_files = True,
        # Parallel training
        train_parallel = False,
        n_devices = 3,
    )
    return conf

def OptiModel_ImprovedUnet_config():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "OptiModel_ImprovedUnet_Default",
        model_type = "ImprovedUnet",
        use_improved_unet = True,
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        # Diffusion parameters
        timesteps = 250,
        schedule = "cosine",
        learn_variance = True,
        # Training parameters
        batch_size = 128,
        iterations = 10000,
        learning_rate = 3e-5,
        ema_rate = 0.9999,
        log_interval = 250,
        val_every = 500,
        write_output = True,
        override_files = True,
        optimizer = "Adam",
        # Parallel training
        n_devices = 3,
    )
    return conf

def InitModel_EDM_config():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "InitModel_EDM",
        model_type = "EDMPrecond",
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        dropout = 0.1,
        # Diffusion parameters
        timesteps = 1000,
        learn_variance = False,
        # Training parameters
        batch_size = 128,
        iterations = 60_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 250,
        val_every = 250,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 3,
    )
    return conf


if __name__ == "__main__":
    logging.info(
        "\n\n\n######################\n" \
        "DDPM Workout\n" \
        "######################\n" \
        "Prepare training...\n"
    )

    # Hyperparameters
    conf = InitModel_EDM_config()
    conf.n_devices = 1

    LofarDiffusionTrainer(config=conf).training_loop()
