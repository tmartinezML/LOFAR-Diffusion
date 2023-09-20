from pathlib import Path
import torch.multiprocessing as mp
import logging
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.INFO)
import itertools

import torch

from utils.train_utils import (
    DiffusionTrainer, modelConfig, LofarDiffusionTrainer
)

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


if __name__ == "__main__":
    logging.info(
        "\n\n\n######################\n" \
        "DDPM Workout\n" \
        "######################\n" \
        "Prepare training...\n"
    )
    # mp.set_start_method('forkserver')

    # Hyperparameters
    conf = OptiModel_ImprovedUnet_config()
    conf.n_devices = 1

    # Try different channel configurations:
    parent_dir = Path("/storage/tmartinez/results/channel_configs")
    mults = [(1, 2, 2, 2), (1, 2, 3, 4), (1, 2, 4, 8)]
    inits = [80, 160, 240]
    for mult, init in itertools.product(mults, inits):
        conf.model_name = f"channel_configs_{mult}_{init}"
        dev = torch.device("cuda", 0)
        trainer_kwargs = {
            "device": dev,
            "parent_dir": parent_dir,
        }
        LofarDiffusionTrainer(conf, **trainer_kwargs).train()
