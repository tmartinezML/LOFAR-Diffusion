import os
import socket
from contextlib import closing
import logging
from pathlib import Path

import torch
import torch.distributed as dist

from model.configs import InitModel_EDM_config, DummyConfig
from model.trainer import (
    LofarDiffusionTrainer, LofarParallelDiffusionTrainer, DummyDiffusionTrainer
)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '134.100.120.74'
    os.environ['MASTER_PORT'] = str(find_free_port())
    # initialize the process group
    print(f"Initializing DDP on rank {rank}.")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"DDP initialized on rank {rank}.")


def ddp_cleanup():
    dist.destroy_process_group()


def ddp_training(rank, world_size, conf):
    print(f"Running DDP training on rank {rank}.")
    ddp_setup(rank, world_size)
    print(f"DDP setup complete on rank {rank}.")

    # Create diffusion trainer
    trainer = LofarParallelDiffusionTrainer(config=conf, rank=rank)
    trainer.training_loop()

    ddp_cleanup()


if __name__ == "__main__":
    logging.info(
        "\n\n\n######################\n"
        "DDPM Workout\n"
        "######################\n"
        "Prepare training...\n"
    )

    # Hyperparameters
    conf = InitModel_EDM_config()
    conf.model_name = "InitModel_EDM_lr=2e-5_bsize=256"
    torch.autograd.set_detect_anomaly(True)

    """
    pickup_path = Path(
        "/home/bbd0953/diffusion/results/InitModel_EDM_lr=2e-5_bsize=256"
    )
    trainer = LofarDiffusionTrainer.from_pickup(pickup_path, iterations=200_000)
    """
    trainer = LofarDiffusionTrainer(config=conf)
    trainer.training_loop()
