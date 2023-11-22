import os
import socket
from contextlib import closing
import logging
from pathlib import Path

import torch
import torch.distributed as dist

import model.configs as configs
from model.trainer import (
    LofarDiffusionTrainer, LofarParallelDiffusionTrainer, DummyDiffusionTrainer,
    FIRSTDiffusionTrainer
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
    # Paths
    result_parent = Path("/home/bbd0953/diffusion/results")

    # Hyperparameters
    conf = configs.EDM_small_config()
    conf.iterations = 200_000
    conf.log_interval = 250
    conf.model_name = "EDM_small_splitFix"

    # Class conditioning
    # conf.n_labels = 4
    # conf.label_dropout = 0.1
    # conf.pretrained_model = '/home/bbd0953/diffusion/results/EDM_small_SAFETY/snapshots/ema_iter_00020000.pt'
    # conf.optimizer_file = '/home/bbd0953/diffusion/results/EDM/optimizer_state_EDM.pt'


    pickup_path = Path(f"/home/bbd0953/diffusion/results/") / conf.model_name
    trainer = LofarDiffusionTrainer.from_pickup(pickup_path, iterations=305_000, config=conf)
    
    # trainer = LofarDiffusionTrainer(config=conf)
    trainer.training_loop()