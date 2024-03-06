import os
import socket
from contextlib import closing
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

import model.configs as configs
from model.trainer import (
    LofarDiffusionTrainer, LofarParallelDiffusionTrainer, DummyDiffusionTrainer,
    FIRSTDiffusionTrainer, DiffusionTrainer
)
from datasets.firstgalaxydata import FIRSTGalaxyData
from utils.data_utils import LofarZoomUnclipped80, TrainDataset, train_transform
from utils.device_utils import set_visible_devices
import utils.paths as paths


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
    # Limit visible GPUs
    set_visible_devices(2)

    # Hyperparameters
    conf = configs.EDM_small_config()

    # conf.pretrained_model = '/home/bbd0953/diffusion/results/EDM_valFix/parameters_ema_EDM_valFix.pt'
    # conf.optimizer_file = '/home/bbd0953/diffusion/results/EDM_valFix/optimizer_state_EDM_valFix.pt'
    conf.model_name = f"EDM_SNR5_50as"

    dataset = TrainDataset(paths.LOFAR_SUBSETS['unclipped_SNR>=5_50asLimit'])
    conf.training_data = str(dataset.path)

    trainer = DiffusionTrainer(
        config=conf,
        dataset=dataset,
        pickup=False,
        
    )

    wandb.init(
        project="Diffusion",
        config=conf.param_dict,
        # id='exscg7ib',
        # resume='must',
    )
    trainer.training_loop()

    '''
    pickup_path = result_parent / conf.model_name
    trainer = DiffusionTrainer.from_pickup(
        path=pickup_path,
        config=conf,
        dataset=LofarZoomUnclipped80(),
    )
    '''

    # Class conditioning
    # conf.n_labels = 4
    # conf.label_dropout = 0.1
    # conf.pretrained_model = '/home/bbd0953/diffusion/results/EDM_small_SAFETY/snapshots/ema_iter_00020000.pt'
    # conf.optimizer_file = '/home/bbd0953/diffusion/results/EDM/optimizer_state_EDM.pt'
