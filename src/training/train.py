import os
import socket
from contextlib import closing
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

import model.configs as configs
from training.trainer import DiffusionTrainer
from datasets.firstgalaxydata import FIRSTGalaxyData
from utils.data_utils import TrainDataset, train_transform
from utils.device_utils import set_visible_devices
import utils.paths as paths


if __name__ == "__main__":
    # Limit visible GPUs
    set_visible_devices(2)

    # Hyperparameters
    conf = configs.EDM_small_config()

    # conf.pretrained_model = '/home/bbd0953/diffusion/model_results/Dummy/snapshots/snapshot_iter_00000100.pt'
    # conf.optimizer_file = '/home/bbd0953/diffusion/results/EDM_valFix/optimizer_state_EDM_valFix.pt'
    conf.model_name = f"PowerEMA"

    dataset = TrainDataset(
        paths.LOFAR_SUBSETS["0-clip"],
    )
    conf.training_data = str(dataset.path)
    conf.context = ["max_values_tr"]
    conf.context_dropout = 0.1
    conf.power_ema_snapshots = 20

    trainer = DiffusionTrainer(
        config=conf,
        dataset=dataset,
        pickup=False,
    )

    wandb.init(
        project="Diffusion",
        config=conf.param_dict,
        # id='fdlc49ai',
        # resume='must',
        dir=paths.ANALYSIS_PARENT / "wandb",
    )
    trainer.training_loop()

    """
    pickup_path = result_parent / conf.model_name
    trainer = DiffusionTrainer.from_pickup(
        path=pickup_path,
        config=conf,
        dataset=LofarZoomUnclipped80(),
    )
    """

    # Class conditioning
    # conf.n_labels = 4
    # conf.label_dropout = 0.1
    # conf.pretrained_model = '/home/bbd0953/diffusion/results/EDM_small_SAFETY/snapshots/ema_iter_00020000.pt'
    # conf.optimizer_file = '/home/bbd0953/diffusion/results/EDM/optimizer_state_EDM.pt'
