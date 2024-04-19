from pathlib import Path
import csv
import logging
import json
from datetime import datetime
from copy import deepcopy


import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, SequentialLR
from torch_ema import ExponentialMovingAverage as EMA
import wandb

import model.unet as unet
import utils.train_utils
import training.loss_functions as loss_functions
from training.output_manager import OutputManager
from datasets.firstgalaxydata import FIRSTGalaxyData
from utils.device_utils import visible_gpus_by_space
from utils.init_utils import load_config_from_path, load_parameters
from utils.paths import MODEL_PARENT, DEBUG_DIR
from utils.data_utils import load_data
from utils.train_utils import use_ema


class DiffusionTrainer:
    def __init__(
        self,
        *,
        config,
        dataset,
        device=None,
        pickup=False,
        model_name=None,
        iterations=None,
        parent_dir=MODEL_PARENT,
    ):
        # Initialize config & class attributes
        if config is None:
            assert pickup, "Config must be specified if not pickup."
            assert iterations is not None, (
                "Iterations must be specified if no config is passed, "
                "else no more training will happen."
            )
            assert model_name is not None, (
                "Model name must be specified if no config is passed, "
                "else no files can be found."
            )
            config = load_config_from_path(parent_dir / model_name)
        if iterations is not None:
            config.iterations = iterations
        self.config = config
        self.validate_ema = self.config.validate_ema

        # Initialize output manager
        self.OM = OutputManager(
            self.config.model_name,
            override=self.config.override_files,
            parent_dir=parent_dir,
            pickup=pickup,
        )
        self.iter_start = 0
        if pickup:
            self.iter_start = self.OM.read_iter_count()
            logging.info(f"Starting training at iteration {self.iter_start}.")

        # Initialize device
        device_ids_by_space = visible_gpus_by_space()
        self.device = device or torch.device("cuda", device_ids_by_space[0])
        logging.info(f"Working on: {self.device}")

        # Initialize Model
        self.model = unet.EDMPrecond.from_config(self.config)
        # Load state dict of pretrained model if specified
        if self.config.pretrained_model:
            load_parameters(
                self.model,
                self.config.pretrained_model,
                use_ema=True,
            )
            logging.info(
                f"Loaded pretrained ema model from: \
                  \n\t{self.config.pretrained_model}"
            )
        self.inner_model = self.model
        self.model.to(self.device)

        # Initialize parallel training
        if self.config.n_devices > 1:
            dev_ids = device_ids_by_space[: self.config.n_devices]
            logging.info(f"Parallel training on multiple GPUs: {dev_ids}.")
            self.model.to(f"cuda:{dev_ids[0]}")  # Necessary for DataParallel
            self.model = DataParallel(self.model, device_ids=dev_ids)
            self.inner_model = self.model.module

        # Initialize EMA model
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.inner_model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                self.config.ema_rate
            ),
        )

        # Initialize power-ema models
        self.power_ema_gammas = [16.97, 6.94]
        self.power_ema_models = [
            torch.optim.swa_utils.AveragedModel(
                self.inner_model,
                avg_fn=utils.train_utils.get_power_ema_avg_fn(gamma),
            )
            for gamma in self.power_ema_gammas
        ]

        # Initialize data
        self.dataset = dataset
        if hasattr(self.config, "context"):
            logging.info(f"Working with context: {self.config.context}.")
            if "max_values_tr" in self.config.context:
                self.dataset.transform_max_vals()
            self.dataset.set_context(*self.config.context)
        self.config.batch_size = int(self.config.batch_size)
        self.val_every = (
            self.config.val_every
            if hasattr(self.config, "val_every")
            else self.config.log_interval
        )
        self.init_data_sets(split=bool(self.val_every))

        # Initialize optimizer & LR scheduler
        self.optimizer = None
        self.init_optimizer()
        if False:
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    StepLR(self.optimizer, step_size=100_000, gamma=1),
                    StepLR(self.optimizer, step_size=5_000, gamma=0.8),
                ],
                milestones=[100_000],
            )

        if pickup:
            logging.info(
                f"Picking up model, EMA and optimizer from {self.OM.model_name}."
            )
            self.load_state()

    def init_data_sets(self, split=True):
        self.train_set = self.dataset
        # Split dataset into train and validation sets
        if split:
            # Manual seed for reproducibility of results
            generator = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = random_split(
                self.dataset, [0.9, 0.1], generator=generator
            )
            assert (
                len(self.val_set) >= self.config.batch_size
            ), f"Batch size {self.config.batch_size} larger than validation set."
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=True,
            )

        assert (
            len(self.train_set) >= self.config.batch_size
        ), f"Batch size {self.config.batch_size} larger than training set."
        self.train_data = load_data(self.train_set, self.config.batch_size)

    def init_optimizer(self):
        if hasattr(self.config, "optimizer"):
            self.optimizer = getattr(optim, self.config.optimizer)(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )

        if (
            hasattr(self.config, "optimizer_file")
            and self.config.optimizer_file is not None
        ):
            logging.info(
                "Loading optimizer state from:" f"\n\t{self.config.optimizer_file}"
            )
            self.load_optimizer(self.config.optimizer_file)

    @classmethod
    def from_pickup(self, path, config=None, iterations=None, **kwargs):
        assert (
            config is not None or iterations is not None
        ), "Either config or iterations must be specified for pickup."

        if config is None:
            config = load_config_from_path(path)

        if iterations is not None:
            config.iterations = iterations

        trainer = self(config=config, pickup=True, **kwargs)

        return trainer

    def read_parameters(self, key):
        return torch.load(self.OM.parameters_file, map_location="cpu")[key]

    def load_model(self):
        self.inner_model.load_state_dict(self.read_parameters("model"))

    def load_ema(self):
        self.ema_model.load_state_dict(self.read_parameters("ema_model"))

    def load_optimizer(self):
        self.optimizer.load_state_dict(self.read_parameters("optimizer"))

    def load_state(self):
        self.load_model()
        self.load_ema()
        self.load_optimizer()

    def training_loop(
        self,
        iterations=None,
        write_output=None,
        OM=None,
        save_model=True,
        train_logging=True,
    ):
        # Prepare output handling
        write_output = write_output or self.config.write_output
        if write_output:
            OM = OM or self.OM
            OM.init_training_loop()
        else:
            logging.basicConfig(
                format="%(levelname)s: %(message)s",
                level=logging.INFO if train_logging else logging.CRITICAL,
                force=True,
            )
            logging.warning("No output files will be written.\n")

        # Prepare training
        iterations = iterations or self.config.iterations
        scaler = GradScaler()
        loss_buffer = []
        t0 = datetime.now()
        power_ema_interval = iterations // self.config.power_ema_snapshots

        def dt():
            return datetime.now() - t0

        # Print start info
        logging.info(
            f"Starting training loop at {t0.strftime('%H:%M:%S')}...\n"
            f"Training for {iterations:_} iterations - "
            f"Starting from {self.iter_start:_} - "
            f"Remaining iterations {iterations - self.iter_start:_}"
        )

        # Training loop
        for i in range(self.iter_start, iterations):

            # Perform training step
            loss = self.training_step(scaler)
            loss_buffer.append([i + 1, loss.item()])

            # Log loss to wandb
            wandb.log({"loss": loss.item()}, step=i + 1)

            if (i + 1) % self.config.log_interval == 0:

                # Log progress
                t_per_it = dt() / (i + 1 - self.iter_start)
                self.OM.log_training_progress(dt(), t_per_it, i, iterations, loss)

                # Write output
                if write_output:
                    self.log_step_write_output(OM, save_model, loss_buffer, i)

            # Calculate validation loss, log & write
            if self.val_every and (i + 1) % self.val_every == 0:

                # Calculate & log validation loss
                val_loss = self.validation_loss(eval_ema=self.validate_ema)
                self.OM.log_val_loss(i, val_loss)

                # Write output
                if write_output:
                    OM.write_val_losses([[i + 1, *val_loss]])

                # Log val loss to wandb
                wandb.log(
                    {"val_loss": val_loss[0], "val_loss_ema": val_loss[1]}, step=i + 1
                )

            # Save snapshot
            if (
                self.config.snapshot_interval
                and write_output
                and save_model
                and (i + 1) % self.config.snapshot_interval == 0
            ):
                logging.info(f"Saving snapshot at iteration {i+1}...")
                OM.save_snapshot(
                    f"iter_{i+1:08d}", self.inner_model, self.ema_model, self.optimizer
                )

            # Save power ema models
            if (i + 1) % power_ema_interval == 0:
                logging.info(f"Saving power ema models at iteration {i+1}...")
                OM.save_power_ema(self.power_ema_models, i + 1, self.power_ema_gammas)

        logging.info(f"Training time {dt()} - Done!")

    def handle_batch(self, batch):
        context, labels = None, None
        if isinstance(batch, list):
            match len(batch):
                case 2 if self.inner_model.model.context_dim:
                    batch, context = batch

                case 2 if not self.inner_model.model.context_dim:
                    batch, labels = batch

                case 3:
                    batch, context, labels = batch

                case _:
                    raise ValueError(
                        "Batch must be a list of length 2 or 3, " f"not {len(batch)}."
                    )
        return batch, context, labels

    def training_step(self, scaler):
        # Zero gradients
        self.optimizer.zero_grad()

        # Get batch
        batch, context, labels = self.handle_batch(next(self.train_data))

        # Calculate loss
        with autocast():
            loss = self.batch_loss(batch, context=context, labels=labels)

        # Backward pass & optimizer step
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        if hasattr(self, "scheduler"):
            self.scheduler.step()

        # Update EMA model
        self.ema_model.update_parameters(self.inner_model)

        # Update power ema models
        for power_ema_model in self.power_ema_models:
            power_ema_model.update_parameters(self.inner_model)

        return loss

    def validation_loss(self, eval_ema=None):
        eval_ema = eval_ema or self.validate_ema
        # Validate model
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            losses = []
            ema_losses = []

            # Loop through all batches in validation set
            for batch in self.val_loader:

                # Set class labels if present
                batch, context, labels = self.handle_batch(batch)

                # Calculate loss
                losses.append(
                    self.batch_loss(batch, context=context, labels=labels).item()
                )

                # Calculate EMA loss
                if eval_ema:
                    with use_ema(self.inner_model, self.ema_model):
                        ema_losses.append(
                            self.batch_loss(
                                batch, context=context, labels=labels
                            ).item()
                        )

        # Return mean loss
        output = [torch.Tensor(l).mean().item() for l in [losses, ema_losses]]
        self.model.train()
        self.ema_model.train()

        return output

    def batch_loss(self, batch, context=None, labels=None):

        # Move input to gpu
        batch = batch.to(self.device)
        if context is not None:
            context = context.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Calculate loss
        with autocast():
            loss = loss_functions.edm_loss(
                self.model,
                batch,
                context=context,
                class_labels=labels,
                sigma_data=self.config.sigma_data,
                P_mean=self.config.P_mean,
                P_std=self.config.P_std,
            )

        return loss

    def log_step_write_output(self, OM, save_model, loss_buffer, i):
        OM.write_train_losses(loss_buffer)
        if save_model:
            # Save model parameters, EMA parameters, EMA state & optimizer state
            OM.save_params(
                self.inner_model,
                self.ema_model,
                self.optimizer,
                self.power_ema_models,
                self.power_ema_gammas,
            )
        OM.save_config(self.config.param_dict, iterations=i + 1)
        loss_buffer.clear()


class ParallelDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, dataset, rank, parent_dir=None):
        config = deepcopy(config)
        config.n_devices = 1
        self.rank = rank
        super().__init__(
            config=config, dataset=dataset, device=rank, parent_dir=parent_dir
        )
        self.model = DDP(self.model, device_ids=[rank])
        self.ema = EMA(self.model.parameters(), decay=self.config.ema_rate)
        self.init_optimizer()

    def training_loop(
        self,
    ):
        rankzero = self.rank == 0
        kwargs = {
            "write_output": rankzero,
            "save_model": rankzero,
            "train_logging": rankzero,
        }
        return super().training_loop(**kwargs)


class FIRSTDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, **kwargs):
        assert (
            config.n_labels == 4
        ), f"FIRSTDiffusionTrainer supports exactly 4 labels, \
              {config.n_labels} given."
        super().__init__(
            config=config,
            dataset=FIRSTGalaxyData(root="/home/bbd0953/diffusion/image_data"),
            **kwargs,
        )
