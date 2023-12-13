from pathlib import Path
import csv
import logging
import json
from datetime import datetime
from copy import deepcopy

from inputimeout import inputimeout, TimeoutOccurred
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
from model.diffusion import EDM_Diffusion
from utils.data_utils import LofarSubset, LofarDummySet, load_data
from datasets.firstgalaxydata import FIRSTGalaxyData
from utils.device_utils import visible_gpus_by_space
from utils.init_utils import load_config_from_path


class outputManager():

    def __init__(self, model_name,
                 override=False, pickup=False, parent_dir=None):
        self.model_name = model_name
        self.override = override
        self.pickup = pickup

        self.parent_dir = parent_dir or Path("/storage/tmartinez/results")
        self.results_folder = self.parent_dir.joinpath(self.model_name)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.train_loss_file = self.results_folder.joinpath(
            f"losses_train_{self.model_name}.csv"
        )
        self.val_loss_file = self.results_folder.joinpath(
            f"losses_val_{self.model_name}.csv"
        )
        self.pars_model_file = self.results_folder.joinpath(
            f"parameters_model_{self.model_name}.pt"
        )
        self.pars_ema_file = self.results_folder.joinpath(
            f"parameters_ema_{self.model_name}.pt"
        )
        self.ema_state_file = self.results_folder.joinpath(
            f"ema_state_{self.model_name}.pt"
        )
        self.optim_state_file = self.results_folder.joinpath(
            f"optimizer_state_{self.model_name}.pt"
        )
        self.config_file = self.results_folder.joinpath(
            f"config_{self.model_name}.json"
        )

        self.out_files = [
            self.train_loss_file, self.val_loss_file,
            self.pars_model_file, self.pars_ema_file,
            self.ema_state_file, self.optim_state_file,
            self.config_file,
        ]

        self.log_file = self.results_folder.joinpath(
            f"training_log_{self.model_name}.log"
        )
        logging.basicConfig(
            format='%(levelname)s: %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    self.log_file,
                    mode="a" if self.log_file.exists() else "w",
                ),
                logging.StreamHandler(),
            ],
            force=True,
        )

    def _loss_files_exist(self):
        exist = [f.exists()
                 for f in [self.train_loss_file, self.val_loss_file]]
        return all(exist)

    def _writer(self, f):
        return csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONE)

    def _init_loss_file(self, file, columns=["iteration", "loss"]):
        with open(file, "w") as f:
            self._writer(f).writerow(columns)

    def init_training_loop_create(self):
        for f in self.out_files:
            if not self.override and f.exists():
                raise FileExistsError(f"File {f} already exists.")
            if self.override and f.exists():
                logging.warning(
                    f"Overriding files for model {self.model_name}.\n")
                try:
                    inputimeout(
                        prompt="Press enter to continue...", timeout=10)
                except TimeoutOccurred:
                    logging.info("No key was pressed - training aborted.\n")
                    raise SystemExit
                break
        self._init_loss_file(self.train_loss_file)
        self._init_loss_file(self.val_loss_file,
                             columns=["iteration", "loss", "ema_loss"])

    def init_training_loop_pickup(self):
        assert self._loss_files_exist(), \
            f"Loss files for model {self.model_name} do not exist."

    def init_training_loop(self):
        if self.pickup:
            self.init_training_loop_pickup()
        else:
            self.init_training_loop_create()

    def _write_loss(self, file, iteration, loss):
        with open(file, "a") as f:
            self._writer(f).writerow(
                [iteration, f"{loss:.2e}"]
            )

    def _write_losses(self, file, losses):
        with open(file, "a") as f:
            self._writer(f).writerows(losses)

    def write_train_losses(self, losses):
        self._write_losses(self.train_loss_file, losses)

    def write_val_losses(self, losses):
        self._write_losses(self.val_loss_file, losses)

    def save_params_model(self, model):
        torch.save(model.state_dict(), self.pars_model_file)

    def save_snapshot(self, snap_name, model, ema, optimizer):
        snap_dir = self.results_folder.joinpath("snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        with ema.average_parameters():
            ema_params = model.state_dict()

        torch.save({
            "model": model.state_dict(),
            "ema_params": ema_params,
            "ema_state": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
            snap_dir.joinpath(f"snapshot_{snap_name}.pt")
        )

    def save_params_ema(self, model, ema):
        with ema.average_parameters():
            torch.save(model.state_dict(), self.pars_ema_file)

    def save_ema_state(self, ema):
        torch.save(ema.state_dict(), self.ema_state_file)

    def save_optim_state(self, optim):
        torch.save(optim.state_dict(), self.optim_state_file)

    def save_config(self, param_dict, iterations=None):
        if iterations is not None:
            param_dict["iterations"] = iterations
        with open(self.config_file, "w") as f:
            json.dump(param_dict, f, indent=4)

    def read_iter_count(self):
        with open(self.config_file, "r") as f:
            config = json.load(f)
        return config["iterations"]


DEBUG_DIR = Path('/home/bbd0953/diffusion/results/debug')


class DiffusionTrainer:
    def __init__(self, *, config, dataset, device=None, parent_dir=None,
                 pickup=False):
        # Initialize config & class attributes
        self.config = config
        self.validate_ema = self.config.validate_ema

        # Initialize output manager
        self.OM = outputManager(self.config.model_name,
                                override=self.config.override_files,
                                parent_dir=parent_dir, pickup=pickup)
        self.iter_start = 0
        if pickup:
            self.iter_start = self.OM.read_iter_count()
            logging.info(f"Starting training at iteration {self.iter_start}.")

        # Initialize device
        device_ids_by_space = visible_gpus_by_space()
        self.device = device or torch.device("cuda", device_ids_by_space[0])
        logging.info(f"Working on: {self.device}")

        # Initialize Model
        unetClass = getattr(unet, self.config.model_type)
        self.model = unetClass.from_config(self.config)
        # Load state dict of pretrained model if specified
        if self.config.pretrained_model:
            self.model.load_state_dict(
                torch.load(self.config.pretrained_model, map_location='cpu')
            )
            logging.info(
                f"Loaded pretrained model from: \
                  \n\t{self.config.pretrained_model}"
            )
        self.inner_model = self.model
        self.model.to(self.device)
        # Initialize parallel training
        if self.config.n_devices > 1:
            dev_ids = device_ids_by_space[:self.config.n_devices]
            logging.info(
                f"Parallel training on multiple GPUs: {dev_ids}."
            )
            self.model.to(f'cuda:{dev_ids[0]}')  # Necessary for DataParallel
            self.model = DataParallel(self.model, device_ids=dev_ids)
            self.inner_model = self.model.module

        # Initialize EMA
        self.ema = EMA(self.model.parameters(), decay=self.config.ema_rate)

        # Initialize diffusion
        self.diffusion = EDM_Diffusion.from_config(self.config)

        # Initialize data
        self.dataset = dataset
        self.config.batch_size = int(self.config.batch_size)
        self.val_every = (
            self.config.val_every if hasattr(self.config, "val_every")
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
                milestones=[100_000]
            )

    def init_data_sets(self, split=True):
        self.train_set = self.dataset
        # Split dataset into train and validation sets
        if split:
            # Manual seed for reproducibility of results
            generator = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = random_split(
                self.dataset, [0.9, 0.1], generator=generator
            )
            assert len(self.val_set) >= self.config.batch_size, \
                f"Batch size {self.config.batch_size} larger than validation set."
            self.val_loader = DataLoader(self.val_set,
                                         batch_size=self.config.batch_size,
                                         shuffle=False, num_workers=1,
                                         drop_last=True)

        assert len(self.train_set) >= self.config.batch_size, \
            f"Batch size {self.config.batch_size} larger than training set."
        self.train_data = load_data(self.train_set, self.config.batch_size)

    def init_optimizer(self):
        if hasattr(self.config, "optimizer"):
            self.optimizer = getattr(optim, self.config.optimizer)(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate)

        if hasattr(self.config, "optimizer_file") and \
                self.config.optimizer_file is not None:
            logging.info(f"Loading optimizer state from: \
                            \n\t{self.config.optimizer_file}")
            self.load_optimizer(self.config.optimizer_file)

    @classmethod
    def from_pickup(self, path, config=None, iterations=None, **kwargs):
        assert config is not None or iterations is not None, \
            "Either config or iterations must be specified for pickup."
        if config is None:
            config = load_config_from_path(path)
        if iterations is not None:
            config.iterations = iterations
        trainer = self(config=config, pickup=True, **kwargs)
        trainer.load_model()
        trainer.load_ema()
        trainer.load_optimizer()
        return trainer

    def load_model(self):
        self.inner_model.load_state_dict(
            torch.load(self.OM.pars_model_file, map_location='cpu')
        )

    def load_ema(self):
        if self.OM.ema_state_file.exists():
            self.load_ema_from_ema_state()
        else:
            logging.warning(
                "EMA state file not found. Loading from model parameters.")
            self.load_ema_from_model_dict()

    def load_ema_from_model_dict(self):
        dummy_model = deepcopy(self.inner_model)
        dummy_model.load_state_dict(
            torch.load(self.OM.pars_ema_file, map_location='cpu')
        )
        self.ema.shadow_params = [
            p.clone().detach() for p in dummy_model.parameters()
        ]
        self.ema.num_updates = self.iter_start
        del dummy_model

    def load_ema_from_ema_state(self):
        self.ema.load_state_dict(
            torch.load(self.OM.ema_state_file, map_location='cpu')
        )

    def load_optimizer(self, file=None):
        file = file or self.OM.optim_state_file
        if type(file) == str:
            file = Path(file)

        if file.exists():
            self.optimizer.load_state_dict(
                torch.load(file, map_location='cpu')
            )
        else:
            logging.warning(
                "Optimizer state file not found. Using new optimizer."
            )

    def load_state_dict(self, state_dict):
        self.inner_model.load_state_dict(state_dict)
        self.ema = EMA(self.model.parameters(), decay=self.config.ema_rate)
        self.init_optimizer()

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.ema.to(device)
        return self

    def training_loop(self,
                      iterations=None, write_output=None, OM=None,
                      save_model=True, train_logging=True):
        # Prepare output handling
        if write_output is None:
            write_output = self.config.write_output
        if write_output:
            OM = OM or self.OM
            OM.init_training_loop()
        else:
            logging.basicConfig(
                format='%(levelname)s: %(message)s',
                level=logging.INFO if train_logging else logging.CRITICAL,
                force=True,
            )
            logging.warning("No output files will be written.\n")

        # Prepare training
        iterations = iterations or self.config.iterations
        scaler = GradScaler()
        loss_buffer = []
        t0 = datetime.now()
        def dt(): return datetime.now() - t0

        # Start training loop
        logging.info(
            f"Starting training loop at {t0.strftime('%H:%M:%S')}...\n"
            f"Training for {iterations:_} iterations - "
            f"Starting from {self.iter_start:_} - "
            f"Remaining iterations {iterations - self.iter_start:_}"
        )
        for i in range(self.iter_start, iterations):

            # Perform training step
            loss = self.training_step(scaler)
            loss_buffer.append([i + 1, loss.item()])
            wandb.log({"loss": loss.item()}, step=i + 1)

            # Log progress & write output
            if (i + 1) % self.config.log_interval == 0:
                t_per_it = dt() / (i + 1 - self.iter_start)
                logging.info(
                    f"{datetime.now().strftime('%H:%M:%S')} "
                    f"- Running {dt()} "
                    f"- Iteration {i+1} - Loss: {loss.item():.2e} "
                    f"- {t_per_it * (iterations - i - 1)} remaining "
                    f"- {t_per_it} per it."
                    f""
                )
                if write_output:
                    self.log_step_write_output(OM, save_model, loss_buffer, i)

            # Calculate validation loss, log & write
            if self.val_every and (i + 1) % self.val_every == 0:
                val_loss = self.validation_loss(eval_ema=self.validate_ema)
                logging.info(
                    f"{datetime.now().strftime('%H:%M:%S')} "
                    f"- Iteration {i+1} - Validation loss: {val_loss[0]:.2e} "
                    f"- Validation EMA loss: {val_loss[1]:.2e}"
                )
                if write_output:
                    OM.write_val_losses([[i + 1, *val_loss]])
                wandb.log(
                    {"val_loss": val_loss[0], "val_loss_ema": val_loss[1]},
                    step=i + 1
                )

            # Save snapshot
            if self.config.snapshot_interval and \
                    write_output and \
                    save_model and \
                    (i + 1) % self.config.snapshot_interval == 0:
                logging.info(f"Saving snapshot at iteration {i+1}...")
                OM.save_snapshot(f"iter_{i+1:08d}",
                                 self.inner_model,
                                 self.ema,
                                 self.optimizer)

        logging.info(f"Training time {dt()} - Done!")

    def training_step(self, scaler):
        self.optimizer.zero_grad()
        batch = next(self.train_data)
        labels = None
        if isinstance(batch, list):
            batch, labels = batch

        with autocast():
            try:
                loss = self.batch_loss(batch, labels=labels)
            except (ValueError, RuntimeError) as e:
                print("Error in training step batch loss calculation.")
                torch.save(self.inner_model.state_dict(),
                           DEBUG_DIR.joinpath("model.pt"))
                torch.save(batch, DEBUG_DIR.joinpath("batch.pt"))
                raise e

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        self.ema.update()
        return loss

    def validation_loss(self, eval_ema=None):
        eval_ema = eval_ema or self.validate_ema
        # Validate model
        self.model.eval()
        with torch.no_grad():
            losses = []
            ema_losses = []

            # Loop through all batches in validation set
            for batch in self.val_loader:

                # Set class labels if present
                labels = None
                if isinstance(batch, list):
                    batch, labels = batch

                # Calculate loss
                losses.append(
                    self.batch_loss(batch, labels=labels).item()
                )

                # Calculate EMA loss
                if eval_ema:
                    ema_losses.append(
                        self.batch_loss_ema(batch, labels=labels).item()
                    )

        # Return mean loss
        output = [torch.Tensor(l).mean().item() for l in [losses, ema_losses]]
        self.model.train()

        return output

    def batch_loss(self, batch, labels=None):
        batch = batch.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        with autocast():
            loss = self.diffusion.edm_loss(
                self.model, batch, class_labels=labels
            )

        return loss

    def batch_loss_ema(self, batch, labels=None):
        with self.ema.average_parameters():
            loss = self.batch_loss(batch, labels=labels)
        return loss

    def log_step_write_output(self, OM, save_model, loss_buffer, i):
        OM.write_train_losses(loss_buffer)
        if save_model:
            # Save model parameters, EMA parameters, EMA state & optimizer state
            OM.save_params_model(self.inner_model)
            OM.save_params_ema(self.inner_model, self.ema)
            OM.save_ema_state(self.ema)
            OM.save_optim_state(self.optimizer)
        OM.save_config(self.config.param_dict, iterations=i + 1)
        loss_buffer.clear()


class LofarDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, **kwargs):
        super().__init__(config=config, dataset=LofarSubset(),
                         **kwargs)


class DummyDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, **kwargs):
        config.batch_size = 1
        super().__init__(config=config, dataset=LofarDummySet(),
                         **kwargs)


class ParallelDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, dataset, rank, parent_dir=None):
        config = deepcopy(config)
        config.n_devices = 1
        self.rank = rank
        super().__init__(config=config, dataset=dataset, device=rank,
                         parent_dir=parent_dir)
        self.model = DDP(self.model, device_ids=[rank])
        self.ema = EMA(self.model.parameters(), decay=self.config.ema_rate)
        self.init_optimizer()

    def training_loop(self,):
        rankzero = self.rank == 0
        kwargs = {
            'write_output': rankzero,
            'save_model': rankzero,
            'train_logging': rankzero,
        }
        return super().training_loop(**kwargs)


class LofarParallelDiffusionTrainer(ParallelDiffusionTrainer):
    def __init__(self, *, config, rank, parent_dir=None):
        super().__init__(config=config, dataset=LofarSubset(),
                         rank=rank, parent_dir=parent_dir)


class FIRSTDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, **kwargs):
        assert config.n_labels == 4, \
            f"FIRSTDiffusionTrainer supports exactly 4 labels, \
              {config.n_labels} given."
        super().__init__(
            config=config,
            dataset=FIRSTGalaxyData(root="/home/bbd0953/diffusion/image_data"),
            **kwargs)
