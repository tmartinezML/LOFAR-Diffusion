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
from torch_ema import ExponentialMovingAverage as EMA
from torch.cuda.amp import autocast, GradScaler

import model.unet as unet
from model.diffusion import Diffusion
from model.edm_diffusion import EDM_Diffusion
from utils.data_utils import LofarSubset, LofarDummySet, load_data
from utils.device_utils import visible_gpus_by_space
from utils.init_utils import load_config_from_path


class outputManager():

    def __init__(self, model_name,
                 override=False, pickup=False, parent_dir=None):
        self.model_name = model_name
        self.override = override
        self.pickup=pickup

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
        self.log_file = self.results_folder.joinpath(
            f"training_log_{self.model_name}.log"
        )

        self.files = [
            self.train_loss_file, self.val_loss_file,
            self.pars_model_file, self.pars_ema_file,
            self.ema_state_file, self.optim_state_file,
            self.config_file, self.log_file,
        ]
    
    def _loss_files_exist(self):
        exist = [f.exists() for f in [self.train_loss_file, self.val_loss_file]]
        return all(exist)

    def _writer(self, f):
        return csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONE)

    def _init_loss_file(self, file, columns=["iteration", "loss"]):
        with open(file, "w") as f:
            self._writer(f).writerow(columns)

    def init_training_loop_create(self):
        for f in self.files:
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

        logging.basicConfig(
            format='%(levelname)s: %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.log_file,
                                    mode=('a' if self.log_file.exists() else 'w')),
                logging.StreamHandler(),
            ],
            force=True,
        )

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

    def save_snapshot(self, snap_name, model, ema):
        snap_dir = self.results_folder.joinpath("snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)
        # Save model
        torch.save(model.state_dict(),
                   snap_dir.joinpath(f"model_{snap_name}.pt"))
        # Save EMA
        with ema.average_parameters():
            torch.save(model.state_dict(),
                       snap_dir.joinpath(f"ema_{snap_name}.pt"))

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


def nan_hook(self, inp, out):
    outputs = out if isinstance(out, tuple) else [out]
    inputs = inp if isinstance(inp, tuple) else [inp]

    has_nan = lambda x: torch.isnan(x).any()
    layer_name = self.__class__.__name__

    for i, x in enumerate(inputs):
        if x is not None and has_nan(x):
            raise RuntimeError(f"{layer_name} input {i} has NaNs")
    
    for i, x in enumerate(outputs):
        if x is not None and has_nan(x):
            raise RuntimeError(f"{layer_name} output {i} has NaNs")
        

class DiffusionTrainer:
    def __init__(self, *, config, dataset, device=None, parent_dir=None, 
                 pickup=False):
        # Initialize config & class attributes
        self.config = config
        self.loss_type = self.config.loss_type
        self.validate_ema = False
        if hasattr(self.config, "validate_ema"):
            self.validate_ema = self.config.validate_ema

        # Initialize device
        device_ids_by_space = visible_gpus_by_space()
        self.device = device or (
            torch.device("cuda", device_ids_by_space[0])
            if torch.cuda.is_available()
            else "cpu"
        )
        logging.info(f"Working on: {self.device}")

        # Initialize model & EMA
        unetModel = getattr(unet, self.config.model_type)
        self.model = unetModel.from_config(self.config)
        for layer in self.model.model.modules():
            layer.register_forward_hook(nan_hook)
        self.inner_model = self.model
        self.model.to(self.device)
        if self.config.n_devices > 1:
            # Initialize parallel training
            dev_ids = device_ids_by_space[:self.config.n_devices]
            logging.info(f"Parallel training on multiple GPUs: \
                        {dev_ids}.")
            self.model.to(f'cuda:{dev_ids[0]}')  # Necessary for DDP
            self.model = DataParallel(
                self.model,
                device_ids=dev_ids
            )
            self.inner_model = self.model.module
        self.ema = EMA(self.model.parameters(), decay=self.config.ema_rate)

        # Initialize diffusion
        diffusion_class = (
            EDM_Diffusion if self.config.model_type == "EDMPrecond"
            else Diffusion
        )
        self.diffusion = diffusion_class.from_config(self.config)

        # Initialize data
        self.config.batch_size = int(self.config.batch_size)
        self.train_set = dataset
        self.val_every = (
            self.config.val_every if hasattr(self.config, "val_every")
            else self.config.log_interval
        )
        if self.val_every:
            # Split dataset into train and validation sets
            self.train_set, self.val_set = random_split(
                dataset, [0.1, 0.9]
            )
            assert len(self.val_set) >= self.config.batch_size, \
                f"Batch size {self.config.batch_size} larger than validation set."
            self.val_loader = DataLoader(self.train_set,
                                         batch_size=self.config.batch_size,
                                         shuffle=False, num_workers=1,
                                         drop_last=True)

        assert len(self.train_set) >= self.config.batch_size, \
            f"Batch size {self.config.batch_size} larger than training set."
        self.train_data = load_data(self.train_set, self.config.batch_size)

        # Initialize optimizer
        self.optimizer = None
        self.init_optimizer()

        # Initialize output manager
        self.OM = outputManager(self.config.model_name,
                                override=self.config.override_files,
                                parent_dir=parent_dir, pickup=pickup)
        
        self.iter_start = 0
        if pickup:
            self.iter_start = self.OM.read_iter_count()
            logging.info(f"Starting training at iteration {self.iter_start}.")
        
    def init_optimizer(self):
        if hasattr(self.config, "optimizer"):
            self.optimizer = getattr(optim, self.config.optimizer)(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate)
    
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

    def load_optimizer(self):
        if self.OM.optim_state_file.exists():
            self.optimizer.load_state_dict(
                torch.load(self.OM.optim_state_file, map_location='cpu')
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
            f"Training for {iterations:_} iterations - "\
            f"Starting from {self.iter_start:_} - "\
            f"Remaining iterations {iterations - self.iter_start:_}"
        )
        for i in range(self.iter_start, iterations):
            # Perform training step
            loss = self.training_step(scaler)
            loss_buffer.append([i + 1, loss.item()])

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

            # Save snapshot
            if self.config.snapshot_interval and \
                    write_output and \
                    save_model and \
                    (i + 1) % self.config.snapshot_interval == 0:
                logging.info(f"Saving snapshot at iteration {i+1}...")
                OM.save_snapshot(f"iter_{i+1:08d}", self.inner_model, self.ema)

        logging.info(f"Training time {dt()} - Done!")

    def training_step(self, scaler):
        self.optimizer.zero_grad()
        batch = next(self.train_data).to(self.device)
        with autocast():
            loss = self.batch_loss(batch)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.ema.update()
        return loss

    def validation_loss(self, eval_ema=False):
        # Validate model
        self.model.eval()
        with torch.no_grad():
            losses = []
            ema_losses = []
            for batch in self.val_loader:
                losses.append(self.batch_loss(batch).item())
                if eval_ema:
                    ema_losses.append(self.batch_loss_ema(batch).item())
        output = [torch.Tensor(l).mean().item() for l in [losses, ema_losses]]
        self.model.train()
        return output

    def batch_loss(self, batch):
        batch = batch.to(self.device)
        with autocast():
            if isinstance(self.diffusion, EDM_Diffusion):
                loss = self.diffusion.edm_loss(self.model, batch)
            else:
                t = torch.randint(0, self.diffusion.timesteps,
                                  (batch.shape[0],),
                                  device=self.device).long()
                loss = self.diffusion.p_losses(self.model, batch, t,
                                               loss_type=self.loss_type)
        return loss

    def batch_loss_ema(self, batch):
        with self.ema.average_parameters():
            loss = self.batch_loss(batch)
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
