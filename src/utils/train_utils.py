import subprocess
from io import StringIO
from pathlib import Path
import csv
import logging
import json
import time
from datetime import datetime

from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DataParallel
from torch_ema import ExponentialMovingAverage as EMA
import pandas as pd

from model.unet import Unet, ImprovedUnet
from model.diffusion import Diffusion
from utils.data_utils import LofarSubset, load_data

def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode()),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    # print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(
        lambda x: x.rstrip(' [MiB]')
    )
    gpu_df.sort_values(by='memory.free', inplace=True, ascending=False)
    return list(gpu_df.index)

class outputManager():

    def __init__(self, model_name,
                 override=False, parent_dir=None):
        self.model_name = model_name
        self.override = override

        self.parent_dir = parent_dir or Path("/storage/tmartinez/results")
        self.results_folder = self.parent_dir.joinpath(self.model_name)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.train_loss_file = self.results_folder.joinpath(
            f"losses_train_{self.model_name}.csv"
        )
        self.val_loss_file = self.results_folder.joinpath(
            f"losses_val_{self.model_name}.csv"
        )
        self.model_file = self.results_folder.joinpath(
            f"parameters_model_{self.model_name}.pt"
        )
        self.ema_file = self.results_folder.joinpath(
            f"parameters_ema_{self.model_name}.pt"
        )
        self.config_file = self.results_folder.joinpath(
            f"config_{self.model_name}.json"
        )
        self.log_file = self.results_folder.joinpath(
            f"training_log_{self.model_name}.log"
        )
    
    def _writer(self, f):
        return csv.writer(f, delimiter=";", quoting=csv.QUOTE_NONE)

    def _init_loss_file(self, file):
        with open(file, "w") as f:
            self._writer(f).writerow(
                ["iteration", "loss"]
            )

    def init_training_loop(self):
        for f in [self.train_loss_file, self.model_file, self.ema_file,
                    self.hyperparam_file]:
            if not self.override and f.exists():
                raise FileExistsError(f"File {f} already exists.")
            if self.override and f.exists():
                logging.warning(
                    f"Overriding existing files for model {self.model_name}.\n"\
                    "Continuing in...")
                for _ in tqdm(range(10), bar_format="{bar}{remaining}", ncols=20):
                    time.sleep(1)
                break
        self._init_loss_file(self.train_loss_file)
        self._init_loss_file(self.val_loss_file)
        logging.basicConfig(
            format='%(levelname)s: %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.log_file, mode='w'),
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

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_file)

    def save_ema(self, model, ema):
        with ema.average_parameters():
                torch.save(model.state_dict(), self.ema_file)

    def save_config(self, param_dict, iterations=None):
        if iterations is not None:
            param_dict["iterations"] = iterations
        with open(self.config_file, "w") as f:
            json.dump(param_dict, f, indent=4)


class DiffusionTrainer:
    def __init__(self, *, config, dataset, device=None, parent_dir=None):
        self.config = config
        self.loss_type = "huber" if not self.config.learn_variance else "hybrid"
        # Get free GPU
        device_ids_by_space = get_free_gpu()
        self.device = device or (
            torch.device("cuda", device_ids_by_space[0])
            if torch.cuda.is_available()
            else "cpu"
        )
        logging.info(f"Working on: {self.device}")

        # Initialize model
        unetModel = ImprovedUnet if self.config.use_improved_unet else Unet
        self.model = unetModel.from_config(self.config).to(self.device)

        if self.config.n_devices > 1:
            logging.info(f"Parallel training on multiple GPUs: \
                        {device_ids_by_space[:self.config.n_devices]}.")
            self.model.to('cuda:0')  # Necessary for DDP
            self.model = DataParallel(
                self.model, 
                device_ids=device_ids_by_space[:self.config.n_devices]
            )
        
        self.model_params = self.model.parameters()

        # Initialize data, diffusion and training instances
        self.diffusion = Diffusion(
            self.config.timesteps, 
            schedule=self.config.schedule,
            learn_variance=self.config.learn_variance
        )

        self.train_set = dataset
        self.val_every = (
            self.config.val_every if hasattr(self.config, "val_every") else 0
        )
        if self.val_every:
            self.train_set, self.val_set = random_split(
                dataset, [0.1, 0.9]
            )
            assert len(self.val_set) > self.config.batch_size, \
            "Validation set size must be larger than batch size."
            self.val_loader = DataLoader(self.train_set, 
                                       batch_size=self.config.batch_size,
                                       shuffle=False, num_workers=1,
                                       drop_last=True)
        
        assert len(self.train_set) > self.config.batch_size, \
            "Training set size must be larger than batch size."
        self.train_data = load_data(self.train_set, self.config.batch_size)

        if hasattr(self.config, "optimizer"):
            self.optimizer = getattr(optim, self.config.optimizer)(
                self.model_params, lr=self.config.learning_rate
            )
        else:
            self.optimizer = optim.Adam(self.model_params,
                                        lr=self.config.learning_rate)
            
        self.ema = EMA(self.model_params, decay=self.config.ema_rate)
        self.OM = outputManager(self.config.model_name,
                                override=self.config.override_files, 
                                parent_dir=parent_dir)
        
    def validation_loss(self,):
        # Validate model
        self.model.eval()
        with torch.no_grad():
            losses = []
            for batch in self.val_loader:
                batch = batch.to(self.device)
                t = torch.randint(0, self.diffusion.timesteps, (batch.shape[0],),
                                device=self.device).long()
                loss = self.diffusion.p_losses(self.model, batch, t,
                                                loss_type=self.loss_type)
                losses.append(loss.item())
        return torch.Tensor(losses).mean().item()

    def training_loop(self,
                      iterations=None, write_output=None, OM=None, 
                      save_model=True):
        
        iterations = iterations or self.config.iterations
        write_output = write_output or self.config.write_output

        if write_output:
            OM = OM or self.OM
            OM.init_training_loop()
        else:
            logging.warning("Not writing output to files.\n")

        # Start training loop
        t0 = datetime.now()
        def dt(): return datetime.now() - t0

        loss_buffer = []
        logging.info(f"Starting training loop at {t0.strftime('%H:%M:%S')}...")
        for i in range(iterations):
            self.optimizer.zero_grad()
            batch = next(self.train_data).to(self.device)
            
            # Algorithm 1, line 3: sample t uniformally for every example in
            # the batch.
            t = torch.randint(0, self.diffusion.timesteps, (batch.shape[0],),
                            device=self.device).long()

            try:
                loss = self.diffusion.p_losses(self.model, batch, t,
                                            loss_type=self.loss_type)
            except RuntimeError:
                print(f"Error at iteration {i}, batch shape {batch.shape}")
                raise

            loss_buffer.append([i, loss.item()])

            loss.backward()
            self.optimizer.step()
            self.ema.update()

            if (i + 1) % self.config.log_interval == 0:
                logging.info(
                    f"{datetime.now().strftime('%H:%M:%S')} "\
                    f"- Running {dt()} "\
                    f"- Iteration {i+1} - Loss: {loss.item():.2e} "\
                    f"- {dt() * (iterations - i - 1) / (i + 1)} remaining "\
                    f"- {dt() / (i + 1)} per it."
                )
                if write_output:
                    OM.write_train_losses(loss_buffer)
                    loss_buffer = []
                    # Save model, such that we can early stop at any time 
                    # without losing progress.
                    if save_model:
                        OM.save_model(self.model)
                        OM.save_ema(self.model, self.ema)
                    OM.save_config(self.config.hparam_dict, 
                                            iterations=i+1)
                    
            if self.val_every and (i+1) % self.val_every == 0:
                val_loss = self.validation_loss()
                logging.info(
                    f"{datetime.now().strftime('%H:%M:%S')} "\
                    f"- Iteration {i+1} - Validation loss: {val_loss:.2e} "\
                )
                if write_output:
                    OM.write_val_losses([[i+1, val_loss]])

        logging.info(f"Training time {dt()} - Done!")

class LofarDiffusionTrainer(DiffusionTrainer):
    def __init__(self, *, config, device=None, parent_dir=None):
        super().__init__(config=config, dataset=LofarSubset(),
                         device=device, parent_dir=parent_dir)