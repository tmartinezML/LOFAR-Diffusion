import logging
import csv
import json
from inputimeout import inputimeout, TimeoutOccurred
from datetime import datetime

import torch

from utils.paths import MODEL_PARENT


class OutputManager:

    def __init__(
        self, model_name, override=False, pickup=False, parent_dir=MODEL_PARENT
    ):
        self.model_name = model_name
        self.override = override
        self.pickup = pickup

        self.parent_dir = parent_dir
        self.results_folder = self.parent_dir.joinpath(self.model_name)

        if not (self.override or self.pickup):
            # Check if model name already exists, if so rename both model and
            # results folder.
            self._check_rename_model()

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.train_loss_file = self.results_folder.joinpath(
            f"losses_train_{self.model_name}.csv"
        )
        self.val_loss_file = self.results_folder.joinpath(
            f"losses_val_{self.model_name}.csv"
        )
        self.parameters_file = self.results_folder.joinpath(
            f"parameters_{self.model_name}.pt"
        )
        self.config_file = self.results_folder.joinpath(
            f"config_{self.model_name}.json"
        )
        self.power_ema_file = self.results_folder.joinpath(
            f"power_ema_{self.model_name}.pt"
        )

        self.out_files = [
            self.train_loss_file,
            self.val_loss_file,
            self.parameters_file,
            self.config_file,
            self.power_ema_file,
        ]

        self.log_file = self.results_folder.joinpath(
            f"training_log_{self.model_name}.log"
        )
        logging.basicConfig(
            format="%(levelname)s: %(message)s",
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

    def _check_rename_model(self):
        model_name = self.model_name
        results_folder = self.parent_dir.joinpath(model_name)
        if results_folder.exists():
            i = 1
            while results_folder.exists():
                model_name = f"{model_name}_{i}"
                results_folder = self.parent_dir.joinpath(self.model_name)
                i += 1
            logging.warning(
                f"Model name {model_name} already exists."
                f" Renaming to {self.model_name}."
            )
            self.model_name = model_name
            self.results_folder = results_folder

    def _loss_files_exist(self):
        exist = [f.exists() for f in [self.train_loss_file, self.val_loss_file]]
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
                logging.warning(f"Overriding files for model {self.model_name}.\n")
                try:
                    inputimeout(prompt="Press enter to continue...", timeout=10)
                except TimeoutOccurred:
                    logging.info("No key was pressed - training aborted.\n")
                    raise SystemExit
                break
        self._init_loss_file(self.train_loss_file)
        self._init_loss_file(
            self.val_loss_file, columns=["iteration", "loss", "ema_loss"]
        )

    def init_training_loop_pickup(self):
        assert (
            self._loss_files_exist()
        ), f"Loss files for model {self.model_name} do not exist."

    def init_training_loop(self):
        if self.pickup:
            self.init_training_loop_pickup()
        else:
            self.init_training_loop_create()

    def _write_loss(self, file, iteration, loss):
        with open(file, "a") as f:
            self._writer(f).writerow([iteration, f"{loss:.2e}"])

    def _write_losses(self, file, losses):
        with open(file, "a") as f:
            self._writer(f).writerows(losses)

    def write_train_losses(self, losses):
        self._write_losses(self.train_loss_file, losses)

    def write_val_losses(self, losses):
        self._write_losses(self.val_loss_file, losses)

    def save_params(
        self, model, ema_model, optimizer, power_ema_models, gammas, path=None
    ):
        path = path or self.parameters_file

        # Helper function
        def state_dict_or_none(obj):
            return obj.state_dict() if obj is not None else None

        # Save model, ema_model and optimizer state
        torch.save(
            {
                "model": state_dict_or_none(model),
                "ema_model": state_dict_or_none(ema_model),
                "optimizer": state_dict_or_none(optimizer),
                **{
                    f"power_ema_{gamma}": state_dict_or_none(model)
                    for model, gamma in zip(power_ema_models, gammas)
                },
            },
            path,
        )

    def save_snapshot(self, snap_name, model, ema_model, optimizer):
        snap_dir = self.results_folder.joinpath("snapshots")
        snap_dir.mkdir(parents=True, exist_ok=True)

        self.save_params(
            model, ema_model, optimizer, snap_dir.joinpath(f"snapshot_{snap_name}.pt")
        )

    def save_power_ema(self, models, t, gammas):
        power_ema_dir = self.results_folder.joinpath("power_ema")
        power_ema_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "time": t,
                **{
                    f"model_{gamma}": model.state_dict()
                    for model, gamma in zip(models, gammas)
                },
            },
            power_ema_dir.joinpath(f"power_ema_{t}.pt"),
        )

    def save_config(self, param_dict, iterations=None):
        if iterations is not None:
            param_dict["iterations"] = iterations
        with open(self.config_file, "w") as f:
            json.dump(param_dict, f, indent=4)

    def read_iter_count(self):
        with open(self.config_file, "r") as f:
            config = json.load(f)
        return config["iterations"]

    def log_training_progress(self, dt, t_per_it, i, i_tot, loss):
        logging.info(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"- Running {dt} "
            f"- Iteration {i+1} - Loss: {loss.item():.2e} "
            f"- {t_per_it * (i_tot - i - 1)} remaining "
            f"- {t_per_it} per it."
            f""
        )

    def log_val_loss(self, i, val_loss):
        logging.info(
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"- Iteration {i+1} - Validation loss: {val_loss[0]:.2e} "
            f"- Validation EMA loss: {val_loss[1]:.2e}"
        )
