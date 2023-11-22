import os
import tempfile
import logging

from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
import torch
from torch.cuda.amp import GradScaler
from math import isnan

from model.configs import InitModel_EDM_config, DummyConfig
from utils.device_utils import set_visible_devices
from model.trainer import (
    LofarDiffusionTrainer, DummyDiffusionTrainer
)


def train_wrapper(parameter_space, pretrained=None):
    # Set configuration
    model_conf = GLOBAL_CONF
    model_conf.update(parameter_space)
    model_conf.n_devices = N_GPU if USE_DDP else 1
    model_conf.validate_ema = True
    device = torch.device('cuda')

    # Load trainer
    torch.manual_seed(42)
    trainer = GLOBAL_TRAINER_CLASS(config=model_conf,
                                   device=device)

    # Set iterations
    iterations = model_conf.iterations
    i = 0

    # Load pretrained
    if pretrained:
        print('Loading pretrained model:\n', pretrained)
        trainer.load_state_dict(
            torch.load(pretrained, map_location=device)
        )

    # Load checkpoint
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt")
            )
        trainer.model.load_state_dict(checkpoint_dict["model"])
        trainer.ema.load_state_dict(checkpoint_dict["ema"])
        trainer.optimizer.load_state_dict(checkpoint_dict["optimizer"])

        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = parameter_space["learning_rate"]

        i = checkpoint_dict["iteration"]

    # Custom training loop
    scaler = GradScaler()
    while i < iterations:
        # Report initial metrics before training
        if i == 0:
            val_loss = trainer.validation_loss(eval_ema=True)
            metrics = {
                "loss": 0,
                "val_loss": val_loss[0],
                "val_ema_loss": val_loss[1],
                'training_iteration': 0,
                'nan': False,
            }
            train.report(metrics)

        # Train
        loss = trainer.training_step(scaler)

        # Report metrics
        if (i + 1) % REPORT_INTERVAL == 0:
            val_loss = trainer.validation_loss(eval_ema=True)
            is_nan = any([isnan(val) for val in (loss.item(), *val_loss)])
            metrics = {
                "loss": loss.item(),
                "val_loss": val_loss[0],
                "val_ema_loss": val_loss[1],
                'training_iteration': i + 1,
                'nan': is_nan,
            }

            # Possibly save checkpoint
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                with tempfile.TemporaryDirectory() as tmpdir:
                    torch.save({
                        "model": trainer.model.state_dict(),
                        "ema": trainer.ema.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "iteration": i + 1,
                    }, os.path.join(tmpdir, "checkpoint.pt"))
                    checkpoint = Checkpoint.from_directory(tmpdir)
                    train.report(metrics, checkpoint=checkpoint)
            else:
                train.report(metrics)

        i += 1


def run_tune_BOHB(parameter_space, name, n_samples, 
                  max_t=10_000, pretrained=None):
    algo = tune.search.ConcurrencyLimiter(
        TuneBOHB(), max_concurrent=4
    )
    scheduler = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=max_t,
        reduction_factor=3,

    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_wrapper, pretrained=pretrained),
            resources={"cpu": N_CPU, "gpu": (N_GPU if USE_DDP else 1)}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            metric="val_loss",
            mode="min",
            num_samples=n_samples,
            search_alg=algo,
        ),
        run_config=train.RunConfig(
            storage_path=STORAGE_PARENT,
            name=name,
            stop={"nan": True},
        ),
        param_space=parameter_space,
    )
    results = tuner.fit()
    return results


if __name__ == "__main__":
    # Set up logger to output to file,
    # and to always print timestamp
    logfile = "/home/bbd0953/diffusion/results/tune/tune.log"
    logger = logging.getLogger("tune")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s"
    )
    fh.setFormatter(formatter)

    # Global setup
    GLOBAL_CONF = InitModel_EDM_config()
    GLOBAL_CONF.channel_mults = (1, 2, 2, 2)
    GLOBAL_TRAINER_CLASS = LofarDiffusionTrainer
    STORAGE_PARENT = "/home/bbd0953/diffusion/results/tune"
    PRETRAINED_PARENT = "/home/bbd0953/diffusion/results/InitModel_EDM_lr=2e-5_bsize=256_(1,2,2,2)/snapshots"

    # Device setup
    N_CPU = 16
    N_GPU = 2
    USE_DDP = True
    set_visible_devices(N_GPU)

    # Parameter space
    parameter_space = {
        "learning_rate": tune.qloguniform(1e-5, 1e-3, 1e-5),
        # "batch_size": tune.choice([128, 256]),
    }

    # Optimization 1: From scratch
    if False:
        logger.info("Running: Optimization 1 (From scratch)")
        GLOBAL_CONF.iterations = 1_000
        REPORT_INTERVAL = 100
        CHECKPOINT_INTERVAL = 200
        run_tune_BOHB(
            parameter_space,
            name="lr_(1,2,2,2)_fromScratch",
            n_samples=30,
        )
        logger.info("Finished: Optimization 1 (From scratch)")

    # Optimization 2: From pretrained 5k
    logger.info("Running: Optimization 2 (From pretrained 5k)")
    pretrained_cp = (
        PRETRAINED_PARENT + '/model_iter_00005000.pt'
    )
    GLOBAL_CONF.iterations = 6_000
    REPORT_INTERVAL = 600
    CHECKPOINT_INTERVAL = REPORT_INTERVAL
    run_tune_BOHB(
        parameter_space,
        name="lr_(1,2,2,2)_pretrained-05k",
        max_t = GLOBAL_CONF.iterations,
        n_samples=27,
        pretrained = pretrained_cp,
    )
    logger.info("Finished: Optimization 2 (From pretrained 5k)")

    # Optimization 3: From pretrained 15k
    logger.info("Running: Optimization 3 (From pretrained 15k)")
    pretrained_cp = (
        PRETRAINED_PARENT + '/model_iter_00015000.pt'
    )
    GLOBAL_CONF.iterations = 15_000
    REPORT_INTERVAL = 1000
    CHECKPOINT_INTERVAL = REPORT_INTERVAL
    run_tune_BOHB(
        parameter_space,
        name="lr_(1,2,2,2)_pretrained-15k",
        n_samples=27,
        max_t = GLOBAL_CONF.iterations,
        pretrained = pretrained_cp,
    )
    logger.info("Finished: Optimization 3 (From pretrained 15k)")
