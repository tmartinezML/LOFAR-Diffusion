import os
import tempfile

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


def run_tune_Bayes_ASHA(parameter_space, name, algo, n_samples, max_iter, n_gpu):
    algo = BayesOptSearch(random_search_steps=4)
    scheduler = ASHAScheduler(
        max_t=max_iter,
        grace_period=1000,
        reduction_factor=2
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_wrapper),
            resources={"cpu": N_CPU, "gpu": n_gpu}
        ),
        tune_config=tune.TuneConfig(
            metric="val_ema_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=n_samples,
            search_alg=algo,
        ),
        run_config=train.RunConfig(
            storage_path=STORAGE_PARENT,
            name=name,
        ),
        param_space=parameter_space,
    )
    results = tuner.fit()
    return results


def run_tune_PBT(parameter_space, name, n_samples):
    scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=CHECKPOINT_INTERVAL,  # Recommended
        metric='val_ema_loss',
        mode='min',
        hyperparam_mutations=parameter_space,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_wrapper),
            resources={"cpu": N_CPU, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=n_samples,
        ),
        run_config=train.RunConfig(
            storage_path=STORAGE_PARENT,
            name=name,
            stop={"nan": True}
        ),
        param_space=parameter_space,
    )
    results = tuner.fit()
    return results


def run_tune_BOHB(parameter_space, name, n_samples, pretrained=None):
    algo = tune.search.ConcurrencyLimiter(
        TuneBOHB(), max_concurrent=4
    )
    scheduler = HyperBandForBOHB(
        time_attr='training_iteration',
        max_t=10_000,
        reduction_factor=2,

    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_wrapper, pretrained=pretrained),
            resources={"cpu": N_CPU, "gpu": (N_GPU if USE_DDP else 1)}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            metric="val_ema_loss",
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
    GLOBAL_CONF = InitModel_EDM_config()
    GLOBAL_CONF.iterations = 100_000
    GLOBAL_TRAINER_CLASS = LofarDiffusionTrainer
    STORAGE_PARENT = "/home/bbd0953/diffusion/results/tune"
    REPORT_INTERVAL = 3_000
    CHECKPOINT_INTERVAL = 10 * REPORT_INTERVAL
    N_CPU = 16
    N_GPU = 2
    USE_DDP = True
    name = "lr_bsize_BOHB_pretrained-20kIter"
    n_samples = 30
    # Set visible devices
    set_visible_devices(N_GPU)

    # Tune
    parameter_space = {
        "learning_rate": tune.quniform(1e-5, 1e-4, 1e-5),
        "batch_size": tune.choice([64, 128, 256]),
    }

    # Pretrained checkpoint (optional)
    pretrained_cp = (
        '/home/bbd0953/diffusion/results/'
        'InitModel_EDM_SnapshotRun/snapshots/ema_iter_20000.pt'
    )

    # Debug:
    # train_wrapper({"learning_rate": 1e-4, "batch_size": 16}, pretrained_cp)

    # Check if name exists, if so add suffix (prevent overwriting)
    new_name = name
    suffix = 00
    while os.path.exists(os.path.join(STORAGE_PARENT, new_name)):
        new_name = f"{name}_{suffix:02d}"
        suffix += 1
    name = new_name

    # Run tune
    results = run_tune_BOHB(
        parameter_space,
        name,
        n_samples=n_samples,
        pretrained=pretrained_cp,
    )
