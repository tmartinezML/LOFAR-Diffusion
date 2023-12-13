from functools import partial

import torch
from torch.cuda.amp import GradScaler
import wandb

from model.configs import EDM_small_config
from model.trainer import DiffusionTrainer, LofarDiffusionTrainer
from utils.data_utils import LofarZoomUnclipped80
from utils.device_utils import visible_gpus_by_space, set_visible_devices


def train_wrapper(sweep_config=None, pretrained=None, dev_ids=None):
    # Initialize wandb run
    with wandb.init(config=sweep_config):
        # Set configuration
        model_conf = GLOBAL_CONF
        model_conf.n_devices = N_GPU if USE_DP else 1
        model_conf.validate_ema = False

        # Set parameters from sweep_config
        model_conf.learning_rate = wandb.config.learning_rate

        # Set device
        if dev_ids is not None:
            set_visible_devices(dev_ids)

        dev_id = visible_gpus_by_space()[0]
        device = torch.device('cuda', dev_id)

        # Load trainer
        torch.manual_seed(42)
        trainer = GLOBAL_TRAINER_CLASS(config=model_conf, device=device)

        # Set iterations
        iterations = model_conf.iterations

        # Load pretrained
        if pretrained:
            print('Loading pretrained model:\n', pretrained)
            trainer.load_state_dict(
                torch.load(pretrained, map_location=device)['model']
            )

        # Custom training loop
        scaler = GradScaler()
        for i in range(iterations):

            # Train
            loss = trainer.training_step(scaler)
            if i == 0:
                loss_smoothed = loss.item()
            if i <= 5:
                # Running average
                loss_smoothed = (loss.item() + loss_smoothed * i) / (i + 1)
            else:
                alpha = 0.95
                loss_smoothed = loss_smoothed * alpha + loss.item() * (1 - alpha)

            # Report metrics
            if (i + 1) % REPORT_INTERVAL == 0:
                val_loss = trainer.validation_loss()
                metrics = {
                    "loss": loss.item(),
                    "val_loss": val_loss[0],
                    'training_iteration': i + 1,
                    "loss_smoothed": loss_smoothed,
                }
                wandb.log(metrics, step=i + 1, commit=True)

            else:
                wandb.log(
                    {'loss': loss.item(), 'loss_smoothed': loss_smoothed},
                    step=i + 1,
                    commit=False
                )

        # Remove everything
        trainer.model.to('cpu')


if __name__ == '__main__':
    # Set global variables
    GLOBAL_CONF = EDM_small_config()
    GLOBAL_TRAINER_CLASS = partial(
        DiffusionTrainer, dataset=LofarZoomUnclipped80()
    )
    N_GPU = 2
    DEV_IDS = set_visible_devices(N_GPU)
    print('Setting visible devices:', DEV_IDS)
    USE_DP = True

    GLOBAL_CONF.iterations = 2000
    REPORT_INTERVAL = 250
    pretrained = (
        '/home/bbd0953/diffusion/results/EDM_small_splitFix/snapshots/snapshot_iter_00010000.pt'
    )

    # Sweep config
    sweep_config = {
        'method': 'grid',
        'name': 'EDM_small_lr_grid',
        'metric': {
            'name': 'loss_smoothed',
            'goal': 'minimize',
        },
        'parameters': {
            'learning_rate': {
                'values': [
                    5e-6, 7e-6, 1e-5, 2e-5, 4e-5,
                ],
                # 'distribution': 'log_uniform_values',
                # 'min': 5e-6,
                # 'max': 8e-5,
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 2,
            # 's': 2,
            'eta': 2,
        },
    }

    # Sweep
    sweep_id = wandb.sweep(sweep_config, project='Diffusion')
    wandb.agent(
        sweep_id,
        function=partial(train_wrapper, pretrained=None, dev_ids=DEV_IDS),
        count=5
    )