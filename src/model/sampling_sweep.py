from inspect import signature
from functools import partial

import numpy as np
import torch
import wandb

from model.diffusion import Diffusion
from model.sample import sample_batch
from analysis.model_evaluation import get_distributions
import utils.paths as paths
import utils.stats_utils as stats
from utils.init_utils import load_model_from_folder
from utils.device_utils import set_visible_devices, distribute_model


def get_kwarg_names(fn):
    sig = signature(fn)
    return [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY]


def sampling_eval(config, model, counts_lofar):

    # Sampling setup
    n_devices = config['n_devices']
    n_samples = config['n_samples']

    batch_size = 1000 * n_devices
    n_batches = int(n_samples / batch_size)
    print(f"Sampling {n_batches} batches.")

    # Diffusion instance
    diffusion = Diffusion(
        timesteps=config['timesteps'],
    )

    # Distribute model
    model, _ = distribute_model(model, n_devices=n_devices)

    # Sample kwargs
    # (passed to diffusion.edm_sampling)
    # Explanation: Extract all kwargs from config that are valid for
    #              diffusion.edm_sampling
    sample_kwargs = {
        k: v for k, v in config.items()
        if k in get_kwarg_names(diffusion.edm_sampling)
    }
    sample_kwargs.pop('batch_size')

    # Sample from model
    batch_list = []
    for i in range(n_batches):
        batch = sample_batch(
            model,
            diffusion,
            bsize=batch_size,
            return_steps=False,
            **sample_kwargs,
        )
        batch_list.append(batch)

    img_batch = torch.cat(batch_list)

    # Clamp to [0, 1]
    img_batch = torch.clamp((img_batch + 1) / 2, 0, 1)

    # Evaluate samples
    img_batch = img_batch.squeeze().numpy()
    counts, _ = np.histogram(img_batch, bins=np.linspace(0, 1, 257))
    W1, W1_err = stats.W1_distance(counts, counts_lofar)

    # Clear GPU
    model = (
        model.module.to('cpu') if isinstance(model, torch.nn.DataParallel)
        else model.to('cpu')
    )

    return W1, W1_err, counts


def sampling_wrapper(config=None, model=None, counts_lofar=None):
    with wandb.init(config=config):
        W1, W1_err, counts = sampling_eval(
            wandb.config, model, counts_lofar
        )
        wandb.log({'counts': counts, 'W1': W1, 'W1_err': W1_err})

        # Add plot
        centers = np.linspace(0, 1, 256)
        centers = (centers[1:] + centers[:-1]) / 2
        '''
        data = [
            [ce, co] for ce, co in zip(centers, stats.norm(counts))
        ]
        table = wandb.Table(
            data=data, columns=["Bin Center", "Frequency"]
        )
        '''
        wandb.log({
            'Pixel_Distribution_Logplot': wandb.plot.line_series(
                xs=centers,
                ys=[np.log(stats.norm(counts)), np.log(stats.norm(counts_lofar))],
                keys=["Sampled", "LOFAR"],
                title="Pixel Distribution",
                xname="Pixel Intensity",
            )
        })


def run_sweep(config, model_path, lofar_path):
    model = load_model_from_folder(model_path)
    counts_lofar, _ = get_distributions(lofar_path)['Pixel_Intensity']

    sweep_id = wandb.sweep(config, project="Diffusion")
    wandb.agent(
        sweep_id,
        function=partial(
            sampling_wrapper,
            model=model,
            counts_lofar=counts_lofar
        ),
    )


if __name__=='__main__':

    sweep_config = {
        'method': 'bayes',
        'name': 'Deterministic Sampling Sweep',
        'metric': {
            'name': 'W1',
            'goal': 'minimize',
        },
        'parameters': {
            'timesteps': {
                'values': [
                    25
                ],
            },
            'sigma_min': {  # Original 2e-3
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-2,
            },
            'sigma_max': {  # Original 80
                'distribution': 'log_uniform_values',
                'min': 10,
                'max': 500,
            },
            'rho': {  # Original 7
                'distribution': 'int_uniform',
                'min': 1,
                'max': 10,
            },
            'batch_size': {
                'value':1000,
            },
            'n_samples': {
                'value': 1000,
            },
            'n_devices': {
                'value': N_GPU,
            },
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
        },
    }

    lofar_path = paths.LOFAR_SUBSETS['120asLimit_SNR>=5']
    model_path = paths.MODEL_PARENT / 'EDM_SNR5_120as'
    run_sweep(sweep_config, model_path, lofar_path)


