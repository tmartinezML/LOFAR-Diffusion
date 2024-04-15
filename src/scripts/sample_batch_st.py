from datetime import datetime
from pathlib import Path

import torch
import numpy as np

from model.sample import sample_batch
from utils.init_utils import (
    load_model_from_folder, load_snapshot, load_old_model_from_folder
)
from utils.device_utils import distribute_model, set_visible_devices
from utils.paths import MODEL_PARENT, ANALYSIS_PARENT

# File name number formatting


def fmt(v):
    match v:
        case int() if not v % 1000:
            return f'{v / 1000:.0f}k'
        case int():
            return str(v)
        case float() if v >= 1:
            return f'{v:.2f}'
        case float():
            return f'{v:.2e}'


def batch_st_sampling(
    model_name,
    n_samples=4000,
    n_devices=1,
    context_fn=None,  # Call signature!
    label=None,
    comment=None,
    snapshot_it=0,
    use_ema=True,
    sample_kwargs={},
    out_folder_name=None,
):

    # Set paths
    out_folder = ANALYSIS_PARENT / (out_folder_name or model_name)
    out_folder.mkdir(exist_ok=True)
    model_dir = MODEL_PARENT / model_name

    # Sampling setup
    batch_size = 1000 * n_devices
    n_batches = max(int(n_samples / batch_size), 1)
    print(f"Sampling {n_batches} batches.")

    # Load model
    if snapshot_it:
        model = load_snapshot(model_dir, snapshot_it, use_ema=use_ema)
    else:
        model = load_model_from_folder(model_dir, use_ema=use_ema)
    model, _ = distribute_model(model, n_devices=n_devices)

    # Output file for samples
    outfile_specifier = (
        f"{batch_size * n_batches}"
        + '_' * bool(len(sample_kwargs))
        + '_'.join(f'{k}={fmt(v)}' for k, v in sample_kwargs.items())
    )
    if snapshot_it:
        outfile_specifier = f'it={snapshot_it}_' + outfile_specifier
    if comment:
        outfile_specifier = f'{comment}_{outfile_specifier}'
    out_file = (
        out_folder /
        f"{model_dir.name}_samples_{outfile_specifier}.pt"
    )

    # Sample from model
    batch_list = []
    context_list = []
    for i in range(n_batches):
        print(f"Sampling batch {i+1}/{n_batches}...")

        # Generate context vector:
        context_batch = context_fn(batch_size) if context_fn else None
        context_list.append(context_batch)

        batch = sample_batch(
            model,
            bsize=batch_size,
            context_batch=torch.tensor(context_batch).reshape(batch_size, -1),
            return_steps=True,
            **sample_kwargs
        )
        batch_list.append(batch)

    model = (
        model.module.to('cpu') if isinstance(model, torch.nn.DataParallel)
        else model.to('cpu')
    )

    # Output of sample_batch is list with T+1 entries of shape
    # (bsize, 1, 80, 80).
    # Batch_list is a list of such lists with n_batches entries,
    # i.e. n_batches x (T+1) x (bsize, 1, 80, 80).
    # We want it as a single tensor of shape
    # (n_batches * bsize, T+1, 1, 80, 80).
    batch_st = torch.concat([
        torch.stack(b, dim=1) for b in batch_list
    ]).cpu()

    # Scale images from [-1, 1] to [0, 1]
    batch_st = (batch_st + 1) / 2
    torch.save(batch_st, out_file)

    # Save context
    if context_fn is not None:
        np.save(
            out_folder / f"{out_file.stem.replace('samples', 'context')}",
            np.concatenate(context_list)
        )

    # Release GPU memory
    del model, batch_st
    torch.cuda.empty_cache()


def sample_snapshot_loop(model_name, snapshot_its: list, n_samples=4000,
                         n_devices=1, timesteps=25,
                         comment=None, sample_kwargs={}, ):
    # Set paths
    out_folder = ANALYSIS_PARENT / model_name / 'snapshot_samples'
    out_folder.mkdir(exist_ok=True)
    model_dir = MODEL_PARENT / model_name

    # Sampling setup
    diffusion.timesteps = timesteps
    batch_size = 1000 * n_devices
    n_batches = int(n_samples / batch_size)
    print(f"Sampling {n_batches } batches.")

    for it in snapshot_its:
        print(f"Sampling from snapshot {it}...")

        # Load model and diffusion
        model = load_snapshot(model_dir, it)
        model, _ = distribute_model(model, n_devices=n_devices)

        # Output file for samples
        outfile_specifier = (
            f"{batch_size * n_batches}"
            + '_' * bool(len(sample_kwargs))  # Add '_' if kwargs are present
            + '_'.join(f'{k}={fmt(v)}' for k, v in sample_kwargs.items())
        )
        outfile_specifier = f'it={it}_' + outfile_specifier
        if comment:
            outfile_specifier = f'{comment}_{outfile_specifier}'
        out_file = (
            out_folder /
            f"{model_dir.name}_samples_{outfile_specifier}.pt"
        )

        # Sample from model
        batch_list = []
        for i in range(n_batches):
            print(f"Sampling batch {i+1}/{n_batches}...")
            batch = sample_batch(model,
                                 bsize=batch_size,
                                 return_steps=False, **sample_kwargs)
            batch_list.append(batch)

        # Here, output of sample_batch is a tensor of shape (bsize, 1, 80, 80).
        # Batch_list is a list of such tensors with n_batches entries,
        # i.e. n_batches x (bsize, 1, 80, 80).
        # We want it as a single tensor of shape (n_batches * bsize, 1, 80, 80).
        batch_st = torch.concat(batch_list).cpu()

        # Scale images from [-1, 1] to [0, 1]
        batch_st = (batch_st + 1) / 2

        # Save to file
        torch.save(batch_st, out_file)

        # Release GPU memory
        model = (
            model.module.to('cpu') if isinstance(model, torch.nn.DataParallel)
            else model.to('cpu')
        )

    # Release GPU memory
    del model, diffusion, batch_st
    torch.cuda.empty_cache()


if __name__ == "__main__":
    from utils.data_utils import ImagePathDataset
    import utils.paths as paths
    from scipy.special import boxcox, inv_boxcox
    from scipy.stats import norm, rv_histogram
    from sklearn import preprocessing as pr

    # Load dataset to get max-val histogram
    dset = ImagePathDataset(paths.LOFAR_SUBSETS['0-clip_unscaled'])
    max_vals = dset.max_values.numpy()
    max_hist = np.histogram(max_vals, bins=100)

    # Apply power-transform to make distribution more gaussian
    boxcox_lambda = -0.22659119
    max_tr = boxcox(max_vals, boxcox_lambda)
    max_tr_sc = pr.scale(max_tr)

    # Make model distribution from histogram for sampling
    hist_tr_sc = np.histogram(max_tr_sc, bins=100)
    model_dist = rv_histogram(hist_tr_sc, density=False)
    def context_fn(bsize): return model_dist.rvs(size=bsize)

    # Set up devices
    n_gpu = 2
    dev_ids = set_visible_devices(n_gpu)
    print(f"Using GPU {dev_ids[:n_gpu]}")

    # Sampling parameters
    model_name = 'Fmax_Context_Dropout'
    n_samples = 10_000

    batch_st_sampling(
        model_name,
        n_samples=n_samples,
        n_devices=n_gpu,
        # context_fn=context_fn,
        # sample_kwargs={'guidance_strength': 0.1,}
        out_folder_name="model_name/unconditioned"
    )

    print(f"Finished sampling {n_samples:_} samples from {model_name}.")

    '''

    snapshot_its = [20_000 * n for n in range(1, 5)]
    n_samples = 4_000

    sample_snapshot_loop(
        model_name, snapshot_its, n_devices=n_gpu, n_samples=n_samples,
    )
    print(f"Finished sampling {n_samples:_} samples from {model_name}.")
    '''
