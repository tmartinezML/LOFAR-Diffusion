from pathlib import Path
import shutil
from datetime import datetime
import json
import random
from functools import partial
import warnings
from collections.abc import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm


from model.sample import sample_set_from_model
from utils.plot_utils import (
    plot_image_grid, plot_distributions,
    pixel_metrics_plot, shape_metrics_plot, plot_collection
)
from utils.device_utils import visible_gpus_by_space
from utils.data_utils import EvaluationDataset
from utils.stats_utils import norm, norm_err, cdf, cdf_err
import utils.paths as paths
from model.sample import sample_set_from_model
from analysis.fid_score import calculate_fid_given_paths, save_fid_stats
import analysis.image_metrics as imet

# Folder with lofar images
LOFAR_PATH = paths.LOFAR_SUBSETS['unclipped_H5']


def metrics_dict_from_data(img_data: Path | Dataset | str, n_bins: int = 4,
                           **h5_kwargs):
    match img_data:

        # Simple .pt file containing sampels as batch stack:
        case Path() if img_data.is_file() and img_data.suffix == ".pt":
            batch_st = torch.load(img_data, map_location='cpu')
            samples_itr = torch.clamp(batch_st[:, -1, :, :, :], 0, 1)

        # Dataset .h5 (or .hdf5) file, or image directory with .png files:
        case Path() if img_data.is_dir() or img_data.suffix in [".h5", ".hdf5"]:
            # The behavior for directory or .hdf5 file is handled within the
            # EvaluationDataset class.
            samples_itr = DataLoader(
                EvaluationDataset(img_data, **h5_kwargs), batch_size=None,
                shuffle=False, num_workers=1
            )

        # Dataset instance:
        case Dataset():
            samples_itr = DataLoader(
                img_data, batch_size=None, shuffle=False, num_workers=1
            )

        # Other type:
        case _:

            # Any iterable (list, array, DataLoader, etc.):
            if isinstance(img_data, Iterable):
                samples_itr = img_data

            # Unknown type:
            else:
                raise ValueError(f"Unknown image data type: {img_data}")

    return imet.metrics_dict_from_iterable(samples_itr, n_bins=n_bins)


def distributions_from_metrics(metrics_dict, bins: dict = {}):
    # Define bins for distributions
    bins_dict = {
        'Image_Mean': np.linspace(0, 1, 257),
        'Image_Sigma': np.linspace(0, 1, 257),
        'Active_Pixels': np.linspace(0, 6400, 257),
        'Bin_Pixels': np.linspace(0, 6400, 33),
        'Bin_Mean': np.linspace(0, 1, 129),
        'Bin_Sigma': np.linspace(0, 1, 129),
        'WPCA_Angle': np.linspace(0, 180, 19),  # 10 deg bins
        'WPCA_Elongation': np.linspace(0, 1, 11),
        'COM': np.arange(0, 81),
        'COM_Radius': np.linspace(0, 60, 31),
        'COM_Angle': np.linspace(0, 360, 37),
        'Scatter': np.linspace(0, 80, 257)
    }

    # Copy metrics dict and remove unnecessary keys
    metrics_dict = metrics_dict.copy()
    pixel_intensity = metrics_dict.pop('Pixel_Intensity', None)

    # Update bins dict with user-specified bins
    if len(bins):
        # Assert keys are contained in bins dict
        assert all([key in bins_dict.keys() for key in bins.keys()]), \
            f"Unknown key in bins dict:"\
            f" {set(bins.keys()) - set(bins_dict.keys())}"
        bins_dict.update(bins)

    # Assert keys of bins_dict and metrics_dict are the same
    assert set(bins_dict.keys()) == set(metrics_dict.keys()), \
        f"Keys of bins dict and metrics dict do not match:"\
        f" {set(bins_dict.keys()) ^ set(metrics_dict.keys())}"

    # Calculate distributions
    distributions_dict = {}
    print("Calculating metric distributions...")
    for key in tqdm(metrics_dict.keys(), total=len(metrics_dict.keys())):

        match key:
            # If it starts with 'Bin', it is a binned metric
            case k if k.startswith('Bin_') and isinstance(metrics_dict[k], dict):
                # Loop through dict of binned metrics
                sub_dict = {}
                for sub_key, sub_val in metrics_dict[key].items():
                    counts, edges = np.histogram(sub_val, bins=bins_dict[key])
                    sub_dict[sub_key] = counts, edges
                distributions_dict[key] = sub_dict

            case 'COM':
                counts, edges, _ = np.histogram2d(
                    *metrics_dict[key].T, bins=bins_dict[key]
                )
                distributions_dict[key] = counts, edges

            case _:
                counts, edges = np.histogram(
                    metrics_dict[key], bins=bins_dict[key]
                )
                distributions_dict[key] = counts, edges

    # Add pixel distribution
    distributions_dict['Pixel_Intensity'] = pixel_intensity

    return distributions_dict


def get_distributions(img_path,
                      force=False,
                      save=True,
                      h5_kwargs={},
                      metrics_n_bins: int = 4,
                      hist_bins: dict = {},
                      parent=paths.ANALYSIS_PARENT,):
    # img_dir can be a directory, .hdf5 (dataset) file or a .pt (samples) file.
    # Set output paths:
    match (img_path.is_file(), img_path.suffix):

        # Distribution file:
        case (True, ".npy"):
            print(f"Loading distribution file for {img_path.name}.")
            return np.load(img_path, allow_pickle=True).item()

        # Samples file:
        case (True, ".pt"):
            out_dir = img_path.parent
            out_file = out_dir / f"{img_path.stem}_distr.npy"

        # Dataset file:
        case (True, ".h5") | (True, ".hdf5"):
            out_dir = parent / img_path.stem
            out_dir.mkdir(exist_ok=True)
            fname = img_path.stem
            if len(h5_kwargs):
                fname += "_" + \
                    "_".join([f"{k}={v}" for k, v in h5_kwargs.items()])
            out_file = out_dir / f"{fname}_distr.npy"

        # Image directory:
        case (False, _):
            out_dir = parent / img_path.name
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"{img_path.name}_distr.npy"

        # Unknown file type:
        case _:
            raise ValueError(f"Unknown file type: {img_path.suffix}")

    # Look for existing file, return content if found
    if not force and out_file.exists():
        print(f"Found existing distribution file for {img_path.name}.")
        return np.load(out_file, allow_pickle=True).item()

    # Get metrics dict and pixel distribution
    metrics_dict = metrics_dict_from_data(
        img_path, n_bins=metrics_n_bins, **h5_kwargs
    )

    # Calculate distributions
    distributions_dict = distributions_from_metrics(
        metrics_dict, bins=hist_bins)

    # Optionally save
    if save:
        np.save(out_file, distributions_dict)

    return distributions_dict


def get_distributions_lofar(**kwargs):
    LOFAR_PATH = paths.LOFAR_SUBSETS['unclipped_SNR>=5_50asLimit']
    return get_distributions(LOFAR_PATH, **kwargs)


def get_metrics_plots(img_dir, save=True,
                      labels=None, cmap='cividis',
                      plot_train=True, force_train=False,
                      train_path=None, train_label='Train Data',
                      parent=paths.ANALYSIS_PARENT,
                      h5_kwargs={},
                      **distribution_kwargs):

    match img_dir:
        case Path():
            # Set output path
            if img_dir.is_file():
                out_dir = img_dir.parent
                fig_name = img_dir.stem
            else:
                out_dir = parent / img_dir.name
                fig_name = img_dir.name
            img_dir = [img_dir]
            save_fig = save

            if labels is None:
                labels = [fig_name]

        case list():
            warnings.warn(
                "Multiple image directories passed to metrics plot, "
                "therefore save_fig is set to False."
            )
            save_fig = False
            if labels is None:
                labels = [d.name if d.is_dir() else d.stem for d in img_dir]

    # Get metrics dict
    gen_distr_dict_list = [
        get_distributions(
            d, h5_kwargs=h5_kwargs, save=save, parent=parent,
            **distribution_kwargs
        )
        for d in img_dir
    ]
    if plot_train:
        if train_path is not None:
            lofar_distr_dict = get_distributions(train_path, force=force_train)
        else:
            lofar_distr_dict = get_distributions_lofar(force=force_train)

    out = []

    for fnc in [pixel_metrics_plot, shape_metrics_plot]:
        collection = (
            [lofar_distr_dict, *gen_distr_dict_list]
            if plot_train else gen_distr_dict_list
        )
        fig, axs = plot_collection(
            collection,
            fnc,
            colors=['darkviolet'] if plot_train else None,
            labels=[train_label, *labels] if plot_train else labels,
            cmap=cmap,
        )
        if save_fig:
            fig.savefig(
                out_dir / f"{fig_name}_{fnc.__name__}.pdf"
            )
        out.append((fig, axs))

    return out


def W1_distances(dict1, dict2):
    # Check that keys are the same
    assert dict1.keys() == dict2.keys(), \
        f"Keys of distributions do not match: {dict1.keys()} vs. {dict2.keys()}"

    # Initialize output dictionary
    out = {}

    # Loop through keys, i.e. distributions
    for key in dict1.keys():
        # Skip COM
        if key == "COM":
            continue

        # Extrtact Counts
        C1, _ = dict1[key]
        C2, _ = dict2[key]

        # Calculate W1 & error
        W1 = np.abs(cdf(C1) - cdf(C2)).sum()
        def W1_err_term(c): return np.sum(cdf(c)) / np.sum(c)  # helper func.
        W1_err = np.sqrt(W1_err_term(C1) + W1_err_term(C2))

        # Add to output dictionary
        out[key] = W1.item(), W1_err.item()
    return out


def get_W1_distances_lofar(img_dir,
                           force_W1=False, force_dist=False,
                           parent=paths.ANALYSIS_PARENT,
                           lofar_path=None):
    # Set output paths
    out_dir = parent / img_dir.name
    out_file = out_dir / f"{img_dir.name}_W1_lofar_score.json"

    # Look for existing file
    if out_file.exists() and not force_W1:
        print(f"Found existing W1 score file for {img_dir.name}.")
        with open(out_file, 'r') as f:
            return json.load(f)

    # Get distributions
    stats_gen = get_distributions(img_dir, parent=parent, force=force_dist)
    stats_lofar = get_distributions(lofar_path or LOFAR_PATH, force=force_dist)

    # Calculate W1 distances
    W1 = W1_distances(stats_gen, stats_lofar)

    # Save
    with open(out_file, 'w') as f:
        json.dump(W1, f)

    return W1


def per_bin_delta(dict1, dict2):
    # Check that keys are the same
    assert dict1.keys() == dict2.keys(), \
        f"Keys of distribution dictionaries do not match: {dict1.keys()} vs. "\
        f"{dict2.keys()}"

    # Initialize output dictionary
    delta_dict = {}

    # Loop through keys, i.e. distributions
    for key in dict1.keys():
        # Extrtact Counts & Edges
        C1, edges = dict1[key]
        C2, _ = dict2[key]

        # Normalize & Calculate error
        c1, c2 = norm(C1), norm(C2)
        e1, e2 = norm_err(C1), norm_err(C2)

        # Calculate delta & error
        delta = 2 * (c1 - c2) / (c1 + c2)
        delta_err = 4 * np.sqrt(
            (e1**2 * c2**2 + e2**2 * c1**2) / (c1 + c2)**4
        )
        # Replace NaN (from division by 0 where c1 = c2 = 0) with 0:
        delta[delta != delta] = 0
        delta_err[delta_err != delta_err] = 0

        # Add to output dictionary
        delta_dict[key] = (delta, delta_err, edges)

    return delta_dict


def RMAE(delta_dict):
    rmae_dict = {}
    for key in delta_dict:
        # Extract delta & error, calculate norm
        per_bin_delta = delta_dict[key][0]
        per_bin_delta_err = delta_dict[key][1]
        norm = len(per_bin_delta)

        # Calculate RMAE & error
        rmae = np.sum(np.abs(per_bin_delta)) / norm
        rmae_err = np.sqrt(np.sum(per_bin_delta_err**2) / norm**2)

        # Add to output dictionary
        rmae_dict[key] = (rmae, rmae_err)

    # Calculate score (& error) as sum of RMAE
    norm = len(rmae_dict)
    score = np.sum([v[0] for v in rmae_dict.values()]) / norm
    score_err = np.sqrt(np.sum([v[1]**2 for v in rmae_dict.values()])) / norm

    # Add to output dictionary
    rmae_dict["Sum"] = (score, score_err)

    return rmae_dict


def get_FID_stats(img_dir,
                  force=False, parent=paths.ANALYSIS_PARENT,):
    # Set output paths
    out_dir = parent / img_dir.name
    out_file = out_dir / f"{img_dir.name}_fid_stats.npz"

    # Look for existing file
    if out_file.exists():
        if not force:
            print(f"Found existing FID stats file for {img_dir.name}.")
            return out_file
        else:
            # Remove existing file
            out_file.unlink()

    # Assert that there are enough images
    n_imgs = len(list(img_dir.glob("*.png")))
    assert n_imgs > 2048, f"Found only {n_imgs} images in {img_dir}."

    # Calculate FID stats
    save_fid_stats([str(img_dir), str(out_file)],
                   device=torch.device('cuda', visible_gpus_by_space()[0]),
                   batch_size=64,
                   dims=2048)
    return out_file


def calculate_FID_lofar(img_dir,
                        force=False, parent=paths.ANALYSIS_PARENT,
                        lofar_path=None,):
    # Get FID stats for image data if possible (min. 2048 images),
    # else return None
    try:
        fid_stats = get_FID_stats(img_dir, force=force, parent=parent)
    except AssertionError as e:
        print(e)
        return None

    # Get FID stats for lofar data
    lofar_stats = get_FID_stats(lofar_path or LOFAR_PATH)

    # Calculate FID
    fid = calculate_fid_given_paths(
        [str(fid_stats), str(lofar_stats)],
        device=torch.device('cuda', visible_gpus_by_space()[0]),
        batch_size=64,
        dims=2048
    )
    return fid


if __name__ == "__main__":

    model_dir = Path(
        "/home/bbd0953/diffusion/results/EDM"
    )

    # Comparison of sampling methods with different T
    batch_size = 4000  # Equally distributed over GPUs
    n_batches = 20
    n_devices = 4

    img_parent = paths.GEN_DATA_PARENT / "timesteps_comparison"
    analysis_parent = paths.ANALYSIS_PARENT / "timesteps_comparison"

    timesteps = [50, 25, 10, 5, 2]

    t0 = datetime.now()
    def dt(): return datetime.now() - t0

    print(f"Starting at {t0.strftime('%H:%M:%S')}")
    for T in timesteps:
        print(f"Running {dt()} - Starting T={T}...")
        suffix = f'T={T}'
        img_dir = sample_set_from_model(
            model_dir,
            batch_size=batch_size,
            n_batches=n_batches,
            n_devices=n_devices,
            T=T,
            use_ema=True,
            img_size=80,
            sampling_method="edm_stochastic_sampling",
            out_parent=img_parent,
            dataset_suffix=suffix,
        )
        image_data_evaluation(img_dir, parent=analysis_parent)
    print(f"Runtime {dt()} - Done.")
