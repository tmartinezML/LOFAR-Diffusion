from pathlib import Path
import shutil
from datetime import datetime
import json
import random

from PIL import Image
import torch
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm

from model.sample import sample_set_from_folder
from utils.plot_utils import plot_samples, plot_distributions
from utils.device_utils import visible_gpus_by_space
from utils.data_utils import GeneratedDataset
from utils.stats_utils import norm, norm_err, cdf, cdf_err
from model.sample import sample_set_from_folder
from analysis.fid_score import calculate_fid_given_paths, save_fid_stats

# Folder with lofar images
LOFAR_PATH = Path("/storage/tmartinez/image_data/lofar_subset")
# Parent folder for generated image data
GEN_DATA_PARENT = Path("/storage/tmartinez/image_data/generated")
# Parent folder for analysis results
ANALYSIS_RESULTS_PARENT = Path("/storage/tmartinez/analysis_results")
# Parent folder for models
MODELS_PARENT = Path("/storage/tmartinez/results")


def get_distributions(img_dir,
                      intensity_bins=256,
                      pixsum_bins=None,
                      act_bins=None, force=False, save=True,
                      act_threshold=0.,
                      parent=ANALYSIS_RESULTS_PARENT):
    
    # Set output paths
    out_dir = parent / img_dir.name
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{img_dir.name}_distr.npy"

    # Look for existing file
    if save and not force and out_file.exists():
        print(f"Found existing distribution file for {img_dir.name}.")
        return np.load(out_file, allow_pickle=True).item()

    # Load image data
    dataset = GeneratedDataset(img_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=1, drop_last=False,
    )

    # Set up histogram bins
    std_sum_bins = np.linspace(0, 6400, num=6400 // 50, endpoint=True)
    pixsum_bins = pixsum_bins or std_sum_bins
    act_bins = act_bins or std_sum_bins

    # Initialize empty histograms, will be filled in loop
    def get_nbins(bins): return bins if isinstance(
        bins, int) else len(bins) - 1
    intensity_hist = np.zeros(get_nbins(intensity_bins), dtype=int)
    pixsum_hist = np.zeros(get_nbins(pixsum_bins))
    act_hist = np.zeros(get_nbins(act_bins))

    # Loop through images and fill histograms
    for img in tqdm(dataloader, desc="Calculating distributions..."):
        assert img.shape[1] == 1, f"Only single-channel images supported, "\
                                  f"found {img.shape[1]} channels."
        img = img.squeeze().numpy()

        # Histogram 1: Pixel intensities
        i_count, i_edges = np.histogram(img, bins=intensity_bins, range=(0, 1))
        intensity_hist += i_count.astype(int)

        # Histogram 2: Pixel sums
        p_count, p_edges = np.histogram(
            np.sum(img, axis=(-1, -2)), bins=pixsum_bins, range=(0, 6400)
        )
        pixsum_hist += p_count

        # Histogram 3: Activated pixels
        n_act = np.sum((img > act_threshold).squeeze(), axis=(-2, -1))
        act_count, act_edges = np.histogram(
            n_act, bins=act_bins, range=(0, 6400))
        act_hist += act_count

    # Put histograms in output dictionary and optionally save
    out = {
        "Pixel Intensity": (intensity_hist, i_edges),
        "Pixel Sum": (pixsum_hist, p_edges),
        "Activated Pixels": (act_hist, act_edges)
    }
    if save:
        np.save(out_file, out)
    return out


def get_distributions_lofar(**kwargs):
    return get_distributions(LOFAR_PATH, **kwargs)


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


def get_distribution_plot_with_lofar(img_dir, save=True,
                                     parent=ANALYSIS_RESULTS_PARENT,
                                     **distribution_kwargs):
    # Set output path
    out_dir = parent / img_dir.name

    # Get distributions
    gen_stats = get_distributions(
        img_dir, save=save, parent=parent, **distribution_kwargs
    )
    lofar_stats = get_distributions_lofar(save=save, **distribution_kwargs)

    # Calculate delta
    delta = per_bin_delta(gen_stats, lofar_stats)

    # Plot
    fig = plot_distributions(gen_stats, lofar_stats, delta,
                             title=img_dir.name)

    # Save
    if save:
        fig.savefig(out_dir / f"{out_dir.name}_lofar_stats_distr.pdf")

    return fig


def get_FID_stats(img_dir, force=False, parent=ANALYSIS_RESULTS_PARENT):
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


def calculate_FID_lofar(img_dir, force=False, parent=ANALYSIS_RESULTS_PARENT):
    # Get FID stats for image data if possible (min. 2048 images),
    # else return None
    try:
        fid_stats = get_FID_stats(img_dir, force=force, parent=parent)
    except AssertionError as e:
        print(e)
        return None
    
    # Get FID stats for lofar data
    lofar_stats = get_FID_stats(LOFAR_PATH)

    # Calculate FID
    fid = calculate_fid_given_paths(
        [str(fid_stats), str(lofar_stats)],
        device=torch.device('cuda', visible_gpus_by_space()[0]),
        batch_size=64,
        dims=2048
    )
    return fid


def calculate_W1_distances(dict1, dict2):
    # Check that keys are the same
    assert dict1.keys() == dict2.keys(), \
        f"Keys of distributions do not match: {dict1.keys()} vs. {dict2.keys()}"

    # Initialize output dictionary
    out = {}

    # Loop through keys, i.e. distributions
    for key in dict1.keys():
        # Extrtact Counts
        C1, _ = dict1[key]
        C2, _ = dict2[key]

        # Calculate W1 & error
        W1 = np.abs(cdf(C1) - cdf(C2)).sum()
        W1_err_term = lambda c: np.sum(cdf(c)) / np.sum(c)  # helper func.
        W1_err = np.sqrt(W1_err_term(C1) + W1_err_term(C2))

        # Add to output dictionary
        out[key] = W1.item(), W1_err.item()
    return out


def get_W1_lofar_score(img_dir, force=False, parent=ANALYSIS_RESULTS_PARENT):
    # Set output paths
    out_dir = parent / img_dir.name
    out_file = out_dir / f"{img_dir.name}_W1_lofar_score.json"

    # Look for existing file
    if out_file.exists() and not force:
        print(f"Found existing W1 score file for {img_dir.name}.")
        with open(out_file, 'r') as f:
            return json.load(f)

    # Get distributions
    stats_gen = get_distributions(img_dir, parent=parent)
    stats_lofar = get_distributions(LOFAR_PATH)

    # Calculate W1 distances
    W1_distances = calculate_W1_distances(stats_gen, stats_lofar)

    # Save
    with open(out_file, 'w') as f:
        json.dump(W1_distances, f)

    return W1_distances


def image_data_evaluation(img_dir, force=False, parent=ANALYSIS_RESULTS_PARENT):
    # Make img_dir a Path object and assert that image directory exists
    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)
    assert img_dir.exists(), f"Image directory {img_dir} does not exist."

    # Set output paths
    out_dir = parent / img_dir.name
    out_dir.mkdir(exist_ok=True, parents=True)

    # Copy sampling info file(s) to output directory
    for f in img_dir.glob("*.json"):
        shutil.copy(f, out_dir / f.name)

    # Get distributions
    print(f"Calculating distributions for {img_dir.name}...")
    stats = get_distributions(img_dir, force=force, parent=parent)

    # Plot with lofar
    print(f"Plotting distributions for {img_dir.name}...")
    fig = get_distribution_plot_with_lofar(img_dir, parent=parent)

    # Get FID (if enough images, else fid = None)
    print(f"Calculating FID score for {img_dir.name}...")
    fid = calculate_FID_lofar(img_dir, force=force, parent=parent)

    # Get W1
    print(f"Calculating W1 score for {img_dir.name}...")
    W1 = get_W1_lofar_score(img_dir, force=force, parent=parent)

    # Save FID and W1 scores
    scores = {
        "FID": fid,
        "W1": W1
    }
    with open(out_dir / f"{img_dir.name}_scores.json", 'w') as f:
        json.dump(scores, f)

    # Generate plot with 25 images from folder
    example_plot_file = out_dir / f"{img_dir.name}_example_plot.pdf"
    if not example_plot_file.exists() or force:
        fig = example_plot(img_dir, example_plot_file)

    return fig, fid, W1


def example_plot(img_dir, example_plot_file=None):
    # Sample 25 random images from folder
    example_imgs = random.sample(sorted(img_dir.glob("*.png")), 25)
    example_imgs = np.array(
        [ToTensor()(Image.open(f))[:1, :, :] for f in example_imgs]
    )

    # Plot
    fig, _ = plot_samples(torch.Tensor(example_imgs),
                          title=img_dir.name, vmin=0)

    # Optionally save
    if example_plot_file is not None:
        fig.savefig(example_plot_file)

    return fig


def generate_samples_and_evaluate(model_dir, **sampling_kwargs):
    assert model_dir.exists(), f"Model directory {model_dir} does not exist."
    sample_set_from_folder(model_dir, **sampling_kwargs)
    return image_data_evaluation(GEN_DATA_PARENT / model_dir.name)


if __name__ == "__main__":

    model_dir = Path(
        "/home/bbd0953/diffusion/results/EDM"
    )

    # Comparison of sampling methods with different T
    batch_size = 4000  # Equally distributed over GPUs
    n_batches = 20
    n_devices = 4

    img_parent = GEN_DATA_PARENT / "timesteps_comparison"
    analysis_parent = ANALYSIS_RESULTS_PARENT / "timesteps_comparison"

    timesteps = [50, 25, 10, 5, 2]

    t0 = datetime.now()
    def dt(): return datetime.now() - t0

    print(f"Starting at {t0.strftime('%H:%M:%S')}")
    for T in timesteps:
        print(f"Running {dt()} - Starting T={T}...")
        suffix = f'T={T}'
        img_dir = sample_set_from_folder(
            model_dir,
            batch_size=batch_size,
            n_batches=n_batches,
            n_devices=n_devices,
            T=T,
            use_ema=True,
            img_size=80,
            sampling_method="edm_stochastic_sampling",
            out_parent=img_parent,
            out_folder_suffix=suffix,
        )
        image_data_evaluation(img_dir, parent=analysis_parent)
    print(f"Runtime {dt()} - Done.")
