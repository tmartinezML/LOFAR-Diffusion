import os
from pathlib import Path

from matplotlib import colormaps
import matplotlib.lines as mlines
from numpy.lib.npyio import NpzFile
import scipy.stats as stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from model.sample import sample_batch
from model.diffusion import Diffusion
from model.init_utils import (
    load_model_from_folder,
    load_diffusion_from_folder,
    load_snapshot,
)
from datasets.datasets import TrainDataset
from utils.device_utils import distribute_model, set_visible_devices
from utils.paths import MODEL_PARENT, ANALYSIS_PARENT, LOFAR_SUBSETS

# Limit GPUs to 1
dev_ids = set_visible_devices(1)
print(f"Using GPU {dev_ids[0]}")

# SUBJECT TO CHANGE FOR DIFFERENT MODELS
model_name = f"Unclipped_H5"
lofar_subset = LOFAR_SUBSETS["unclipped_H5"]
snp_iters = [25_000 * n for n in range(1, 5)]

# Set up paths
model_dir = MODEL_PARENT / model_name
out_dir = ANALYSIS_PARENT / model_name
out_dir.mkdir(exist_ok=True)

# Load diffusion and data
diffusion = Diffusion(timesteps=25)
dataset = TrainDataset(lofar_subset)
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

# Prepare noise level loss evaluation
noise_levels = np.logspace(-3, 2, num=100)
out_dict = {"noise_levels": noise_levels}

# Loop through snapshots:
for it in snp_iters:

    # Load model and diffusion
    model = load_snapshot(model_dir, it)
    model, dev_id = distribute_model(model, n_devices=1)

    level_losses = []
    model.eval()
    with torch.no_grad():
        for noise_level in tqdm(noise_levels, desc=f"it={it}", total=len(noise_levels)):
            batch_losses = []
            for batch in dataloader:
                batch = batch.to(f"cuda:{dev_id[0]}")
                sigmas = torch.full(
                    [batch.shape[0], 1, 1, 1], noise_level, device=batch.device
                )

                # Shape: [batch_size, 1, 80, 80]
                loss = diffusion.edm_loss(model, batch, sigmas=sigmas, mean=False)
                # Shape: --> [batch_size]
                L2 = loss.detach().mean(axis=(-1, -2)).squeeze()

                batch_losses.append(L2.cpu().numpy())

            # Shape: [n_batches, batch_size] --> [n_batches * batch_size]
            batch_losses = np.concatenate(batch_losses)
            level_losses.append(batch_losses)

    # Shape: [n_levels, n_batches * batch_size]
    level_losses = np.stack(level_losses)

    out_dict[f"losses_it={it}"] = level_losses


np.savez(out_dir / "noise_level_losses.npz", **out_dict)


def noise_and_error_limits(file_dict, confidence=95):
    noise_levels = file_dict["noise_levels"]
    mean_and_limits = {}
    for key in file_dict.files:
        if key == "noise_levels":
            continue
        mean = file_dict[key].mean(axis=1)

        error_limits = np.percentile(
            file_dict[key], [d := (100 - confidence) / 2, 100 - d], axis=1
        )
        mean_and_limits[key] = mean, error_limits
    return noise_levels, mean_and_limits


def sampling_noise_levels(T, sigma_min=2e-3, sigma_max=80, rho=7):
    # Time steps
    step_inds = np.arange(T)
    rho_inv = 1 / rho
    sigma_steps = (
        sigma_max**rho_inv
        + step_inds / (T - 1) * (sigma_min**rho_inv - sigma_max**rho_inv)
    ) ** rho
    # sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])  # t_N=0
    return step_inds, sigma_steps


def add_loss_plot_data(noise_levels, mean, limits, ax, label, **plot_kwargs):
    p = ax.plot(noise_levels, mean, label=label, alpha=0.9, **plot_kwargs)
    ax.fill_between(
        noise_levels, limits[0], limits[1], alpha=0.25, color=p[0].get_color()
    )


def add_loss_plot_dict(file_dict, ax, label, best_only=False, **plot_kwargs):
    noise_levels, mean_and_limits = noise_and_error_limits(file_dict)
    # Optionally filter for model with most iterations
    if best_only:
        best_key = max(mean_and_limits.keys(), key=lambda key: int(key.split("=")[1]))
        mean_and_limits = {best_key: mean_and_limits[best_key]}

    # Plot mean and error L2 loss over noise level for each model
    cmap = "viridis"
    colors = colormaps[cmap](np.linspace(0, 1, len(mean_and_limits)))

    set_color = ~("color" in plot_kwargs)
    for color, (key, (mean, limits)) in zip(colors, mean_and_limits.items()):
        label_key = label
        if not best_only:
            label_key = label + f"_{key.split('_')[1]}"
        if set_color:
            plot_kwargs["color"] = color
        add_loss_plot_data(
            noise_levels, mean, limits, ax, label=label_key, **plot_kwargs
        )


def load_init_mean_and_limits():
    # Load initial distribution of pretrained model:
    init_dict = np.load(
        "/home/bbd0953/diffusion/analysis_results/EDM_valFix/noise_level_losses.npz"
    )
    init_noise_levels, init_mean_and_limits = noise_and_error_limits(
        init_dict, confidence=90
    )
    return init_noise_levels, *init_mean_and_limits["losses_it=400000"]


def plot_training_distribution(ax, P_mean, P_std, color="green", label=None):
    # Plot log-normal distribution used during training
    x = np.logspace(-3, 2, 10_000)
    pdf = stats.norm.pdf(np.log(x), scale=P_std, loc=P_mean)
    ax.plot(x, pdf, color=color, label=label, alpha=0.9, ls=":")


def plot_loss_over_noise_levels(
    file_dict,
    labels=[],
    best_only=False,
    plot_init=True,
    P_mean=-1.2,
    P_std=1.2,
    rho=7,
    T=25,
    sigma_min=2e-3,
    sigma_max=80,
    savefig=None,
):

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)

    # Plot initial distribution of pretrained model
    if plot_init:
        init_noise_levels, init_mean, init_limits = load_init_mean_and_limits()
        add_loss_plot_data(
            init_noise_levels,
            init_mean,
            init_limits,
            ax,
            label="Original",
            color="black",
        )
        plot_training_distribution(ax, -1.2, 1.2, color="black")

    # Plot data for fine-tuned model(s)
    match file_dict:
        case NpzFile():
            add_loss_plot_dict(file_dict, ax, labels, best_only=best_only)
            plot_training_distribution(ax, P_mean, P_std, label="Training Distr.")

        case list():
            n = len(file_dict)
            assert n == len(labels), "Number of dicts does not match number of labels."

            def match_length(var, name):
                if not isinstance(var, list):
                    var = [var] * n
                else:
                    assert (
                        len(var) == n
                    ), f"Number of dicts does not match number of {name}."
                return var

            P_mean = match_length(P_mean, "P_mean")
            P_std = match_length(P_std, "P_std")

            cmap = "tab10"
            colors = colormaps[cmap](np.linspace(0, 1, n))

            for d, l, P_m, P_s, c in zip(file_dict, labels, P_mean, P_std, colors):
                add_loss_plot_dict(d, ax, label=l, best_only=best_only, color=c)
                plot_training_distribution(ax, P_m, P_s, color=c)

    # Plot vertical lines at sigmas used during sampling
    _, sigmas = sampling_noise_levels(T, sigma_min, sigma_max, rho)
    for sigma in sigmas:
        ax.axvline(sigma, linestyle="--", color="red", alpha=0.3, linewidth=0.5)

    # Add one single patch to legend for sampling sigmas,
    # thereby create legend
    red_line = mlines.Line2D(
        [],
        [],
        color="red",
        linestyle="--",
        alpha=0.5,
        linewidth=0.7,
        label="Sampling Noise Levels",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[red_line, *handles],
        labels=["Sampling Noise Levels", *labels],
        loc="upper center",
    )

    # Set plot properties
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("L2 Loss")
    ax.grid(alpha=0.1)
    ax.set_xscale("log")

    if savefig is not None:
        fig.savefig(savefig)

    return fig, ax
