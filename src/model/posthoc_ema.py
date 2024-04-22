from functools import partial

import numpy as np
import torch
import wandb

from model.sample import sample_batch
from analysis.model_evaluations import get_distributions
import utils.stats_utils as stats
from utils.device_utils import set_visible_devices, distribute_model


def weight_profile(t, gamma, t_max=None):
    t_max = t_max or t.max()
    return (gamma + 1) / t_max ** (gamma + 1) * t**gamma * (t <= t_max)


def snapshot_profiles(gammas, n_snapshots, t=None):
    t = t or np.linspace(0, 1, 1000)
    profiles = []
    for gamma in gammas:
        for snp in range(0, n_snapshots):
            profiles.append(weight_profile(t, gamma, t_max=(snp + 1) / n_snapshots))
    return np.array(profiles)


def get_coefficients(gamma, n_snapshots, gammas_snp=[16.97, 6.94]):
    t = np.linspace(0, 1, 1000)
    profiles = snapshot_profiles(gammas_snp, n_snapshots)
    target = snapshot_profiles([gamma], 1)[0]
    coeff = np.linalg.lstsq(profiles.T, target)[0]
    return coeff


def posthoc_model(gamma, snp_dir, coeff, gammas_snp=[16.97, 6.94]):

    # Reshape coeffs: First dim is gamma, 2nd dim is snapshot
    coeff = coeff.reshape(len(gammas_snp), -1)
    snapshots = sorted(snp_dir.glob("*.pt"))

    # Read first snapshot and zero all params
    model = torch.load(snapshots[0]).values()[0]
    for p in model.parameters():
        p.data.zero_()

    # Make linear combination of snapshots with coeffs
    for i, snap in enumerate(snapshots):
        for j, gamma in enumerate(gammas_snp):
            model_i = torch.load(snap)[f"model_{gamma}"]
            for p, p_i in zip(model.parameters(), model_i.parameters()):
                p.data.add_(p_i, alpha=coeff[j, i])

    return model


def sigma_from_gamma(gamma):
    return (gamma + 1) ** (1 / 2) * (gamma + 2) ** (-1) * (gamma + 3) ** (-1 / 2)


def gamma_from_sigma(sigma):
    gammas = np.linspace(0, 1000, 1_000_000)
    return gammas[np.argmin(np.abs(sigma_from_gamma(gammas) - sigma))]


def get_samples(model, n_samples, n_devices, **sample_kwargs):
    batch_size = 1000 * n_devices
    n_batches = int(n_samples / batch_size)
    print(f"Sampling {n_batches} batches.")

    # Distribute model
    model, _ = distribute_model(model, n_devices=n_devices)

    # Sample from model
    batch_list = []
    for i in range(n_batches):
        batch = sample_batch(
            model,
            bsize=batch_size,
            return_steps=False,
            **sample_kwargs,
        )
        batch_list.append(batch)

    img_batch = torch.cat(batch_list)

    # Clamp to [0, 1]
    img_batch = torch.clamp((img_batch + 1) / 2, 0, 1)

    # Clear GPU
    model = (
        model.module.to("cpu")
        if isinstance(model, torch.nn.DataParallel)
        else model.to("cpu")
    )

    return img_batch.squeeze().numpy()


def samples_eval(img_batch, counts_lofar):

    # Evaluate samples
    counts, _ = np.histogram(img_batch, bins=np.linspace(0, 1, 257))
    W1, W1_err = stats.W1_distance(counts, counts_lofar)

    return W1, W1_err, counts


def posthoc_ema_eval(config, snp_dir, counts_lofar, gammas_snp=[16.97, 6.94]):
    sigma = config.sigma
    n_snapshots = len(sorted(snp_dir.glob("*.pt")))
    gamma = gamma_from_sigma(sigma)
    model = posthoc_model(
        gamma=gamma,
        snp_dir=snp_dir,
        coeff=get_coefficients(gamma, n_snapshots, gammas_snp),
        gammas_snp=gammas_snp,
    )
    img_batch = get_samples(
        model, config.n_samples, config.n_devices, **config.sample_kwargs
    )
    return samples_eval(img_batch, counts_lofar)


def posthoc_ema_wrapper(config=None, snp_dir=None, counts_lofar=None):
    with wandb.init(config=config):
        W1, W1_err, counts = posthoc_ema_eval(
            wandb.config, snp_dir, counts_lofar=counts_lofar
        )
        wandb.log({"counts": counts, "W1": W1, "W1_err": W1_err})

        # Add plot
        centers = np.linspace(0, 1, 256)
        centers = (centers[1:] + centers[:-1]) / 2
        wandb.log(
            {
                "Pixel_Distribution_Logplot": wandb.plot.line_series(
                    xs=centers,
                    ys=[np.log(stats.norm(counts)), np.log(stats.norm(counts_lofar))],
                    keys=["Sampled", "LOFAR"],
                    title="Pixel Distribution",
                    xname="Pixel Intensity",
                )
            }
        )


def run_sweep(config, model_path, lofar_path):
    counts_lofar, _ = get_distributions(lofar_path)["Pixel_Intensity"]
    snp_dir = model_path / "power_ema"

    sweep_id = wandb.sweep(config, project="Diffusion")
    wandb.agent(
        sweep_id,
        function=partial(
            posthoc_ema_wrapper, snp_dir=snp_dir, counts_lofar=counts_lofar
        ),
    )
