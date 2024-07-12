from pathlib import Path
from functools import partial

import numpy as np
import torch
import wandb

import utils.paths as paths
import analysis.stats_utils as stats
from model.model_utils import load_model
from analysis.model_evaluation import get_distributions
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


def posthoc_model(gamma, snp_dir, coeff=None, gammas_snp=[16.97, 6.94]):

    # Reshape coeffs: First dim is gamma, 2nd dim is snapshot
    snapshots = sorted(snp_dir.glob("*.pt"))
    if coeff is None:
        coeff = get_coefficients(gamma, len(snapshots), gammas_snp=gammas_snp)
    coeff = coeff.reshape(len(gammas_snp), -1)

    # Read first snapshot and zero all params
    model = load_model(
        next(snp_dir.parent.glob("config_*.json")),
        model_file=snapshots[0],
        key=f"model_{gammas_snp[0]}",
    )
    for p in model.parameters():
        p.data.zero_()

    # Make linear combination of snapshots with coeffs
    for i, snap in enumerate(snapshots):
        for j, gamma in enumerate(gammas_snp):

            # Load state dict of snapshot
            model_i = torch.load(snap, map_location="cpu")[f"model_{gamma}"]

            # Skip first value in snapshot state dict bc. it's the number of
            # model averaging steps
            for p, p_i in zip(model.parameters(), list(model_i.values())[1:]):
                p.data.add_(p_i, alpha=coeff[j, i])

    return model


def sigma_from_gamma(gamma):
    return (gamma + 1) ** (1 / 2) * (gamma + 2) ** (-1) * (gamma + 3) ** (-1 / 2)


def gamma_from_sigma(sigma):
    gammas = np.linspace(0, 1000, 1_000_000)
    return gammas[np.argmin(np.abs(sigma_from_gamma(gammas) - sigma))]


def get_samples(model, n_samples, n_devices, **sample_kwargs):
    # TODO: Implement batch sampling
    raise NotImplementedError("Batch sampling not implemented yet.")


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
        model,
        config.n_samples,
        config.n_devices,
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


if __name__ == "__main__":
    N_GPU = 2
    DEV_IDS = set_visible_devices(N_GPU)
    print("Setting visible devices:", DEV_IDS)

    sweep_config = {
        "name": "Posthoc EMA Sweep 2",
        "method": "grid",
        "metric": {"name": "W1", "goal": "minimize"},
        "parameters": {
            "sigma": {"values": list(np.arange(0.002, 0.1, step=0.002))},
            "n_samples": {"value": 1_000},
            "n_devices": {"value": 1},
        },
    }

    run_sweep(
        sweep_config,
        model_path=paths.MODEL_PARENT / "PowerEMA",
        lofar_path=paths.LOFAR_SUBSETS["0-clip"],
    ),
