import os
from pathlib import Path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


from model.sample import sample_batch
from model.diffusion import Diffusion
from utils.init_utils import (
    load_model_from_folder, load_diffusion_from_folder, load_snapshot
)
from utils.data_utils import TrainDataset
from utils.device_utils import distribute_model, set_visible_devices
from utils.paths import MODEL_PARENT, ANALYSIS_PARENT, LOFAR_SUBSETS

# Limit GPUs to 1
dev_ids = set_visible_devices(1)
print(f"Using GPU {dev_ids[0]}")

# SUBJECT TO CHANGE FOR DIFFERENT MODELS
model_name = f'Unclipped_H5'
lofar_subset = LOFAR_SUBSETS['unclipped_H5']
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
out_dict = {'noise_levels': noise_levels}

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
                loss = diffusion.edm_loss(
                    model, batch, sigmas=sigmas, mean=False
                )
                # Shape: --> [batch_size]
                L2 = loss.detach().mean(axis=(-1, -2)).squeeze()

                batch_losses.append(L2.cpu().numpy())

            # Shape: [n_batches, batch_size] --> [n_batches * batch_size]
            batch_losses = np.concatenate(batch_losses)
            level_losses.append(batch_losses)

    # Shape: [n_levels, n_batches * batch_size]
    level_losses = np.stack(level_losses)

    out_dict[f"losses_it={it}"] = level_losses


np.savez(
    out_dir / "noise_level_losses.npz",
    **out_dict
)
