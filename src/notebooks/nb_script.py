import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
import numpy as np
from utils.init_utils import (
    load_model_from_folder, load_diffusion_from_folder, load_snapshot
)
from torch.utils.data import DataLoader
from model.sample import sample_batch
from utils.init_utils import load_model_from_folder
from model.diffusion import EDM_Diffusion
from utils.data_utils import LofarZoomUnclipped80

from utils.device_utils import distribute_model
from tqdm import tqdm
import numpy as np

# Set directories
model_dir = Path(
    "/home/bbd0953/diffusion/results/EDM_valFix_Pmean=-5"
)
out_dir = Path('/home/bbd0953/diffusion/analysis_results/EDM_valFix_Pmean=-5')
out_dir.mkdir(exist_ok=True) 

# Load diffusion and data
diffusion = EDM_Diffusion(timesteps=25)
dataset = LofarZoomUnclipped80(n_subset=1000)
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

noise_levels = np.logspace(-3, 2, num=100)

out_dict = {
    'noise_levels': noise_levels,
}

snp_iters = [n*10_000 for n in range(1, 6)]
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