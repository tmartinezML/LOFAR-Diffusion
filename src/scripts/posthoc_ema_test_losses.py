from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import model.posthoc_ema as phema
from utils.data_utils import TrainDataset
from training.loss_functions import edm_loss
from utils.paths import MODEL_PARENT, ANALYSIS_PARENT, LOFAR_SUBSETS
from utils.device_utils import distribute_model, set_visible_devices
from utils.init_utils import load_model_by_name

# Limit GPUs to 1
dev_ids = set_visible_devices(1)
print(f"Using GPU {dev_ids[0]}")

# SUBJECT TO CHANGE FOR DIFFERENT MODELS
model_name = f"PowerEMA"
lofar_subset = LOFAR_SUBSETS["0-clip"]
posthoc_sigmas = np.linspace(0.02, 0.25, num=23)

# Set up paths
model_dir = MODEL_PARENT / model_name
out_dir = ANALYSIS_PARENT / model_name
out_dir.mkdir(exist_ok=True)

# Load diffusion and data
dataset = TrainDataset(lofar_subset, n_subset=1_000)
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

# Loop through snapshots:
out_dict = {}
for sigma in tqdm(posthoc_sigmas, desc="Looping through sigmas"):

    # Load model and diffusion
    model = phema.posthoc_model(
        phema.gamma_from_sigma(sigma),
        model_dir / "power_ema",
    )
    model, dev_id = distribute_model(model, n_devices=1)

    model.eval()
    with torch.no_grad():
        batch_losses = []
        for batch in dataloader:
            batch = batch.to(f"cuda:{dev_id[0]}")

            # Shape: [batch_size, 1, 80, 80]
            loss = edm_loss(model, batch, mean=False)
            # Shape: --> [batch_size]
            L2 = loss.detach().mean(axis=(-1, -2)).squeeze()

            batch_losses.append(L2.cpu().numpy())

        # Shape: [n_batches, batch_size] --> [n_batches * batch_size]
        batch_losses = np.concatenate(batch_losses)

    out_dict[f"losses_sigma={sigma}"] = batch_losses

# Load original model
model = load_model_by_name(model_name)
model, dev_id = distribute_model(model, n_devices=1)

model.eval()
with torch.no_grad():
    batch_losses = []
    for batch in dataloader:
        batch = batch.to(f"cuda:{dev_id[0]}")

        # Shape: [batch_size, 1, 80, 80]
        loss = edm_loss(model, batch, mean=False)
        # Shape: --> [batch_size]
        L2 = loss.detach().mean(axis=(-1, -2)).squeeze()

        batch_losses.append(L2.cpu().numpy())

    # Shape: [n_batches, batch_size] --> [n_batches * batch_size]
    batch_losses = np.concatenate(batch_losses)

out_dict["losses_original"] = batch_losses


np.savez(out_dir / "posthoc_ema_losses.npz", **out_dict)
