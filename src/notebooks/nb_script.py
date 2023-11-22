import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path
from utils.init_utils import load_model_from_folder, load_diffusion_from_folder

folder = Path(
    "/home/bbd0953/diffusion/results/EDM_small"
)

# Load model and diffusion
model = load_model_from_folder(folder)
diffusion = load_diffusion_from_folder(folder)
from utils.data_utils import LofarSubset
from torch.utils.data import DataLoader
from utils.device_utils import distribute_model

dataset = LofarSubset()
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
from importlib import reload
import utils.device_utils
reload(utils.device_utils)
from utils.device_utils import distribute_model

model, device_ids = distribute_model(model, n_devices=2)
from tqdm import tqdm

device = torch.device(f"cuda:{device_ids[0]}")
model.eval()
losses = []
with torch.no_grad():
    for batch in tqdm(dataloader, total=len(dataset)//256, desc="Calculating Loss..."):
        batch = batch.to(device)
        loss = diffusion.edm_loss(model, batch)
        losses.append(loss.item())
    

# Save losses
losses = torch.Tensor(losses)
torch.save(losses, "/home/bbd0953/diffusion/results/playground/losses.pt")