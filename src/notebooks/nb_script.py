# RUN THIS REGARLDESS OF WHICH PART

from pathlib import Path
from analysis.model_evaluations import (
    get_W1_lofar_score, get_distributions, get_distributions_lofar,
    per_bin_error, example_plot
)
from utils.init_utils import load_config

# File paths used in analysis.model_evaluations
ANALYSIS_RESULTS_PARENT = Path("../analysis_results/snapshot_run")
GEN_DATA_PARENT = Path("../image_data/generated/snapshot_run")
# Generate parent directories
ANALYSIS_RESULTS_PARENT.mkdir(exist_ok=True)
GEN_DATA_PARENT.mkdir(exist_ok=True)

# File paths used for generating data and plotting distributions
model_dir = Path("../results/InitModel_EDM_SnapshotRun")
config_file = model_dir / 'config_InitModel_EDM_SnapshotRun.json'
snapshot_dir = model_dir / 'snapshots'

# Define functions to sample images and calculate pixel distributions
import torch
from model.sample import sample_n_batches
from utils.device_utils import distribute_model

# Global variables for sampling
n_devices = 3
bsize = 512 * n_devices
n_batches = 5
T = 50

def snapshot_eval(model, file, diffusion):
    # Load param dict from file
    checkpoint = torch.load(file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = distribute_model(model, n_devices)
    model.eval()

    # Specify model naming
    model_name = file.stem
    img_dir = GEN_DATA_PARENT / (model_name + 'T=50')
    img_dir.mkdir(exist_ok=True)

    # Sample data set
    sample_n_batches(model, diffusion, img_dir, bsize, n_batches)

    # Get distributions and W1 score
    get_W1_lofar_score(img_dir)

# Execute for all snapshots
from utils.init_utils import load_model
from model.edm_diffusion import EDM_Diffusion

# Load model and diffusion
model = load_model(config_file)
diffusion = EDM_Diffusion.from_config(load_config(config_file))
diffusion.timesteps = T

# Get all ema files in snapshot directory
ema_files = [
    file for file in snapshot_dir.glob('*.pt') if file.stem.startswith('ema')
]

# Evaluate all snapshots
for file in ema_files[::2]:
    print(f"\n\nEvaluating {file.stem}")
    snapshot_eval(model, file, diffusion)