from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from model.unet import Unet, ImprovedUnet
from model.diffusion import Diffusion
from model.create_dataset import sample_set_from_folder
from utils.plot_utils import plot_losses, plot_samples, plot_distributions
from utils.train_utils import get_free_gpu
from utils.data_utils import ImagePathDataset
from analysis.fid_score import calculate_fid_given_paths, save_fid_stats

LOFAR_PATH = Path("/storage/tmartinez/image_data/lofar_subset")
GEN_DATA_PARENT = Path("/storage/tmartinez/image_data/generated")
ANALYSIS_RESULTS_PARENT = Path("/storage/tmartinez/analysis_results")
MODELS_PARENT = Path("/storage/tmartinez/results")

def load_Unet_model(hp, model_file=None):
    UnetModel = ImprovedUnet if hp.use_improved_unet else Unet
    model = UnetModel.from_config(hp)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    return model
    
def model_evaluation(model_folder, use_ema=True):
    model_folder = Path(model_folder)
    # Set file names
    def get_file_name(name):
        l = sorted(model_folder.glob(name))
        assert len(l) == 1, f"Found more than one {name} file:\n{l}"
        return l[0]
    hp_file = get_file_name("hyperparameters_*.pt")
    model_file = get_file_name(
        f"parameters_{'ema' if use_ema else 'model'}_*.pt"
    )
    loss_file = get_file_name("losses_*.csv")

    # Load hyperparams and model
    hp = hyperParams(**torch.load(hp_file))
    device = torch.device("cuda", get_free_gpu()[0])
    print(f"Using device {device}")
    model = load_Unet_model(hp, model_file).to(device)
    diffusion = Diffusion.from_config(hp)

    # Prepare timer
    t0 = datetime.now()
    dt = lambda: datetime.now() - t0

    # Plot losses
    fig, _ = plot_losses(loss_file, title=hp.model_name)
    fig.savefig(model_folder / f"losses_{hp.model_name}.pdf")

    # Plot ddpm samples
    t0 = datetime.now()
    imgs = diffusion.p_sampling(model, hp.image_size, batch_size=25)[-1]
    t_sample = dt()
    title = f"{hp.model_name} (DDPM) --- T={hp.timesteps} " \
            f"[{t_sample.total_seconds():.2f}s for sampling]"
    fig, ax = plot_samples(imgs, title=title)
    fig.savefig(model_folder / f"samples_ddpm_{hp.model_name}.pdf")

    # Plot ddim samples
    t0 = datetime.now()
    imgs = diffusion.ddim_sampling(model, hp.image_size, batch_size=25)[-1]
    t_sample = dt()
    title = f"{hp.model_name} (DDIM) --- T={hp.timesteps} " \
            f"[{t_sample.total_seconds():.2f}s for sampling]"
    fig, _ = plot_samples(imgs, title=title)
    fig.savefig(model_folder / f"samples_ddim_{hp.model_name}.pdf")


def get_distributions(img_dir, 
                      intensity_bins=256,
                      pixsum_bins=None,
                      act_bins=None, force=False):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{img_dir.name}_distr.pt"
    # Look for existing file
    if out_file.exists() and not force:
        print(f"Found existing distribution file for {img_dir.name}.")
        n_images = len(list(img_dir.glob("*.png")))
        return torch.load(img_dir / f"{img_dir.name}_distr.pt"), n_images
    
    # Load data
    dataset = ImagePathDataset(img_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=1, drop_last=False
    )
    
    std_sum_bins = torch.tensor(
        np.linspace(0, 6400, num=6400//50, endpoint=True)
    ).to(torch.float32)
    if pixsum_bins is None:
        pixsum_bins = std_sum_bins
    if act_bins is None:
        act_bins = std_sum_bins
        
    get_nbins = lambda bins: bins if isinstance(bins, int) else len(bins) - 1
    intensity_hist = torch.zeros(get_nbins(intensity_bins)).to('cuda:0')
    pixsum_hist = torch.zeros(get_nbins(pixsum_bins)).to('cuda:0')
    act_hist = torch.zeros(get_nbins(act_bins)).to('cuda:0')

    for img in tqdm(dataloader, desc="Calculating distributions..."):
        img.to('cuda:0')
        # Pixel intensities
        i_count, i_edges = torch.histogram(img, bins=intensity_bins)
        intensity_hist = torch.add(intensity_hist, i_count.to('cuda:0'))
        # Pixel sums
        p_count, p_edges = torch.histogram(torch.sum(img, (-1, -2)),
                                            bins=pixsum_bins)
        pixsum_hist = torch.add(pixsum_hist, p_count.to('cuda:0'))
        # Activated pixels
        n_act = torch.sum((img>=1./256).squeeze(), [-2, -1]).to(torch.float32)  # Shape: [b]
        act_count, act_edges = torch.histogram(n_act, 
                                                   bins=act_bins)
        act_hist = torch.add(act_hist, act_count.to('cuda:0'))
        
    out = {
        "Pixel Intensity": (intensity_hist.to('cpu'), i_edges.to('cpu')),
        "Pixel Sum": (pixsum_hist.to('cpu'), p_edges.to('cpu')),
        "Activated Pixels": (act_hist.to('cpu'), act_edges.to('cpu'))
    }

    torch.save(out, out_file)

    return out, len(dataset)

def per_bin_error(distr1, distr2):
    error_dict = {}

    for key in distr1.keys():
        # Counts & Edges
        C1, edges = distr1[key]
        C2, _ = distr2[key]

        C1_bar = torch.sum(C1)
        C2_bar = torch.sum(C2)

        c1 = C1 / C1_bar
        c2 = C2 / C2_bar

        e1_sq = C1 / C1_bar**3 * (C1_bar - C1)
        e2_sq = C2 / C2_bar**3 * (C2_bar - C2)


        error = 2 * (c1 - c2) / (c1 + c2)
        error_delta_squared = 16 * (
            (e1_sq * c2**2 + e2_sq * c1**2) / (c1 + c2)**4
        )

        # Replace NaN entries (from division by 0 for entries where c1 = c2 = 0)
        # with 0:
        error[error != error] = 0
        error_delta_squared[error_delta_squared != error_delta_squared] = 0

        error_dict[key] = (error, error_delta_squared, edges)
    
    return error_dict

def RMAE(error_dict):
    rmae_dict = {}
    for key in error_dict:
        per_bin_error = error_dict[key][0]
        per_bin_error_delta_sq = error_dict[key][1]
        norm = len(per_bin_error)
        rmae = torch.sum(torch.abs(per_bin_error)) / norm
        rmae_delta_squared = torch.sum(per_bin_error_delta_sq) / norm**2
        
        rmae_dict[key] = (rmae, rmae_delta_squared)
    
    norm = len(rmae_dict)
    score = torch.sum(torch.stack([v[0] for v in rmae_dict.values()])) / norm
    score_delta_squared = torch.sum(torch.stack([v[1] for v in rmae_dict.values()])) / norm
    rmae_dict["Sum"] = (score, score_delta_squared)

    return rmae_dict

def get_distribution_plot_with_lofar(img_dir):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    gen_stats, n_gen = get_distributions(img_dir)
    lofar_stats, n_lofar = get_distributions(LOFAR_PATH)

    error = per_bin_error(gen_stats, lofar_stats)

    fig = plot_distributions(gen_stats, lofar_stats, error, n_gen, n_lofar)
    fig.savefig(out_dir / f"{out_dir.name}_lofar_stats_distr.pdf")
    return fig

def get_FID_stats(img_dir, force=False):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    out_file = out_dir / f"{img_dir.name}_fid_stats.npz"
    if out_file.exists():
        if not force:
            print(f"Found existing FID stats file for {img_dir.name}.")
        else:
            out_file.remove()

    else:
        n_imgs = len(list(img_dir.glob("*.png")))
        assert n_imgs > 2048, f"Found only {n_imgs} images in {img_dir}."
        save_fid_stats([str(img_dir), str(out_file)],
                       device=torch.device('cuda', get_free_gpu()[0]),
                       batch_size=64,
                       dims=2048)
    return out_file

def calculate_FID_lofar(img_dir):
    fid_stats = get_FID_stats(img_dir)
    lofar_stats = get_FID_stats(LOFAR_PATH)
    fid = calculate_fid_given_paths(
        [str(fid_stats), str(lofar_stats)],
        device=torch.device('cuda', get_free_gpu()[0]),
        batch_size=64,
        dims=2048
    )
    return fid

def calculate_W1_distances(distr1, distr2):
    assert distr1.keys() == distr2.keys()
    out = {}
    for key in distr1.keys():
        C1, _ = distr1[key]
        C2, _ = distr2[key]
        out[key] = wasserstein_distance(C1 / C1.sum(), C2 / C2.sum())
    return out

def get_W1_lofar_score(img_dir, force=False):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    out_file = out_dir / f"{img_dir.name}_W1_lofar_score.json"
    if out_file.exists() and not force:
            print(f"Found existing W1 score file for {img_dir.name}.")
            with open(out_file, 'r') as f:
                return json.load(f)

    stats_gen, _ = get_distributions(img_dir)
    stats_lofar, _ = get_distributions(LOFAR_PATH)

    W1_distances = calculate_W1_distances(stats_gen, stats_lofar)

    with open(out_file, 'w') as f:
        json.dump(W1_distances, f)

    return W1_distances

def image_data_evaluation(img_dir, force=False):
    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)
    assert img_dir.exists(), f"Image directory {img_dir} does not exist."

    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    out_dir.mkdir(exist_ok=True)

    # Get distributions
    stats, n_gen = get_distributions(img_dir, force=force)
    # Plot with lofar
    fig = get_distribution_plot_with_lofar(img_dir, force=force)
    # Get FID
    n_imgs = len(list(img_dir.glob("*.png")))
    if n_imgs < 2048:
        print(f"Found only {n_imgs} images in {img_dir}.")
        fid = None
    else:
        fid = calculate_FID_lofar(img_dir, force=force)
    # Get W1
    W1 = get_W1_lofar_score(img_dir, force=force)

    scores = {
        "FID": fid,
        "W1": W1
    }

    with open(out_dir / f"{img_dir.name}_scores.json", 'w') as f:
        json.dump(scores, f)

    return fig, fid, W1

def generate_samples_and_evaluate(model_dir, **sampling_kwargs):
    assert model_dir.exists(), f"Model directory {model_dir} does not exist."
    sample_set_from_folder(model_dir, **sampling_kwargs)
    return image_data_evaluation(GEN_DATA_PARENT / model_dir.name)

if __name__ == "__main__":
    model_folder = "OptiModel_Initial"
    generate_samples_and_evaluate(
        MODELS_PARENT / model_folder,
        batch_size = 384,
        n_batches = 7,
        n_devices = 3,
        T = 250,
        )