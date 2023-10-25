from pathlib import Path
from datetime import datetime
import json
import random
from functools import partial

from PIL import Image
import torch
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
from scipy.stats import wasserstein_distance

from model.unet import Unet, ImprovedUnet
from model.diffusion import Diffusion
from model.edm_diffusion import EDM_Diffusion
from model.sample import sample_set_from_folder
from utils.plot_utils import plot_losses, plot_samples, plot_distributions
from utils.device_utils import visible_gpus_by_space
from utils.data_utils import GeneratedDataset
from model.sample import sneak_peek
from utils.config_utils import modelConfig
from analysis.fid_score import calculate_fid_given_paths, save_fid_stats

LOFAR_PATH = Path("/storage/tmartinez/image_data/lofar_subset")
GEN_DATA_PARENT = Path("/storage/tmartinez/image_data/generated")
ANALYSIS_RESULTS_PARENT = Path("/storage/tmartinez/analysis_results")
MODELS_PARENT = Path("/storage/tmartinez/results")

def load_Unet_model(conf, model_file=None):
    # OUTDATED
    UnetModel = ImprovedUnet if conf.use_improved_unet else Unet
    model = UnetModel.from_config(conf)
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    return model

def get_file_from_dir(dir, pattern):
    l = sorted(dir.glob(pattern))
    assert len(l) <= 1, f"Found more than one {pattern} file:\n{l}"
    assert len(l) == 0, f"Found no {pattern} file."
    return l[0]
    
def model_evaluation(model_folder, use_ema=True):
    # OUTDATED
    model_folder = Path(model_folder)

    # Load files:
    get_file = partial(get_file_from_dir, model_folder)
    try:
        conf_file = get_file("config_*.json")
    except AssertionError:
        print('No config file found. Using hyperparameters_*.pt instead.')
        try:
            conf_file = get_file("hyperparameters_*.pt")
        except AssertionError as e:
            print("No config file found. Aborting.")
            raise e
    model_file = get_file(
        f"parameters_{'ema' if use_ema else 'model'}_*.pt"
    )
    loss_file = get_file("losses_*.csv")

    # Load device
    device = torch.device("cuda", visible_gpus_by_space()[0])
    print(f"Using device {device}")

    # Load config, model and diffusion
    conf = modelConfig(**torch.load(conf_file))
    model = load_Unet_model(conf, model_file).to(device)
    diff_class = EDM_Diffusion if conf.model_type == "EDMPrecond" else Diffusion
    diffusion = diff_class.from_config(conf)

    # Prepare timer
    t0 = datetime.now()
    dt = lambda: datetime.now() - t0

    # Plot losses
    fig, _ = plot_losses(loss_file, title=conf.model_name)
    fig.savefig(model_folder / f"losses_{conf.model_name}.pdf")

    # Plot ddpm samples
    t0 = datetime.now()
    imgs = diffusion.p_sampling(model, conf.image_size, batch_size=25)[-1]
    t_sample = dt()
    title = f"{conf.model_name} (DDPM) --- T={conf.timesteps} " \
            f"[{t_sample.total_seconds():.2f}s for sampling]"
    fig, ax = plot_samples(imgs, title=title)
    fig.savefig(model_folder / f"samples_ddpm_{conf.model_name}.pdf")

    # Plot ddim samples
    t0 = datetime.now()
    imgs = diffusion.ddim_sampling(model, conf.image_size, batch_size=25)[-1]
    t_sample = dt()
    title = f"{conf.model_name} (DDIM) --- T={conf.timesteps} " \
            f"[{t_sample.total_seconds():.2f}s for sampling]"
    fig, _ = plot_samples(imgs, title=title)
    fig.savefig(model_folder / f"samples_ddim_{conf.model_name}.pdf")

def get_distributions(img_dir, 
                      intensity_bins=256,
                      pixsum_bins=None,
                      act_bins=None, force=False, save=True,
                      act_threshold=0.,):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{img_dir.name}_distr.pt"
    # Look for existing file
    if save and not force and out_file.exists():
        print(f"Found existing distribution file for {img_dir.name}.")
        n_images = len(list(img_dir.glob("*.png")))
        return torch.load(out_file), n_images
    
    # Load data
    dataset = GeneratedDataset(img_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=1, drop_last=False,
    )
    
    std_sum_bins = torch.tensor(
        np.linspace(0, 6400, num=6400//50, endpoint=True)
    ).to(torch.float32)
    if pixsum_bins is None:
        pixsum_bins = std_sum_bins
    if act_bins is None:
        act_bins = std_sum_bins

    device = torch.device('cuda', visible_gpus_by_space()[0])
        
    get_nbins = lambda bins: bins if isinstance(bins, int) else len(bins) - 1
    intensity_hist = torch.zeros(get_nbins(intensity_bins)).to(int).to(device)
    pixsum_hist = torch.zeros(get_nbins(pixsum_bins)).to(device)
    act_hist = torch.zeros(get_nbins(act_bins)).to(device)

    for i, img in tqdm(enumerate(dataloader), desc="Calculating distributions..."):
        assert img.shape[1] == 1, f"Expected single channel images,"\
                                  f"got {img.shape[1]} channels."
        img.to(device)
        # Pixel intensities
        kw = {'range': (0,1)} if isinstance(intensity_bins, int) else {}
        i_count, i_edges = torch.histogram(img, bins=intensity_bins, **kw)
        intensity_hist = torch.add(intensity_hist, i_count.to(int).to(device))
        kw = {'range': (0,6400)} if isinstance(pixsum_bins, int) else {}
        p_count, p_edges = torch.histogram(torch.sum(img, (-1, -2)),
                                            bins=pixsum_bins, **kw)
        pixsum_hist = torch.add(pixsum_hist, p_count.to(device))
        # Activated pixels
        kw = {'range': (0,6400)} if isinstance(act_bins, int) else {}
        n_act = torch.sum((img>act_threshold).squeeze(), [-2, -1]).to(torch.float32)  # Shape: [b]
        act_count, act_edges = torch.histogram(n_act, bins=act_bins, **kw)
        act_hist = torch.add(act_hist, act_count.to(device))
        
    out = {
        "Pixel Intensity": (intensity_hist.to('cpu'), i_edges.to('cpu')),
        "Pixel Sum": (pixsum_hist.to('cpu'), p_edges.to('cpu')),
        "Activated Pixels": (act_hist.to('cpu'), act_edges.to('cpu'))
    }

    if save:
        torch.save(out, out_file)

    return out, len(dataset)

def get_distributions_lofar(**kwargs):
    return get_distributions(LOFAR_PATH, **kwargs)

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

def get_distribution_plot_with_lofar(img_dir, save=True, **distribution_kwargs):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    gen_stats, _ = get_distributions(img_dir, save=save, 
                                         **distribution_kwargs)
    lofar_stats, _ = get_distributions_lofar(save=save, 
                                                 **distribution_kwargs)

    error = per_bin_error(gen_stats, lofar_stats)

    fig = plot_distributions(gen_stats, lofar_stats, error,
                             title=img_dir.name)

    if save:
        fig.savefig(out_dir / f"{out_dir.name}_lofar_stats_distr.pdf")
    return fig

def get_FID_stats(img_dir, force=False):
    out_dir = ANALYSIS_RESULTS_PARENT / img_dir.name
    out_file = out_dir / f"{img_dir.name}_fid_stats.npz"
    if out_file.exists():
        if not force:
            print(f"Found existing FID stats file for {img_dir.name}.")
            return out_file
        else:
            out_file.unlink()  # Remove

    n_imgs = len(list(img_dir.glob("*.png")))
    assert n_imgs > 2048, f"Found only {n_imgs} images in {img_dir}."
    save_fid_stats([str(img_dir), str(out_file)],
                    device=torch.device('cuda', visible_gpus_by_space()[0]),
                    batch_size=64,
                    dims=2048)
    return out_file

def calculate_FID_lofar(img_dir, force=False):
    fid_stats = get_FID_stats(img_dir, force=force)
    lofar_stats = get_FID_stats(LOFAR_PATH)
    fid = calculate_fid_given_paths(
        [str(fid_stats), str(lofar_stats)],
        device=torch.device('cuda', visible_gpus_by_space()[0]),
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
    print(f"Calculating distributions for {img_dir.name}...")
    stats, n_gen = get_distributions(img_dir, force=force)
    # Plot with lofar
    print(f"Plotting distributions for {img_dir.name}...")
    fig = get_distribution_plot_with_lofar(img_dir)
    # Get FID
    print(f"Calculating FID score for {img_dir.name}...")
    n_imgs = len(list(img_dir.glob("*.png")))
    if n_imgs < 2048:
        print(f"Found only {n_imgs} images in {img_dir}, need 2048 for FID."\
              f" No FID score will be calculated.")
        fid = None
    else:
        fid = calculate_FID_lofar(img_dir, force=force)
    # Get W1
    print(f"Calculating W1 score for {img_dir.name}...")
    W1 = get_W1_lofar_score(img_dir, force=force)

    scores = {
        "FID": fid,
        "W1": W1
    }

    with open(out_dir / f"{img_dir.name}_scores.json", 'w') as f:
        json.dump(scores, f)

    # Generate plot with 16 images from folder
    example_plot_file = out_dir / f"{img_dir.name}_example_plot.pdf"
    if not example_plot_file.exists() or force:
        fig = example_plot(img_dir, example_plot_file)
    
    return fig, fid, W1

def example_plot(img_dir, example_plot_file=None):
    example_imgs = random.sample(sorted(img_dir.glob("*.png")), 25)
    example_imgs = np.array(
            [ToTensor()(Image.open(f))[:1,:,:] for f in example_imgs]
        )
    fig, _ = plot_samples(torch.Tensor(example_imgs),
                               title=img_dir.name, vmin=0)
    if example_plot_file is not None:
        fig.savefig(example_plot_file)
    return fig

def generate_samples_and_evaluate(model_dir, **sampling_kwargs):
    assert model_dir.exists(), f"Model directory {model_dir} does not exist."
    sample_set_from_folder(model_dir, **sampling_kwargs)
    return image_data_evaluation(GEN_DATA_PARENT / model_dir.name)

if __name__ == "__main__":

    # get_distributions_lofar(force=True)

    # List all folders in GEN_DATA_PARENT
    folders = sorted(GEN_DATA_PARENT.glob("*"))

    # Evaluate all folders
    for folder in folders:
        print(f"\n\nEvaluating {folder.name}...")
        image_data_evaluation(folder, force=True)