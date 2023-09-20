from pathlib import Path
from datetime import datetime

import torch

from model.unet import Unet, ImprovedUnet
from model.train import hyperParams
from model.diffusion import Diffusion
from utils.plot_utils import plot_losses, plot_samples
from utils.utils import get_free_gpu

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
    diffusion = Diffusion.from_hyperparams(hp)

    # Prepare timer
    t0 = datetime.now()
    dt = lambda: datetime.now() - t0

    # Plot losses
    fig, _ = plot_losses(loss_file, title=hp.model_name)
    fig.savefig(model_folder / f"losses_{hp.model_name}.pdf")

    # Plot ddpm samples
    t0 = datetime.now()
    imgs = diffusion.sample(model, hp.image_size, batch_size=25)[-1]
    t_sample = dt()
    title = f"{hp.model_name} (DDPM) --- T={hp.timesteps} " \
            f"[{t_sample.total_seconds():.2f}s for sampling]"
    fig, ax = plot_samples(imgs, title=title)
    fig.savefig(model_folder / f"samples_ddpm_{hp.model_name}.pdf")

    # Plot ddim samples
    t0 = datetime.now()
    imgs = diffusion.ddim(model, hp.image_size, batch_size=25)[-1]
    t_sample = dt()
    title = f"{hp.model_name} (DDIM) --- T={hp.timesteps} " \
            f"[{t_sample.total_seconds():.2f}s for sampling]"
    fig, _ = plot_samples(imgs, title=title)
    fig.savefig(model_folder / f"samples_ddim_{hp.model_name}.pdf")

if __name__ == "__main__":
    model_folder = "/home/bbd0953/diffusion/results/OptiModel_ImprovedUnet"
    # model_evaluation(model_folder)
    print(Path.cwd())