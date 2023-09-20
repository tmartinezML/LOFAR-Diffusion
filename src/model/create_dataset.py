from pathlib import Path
from model.unet import Unet
from model.diffusion import Diffusion
from torchvision.utils import save_image
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.transforms import Lambda
from torch.nn import DataParallel

def create(model, out_folder, bsize, T=1000, img_size=80, iteration=0, 
           DDIM=True):
    t0 = datetime.now()
    diffusion = Diffusion(timesteps=T)
    if DDIM:
        imgs = diffusion.ddim_sample(model, 80, batch_size=bsize)[-1]  # Only last time step
    
    else:
        imgs = diffusion.sample(model, 80, batch_size=bsize)[-1]
    
    for i, img in tqdm(enumerate(imgs), desc="Saving..."):
        rescale = Lambda(lambda t: (t + 1) / 2)
        save_image(rescale(img), 
                   out_folder.joinpath(f"{iteration}_{i:04d}.png"))

    print(f"Done in {datetime.now() - t0}")


if __name__ == "__main__":
    t0_string = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    out_folder = Path(f"./data/generated/diffusion_v1.0_parallelSampling_13824imgs")
    Path.mkdir(out_folder, exist_ok=True, parents=True)

    model_file = "/home/bbd0953/diffusion/results/DDPM_23-09-01_08:03:11_EMA.pt"

    bsize = 1536 * 3
    iterations = 10
    model = Unet(dim=160, image_channels=1, channel_mults=(1, 2, 3, 4))
    model.load_state_dict(torch.load(model_file))
    model.to('cuda:0')
    net = DataParallel(model, device_ids=[0, 1, 2])

    for i in range(iterations):
        create(net, out_folder, bsize, iteration=i, DDIM=False)

