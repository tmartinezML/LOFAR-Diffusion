from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, CenterCrop
from torchvision.utils import save_image

from model.unet import Unet
from model.firstgalaxydata import FIRSTGalaxyData
from model.diffusion import Diffusion


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


if __name__ == "__main__":
    print(
        "\n\n\n######################" \
        "DDPM Workout" \
        "######################"
    )
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    print("Prepare training...")

    image_size = 32
    channels = 1
    batch_size = 12
    epochs = 6
    timesteps = 10
    learning_rate = 1e-4
    save_and_sample_every = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Working on: {device}")

    diffusion = Diffusion(timesteps)

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    transform = Compose([
        ToTensor(),
        CenterCrop(image_size),
        lambda x: x * 2 - 1,  # Scale to [-1, 1]
    ])
    data = FIRSTGalaxyData(root="./", transform=transform,
                           selected_split="train")
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    t0 = datetime.now()
    def dt(): return datetime.now() - t0
    print(f"{t0.strftime('%H:%M:%S')} - Start training...")

    losses = []

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = batch[0].to(device)  # We don't need the labels here

            # Algorithm 1, line 3: sample t uniformally for every example in
            # the batch.
            t = torch.randint(0, timesteps, (batch.shape[0],),
                              device=device).long()

            try:
                loss = diffusion.p_losses(model, batch, t, loss_type="huber")
            except RuntimeError:
                print(f"Error at epoch {epoch}, step {step}, \
                       batch shape {batch.shape}")
                raise

            if step % 100 == 0:
                print(f"Running {dt()} \
                      - epoch {epoch}, step {step} - loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()

            losses.append([epoch, step, loss.item()])

            # save generated images
            if True: #step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                all_images = diffusion.sample(model, image_size=image_size, 
                                              batch_size=4, channels=channels)
                # Shape: n_steps, 4, 128, 128
                every_n = timesteps // 5  # Only 5 images in example
                all_images = torch.cat(
                    all_images[::every_n, :, :, :],  # Every n_th timestep
                    all_images[-1, :, :, :]  # Last time step, i.e. final image
                )
                all_images = all_images.view(-1, 1, 128, 128)
                all_images = (all_images + 1) / 0.5

                outname = str(
                    results_folder/f"sample_epoch{epoch}-{milestone}.png"
                )
                save_image(all_images, outname, nrow=4)
            break
        break


    t0_string = t0.strftime("%y-%m-%d_%H:%M:%S")
    np.savetxt(f"/work/bbd0953/DDPM/results/losses_{t0_string}.csv",
               np.array(losses), header="epoch;step;loss", delimiter=";",
               fmt="%.6f")

    print(f"Training time {dt()} - Done!")

    # Save model
    torch.save(model.state_dict(),
               f"/work/bbd0953/DDPM/results/DDPM_{t0_string}.pt")
