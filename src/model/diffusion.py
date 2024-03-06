import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model.unet import EDMPrecond
from utils.config_utils import construct_from_config
from utils.config_utils import isModel


class Diffusion:
    def __init__(
        self,
        timesteps,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.timesteps = timesteps

    @classmethod
    def from_config(cls, config):
        return construct_from_config(cls, config)

    def sample_sigmas(self, img_batch):
        rnd_normal = torch.randn(
            [img_batch.shape[0], 1, 1, 1], device=img_batch.device
        )
        sigmas = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigmas

    def edm_loss(self, model, img_batch,
                 sigmas=None, noise=None, class_labels=None,
                 return_output=False, mean=True
                 ):

        # Set noise vector
        if noise is not None:
            assert sigmas is not None, (
                "If noise is provided, sigmas must be provided."
            )
            n = noise

        else:
            sigmas = sigmas or self.sample_sigmas(img_batch)
            n = torch.randn_like(img_batch) * sigmas

        # Weight coefficient for loss, as introduced in EDM paper
        weight = (
            (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data)**2
        )

        # Compute denoised image with forward model pass
        D_yn = model(img_batch + n, sigmas, class_labels=class_labels)

        # Compute loss
        loss = weight * (D_yn - img_batch)**2
        if mean:
            loss = loss.mean()

        return (loss, D_yn) if return_output else loss

    @classmethod
    def sampling_noise_levels(cls, timesteps,
                              sigma_min=2e-3, sigma_max=80, rho=7
                              ):
        # Time step indices
        step_inds = torch.arange(timesteps)

        # Noise level for each time step
        rho_inv = 1 / rho
        sigma_steps = (
            (sigma_max**rho_inv + step_inds / (timesteps - 1)
             * (sigma_min**rho_inv - sigma_max**rho_inv))**rho
        )

        # Add t_N=0 at the end
        sigma_steps = torch.cat(
            [sigma_steps, torch.zeros_like(sigma_steps[:1])]
        )

        return step_inds, sigma_steps

    @torch.no_grad()
    def denoised_guided(self, model, img, sigma,
                        class_labels=None, guidance_strength=0.2
                        ):
        # Set batch size
        batch_size = img.shape[0]
        sigma = sigma.expand(batch_size)

        # Calculate denoised image with forward model pass
        denoised = model(img, sigma)

        # Apply class conditioning if labels are provided
        if class_labels is not None:

            # Denoised image with class conditioning
            denoised_cond = model(
                img,
                sigma,
                class_labels=class_labels
            )

            # Apply guidance
            denoised = (
                (1 + guidance_strength) * denoised_cond
                - guidance_strength * denoised
            )

        return denoised

    @torch.no_grad()
    def edm_sampling(self,
                     model,
                     image_size,
                     label_batch=None,
                     guidance_strength=0.2,
                     batch_size=16,
                     sigma_min=2e-3,
                     sigma_max=80,
                     rho=7,
                     S_churn=0,
                     S_min=0,
                     S_max=torch.inf,
                     S_noise=1,
                     ):

        # S_churn = 0 is deterministic sampling
        assert isModel(model, EDMPrecond), (
            "Model must be an EDMPrecond instance for stochastic EDM sampling."
        )

        # Set device
        device = next(model.parameters()).device
        print('Sampling on device:', device)

        # Sample seed for x_0 from normal distribution. Will be scaled with
        # noise level before the sampling loop.
        latents = torch.randn(
            [batch_size, 1, image_size, image_size], device=device
        )

        # Get sigma_min and sigma_max from model, which may further restrict
        # the range of noise levels.
        model_sigma_min = (
            model.module.sigma_min if isinstance(model, torch.nn.DataParallel)
            else model.sigma_min
        )
        model_sigma_max = (
            model.module.sigma_max if isinstance(model, torch.nn.DataParallel)
            else model.sigma_max
        )
        sigma_min = max(sigma_min, model_sigma_min)
        sigma_max = min(sigma_max, model_sigma_max)

        # Time steps (= noise levels).
        step_inds, sigma_steps = self.sampling_noise_levels(
            self.timesteps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
        )

        # Move all tensors to gpu
        step_inds = step_inds.to(device)
        sigma_steps = sigma_steps.to(device)

        if label_batch is not None:
            assert label_batch.shape[0] == batch_size, (
                "Batch size must match label batch size."
            )
            label_batch = label_batch.to(device)

        # Prepare sampling loop.
        imgs = []
        x_next = latents * sigma_steps[0]  # Generate initial sample at t_0
        imgs.append(x_next.cpu())

        # Sampling loop:
        for i, (sigma_cur, sigma_next) in tqdm(
                enumerate(zip(sigma_steps[:-1], sigma_steps[1:])),
                desc="Sampling...",
                total=self.timesteps
        ):
            x_cur = x_next

            # Stochastic sampling: Increase noise temporarily
            if S_churn > 0:
                sigma_cur, x_cur = self.stochastic_churn(
                    S_churn, S_min, S_max, S_noise, sigma_cur, x_cur
                )

            # Calculate denoised image with forward model pass
            denoised = self.denoised_guided(
                model, x_cur, sigma_cur, class_labels=label_batch,
                guidance_strength=guidance_strength
            )

            # Score estimate
            d_cur = (x_cur - denoised) / sigma_cur

            # Euler step
            x_next = x_cur + d_cur * (sigma_next - sigma_cur)

            # Apply 2nd order correction
            if i < self.timesteps - 1:

                # Denoised image for next step
                denoised = self.denoised_guided(
                    model, x_next, sigma_next, class_labels=label_batch,
                    guidance_strength=guidance_strength
                )

                # Score estimate for next step
                d_next = (x_next - denoised) / sigma_next

                # 2nd order correction
                x_next = (
                    x_cur
                    + (sigma_next - sigma_cur) * (.5 * d_cur + .5 * d_next)
                )

            # Append to list
            imgs.append(x_next.cpu())

        return imgs

    def stochastic_churn(self, S_churn, S_min, S_max, S_noise, sigma_cur, x_cur):
        gamma = (  # = 0 if S_churn = 0 (deterministic sampling)
            min(S_churn / self.timesteps, np.sqrt(2) - 1)
            if S_min <= sigma_cur.item() <= S_max
            else 0
        )
        sigma_hat = (1 + gamma) * sigma_cur  # = sigma_cur if gamma = 0
        x_hat = (  # = x_cur if gamma = 0
            x_cur
            + (sigma_hat**2 - sigma_cur**2).sqrt()  # = 0 if gamma = 0
            * S_noise * torch.randn_like(x_cur)
        )

        return sigma_hat, x_hat
