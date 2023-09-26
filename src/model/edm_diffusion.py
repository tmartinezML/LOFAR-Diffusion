import torch
import numpy as np
from tqdm import tqdm

from model.unet import EDMPrecond
from utils.model_utils import construct_from_config, isModel

class EDM_Diffusion:
    def __init__(
        self,
        timesteps,
        P_mean = -1.2,
        P_std = 1.2,
        sigma_data = 0.5
        ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.timesteps = timesteps

    @classmethod
    def from_config(cls, config):
        return construct_from_config(cls, config)

    def sample_sigmas(self, img_batch):
        rnd_normal = torch.randn([img_batch.shape[0], 1, 1, 1],
                                 device=img_batch.device)
        sigmas = (rnd_normal * self.P_std + self.P_mean).exp()
        return sigmas

    def edm_loss(self, model, img_batch, sigmas=None):
        if sigmas is None:
            sigmas = self.sample_sigmas(img_batch)
        weight = (
            (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data)**2
        )
        n = torch.randn_like(img_batch) * sigmas
        D_yn = model(img_batch + n, sigmas)
        loss = weight * (D_yn - img_batch)**2
        return loss.mean()
    
    @torch.no_grad()
    def edm_stochastic_sampling(self,
                            model,
                            image_size,
                            batch_size=16,
                            sigma_min=2e-3,
                            sigma_max=80,
                            rho=7,
                            S_churn=0,
                            S_min=0,
                            S_max=torch.inf,
                            S_noise=1,
                            ):
        assert isModel(model, EDMPrecond), (
            "Model must be an EDMPrecond instance for stochastic EDM sampling."
        )
        device = next(model.parameters()).device
        latents = torch.randn(
            [batch_size, 1, image_size, image_size], device=device
        )
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

        # Time steps
        step_inds = torch.arange(self.timesteps, device=device)
        rho_inv = 1 / rho
        t_steps = (
            (sigma_max**rho_inv + step_inds / (self.timesteps - 1) 
            * (sigma_min**rho_inv - sigma_max**rho_inv))**rho
        )
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N=0

        # Sampling loop.
        imgs = []
        x_next = latents * t_steps[0]  # Generate initial sample at t_0
        imgs.append(x_next)
        input_shape = lambda t: torch.full((batch_size,), t.item())
        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:])),
            desc="Sampling...",
            total=self.timesteps):

            x_cur = x_next

            # Increase noise temporarily
            gamma = (
                min(S_churn / self.timesteps, np.sqrt(2) - 1)
                if S_min <= t_cur.item() <= S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            x_hat = (
                x_cur
                + (t_hat**2 - t_cur**2).sqrt()
                * S_noise * torch.randn_like(x_cur)
            )

            # Euler step
            denoised = model(x_hat, input_shape(t_hat))
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + d_cur * (t_next - t_hat)

            # Apply 2nd order correction
            if i < self.timesteps - 1:
                denoised = model(x_next, input_shape(t_next))
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (.5 * d_cur + .5 * d_prime)
            imgs.append(x_next)
        return imgs

    @torch.no_grad()
    def edm_deterministic_sampling(self,
                               model,
                               image_size,
                               batch_size=16,
                               sigma_min=2e-3,
                               sigma_max=80,
                               rho=7,
                               S_churn=0,
                               S_min=0,
                               S_max=torch.inf,
                               S_noise=1,
                               ):
        assert isinstance(model, EDMPrecond) (
            "Model must be an EDMPrecond instance for deterministic EDM sampling."
        )
        device = next(model.parameters()).device
        latents = torch.randn(
            [batch_size, 1, image_size, image_size], device=device
        )
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

        # Time steps
        step_inds = torch.arange(self.timesteps, device=device)
        rho_inv = 1 / rho
        t_steps = (
            (sigma_max**rho_inv + step_inds / (self.timesteps - 1) 
            * (sigma_min**rho_inv - sigma_max**rho_inv))**rho
        )
        # t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N=0
        
