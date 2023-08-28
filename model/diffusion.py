import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def cosine_beta_schedule(timesteps, s=8e-3):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = (x / timesteps + s) / (1 + s) * torch.pi * 0.5
    alphas_cumprod = torch.cos(alphas_cumprod)**2
    alphas_cumprod /= alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)**2


def sigmoid_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    """
    Extracts the parameters (alphas, betas, etc.) from a corresponding to time 
    steps t of a batch, returns the parameters in shape compatible with 
    operations on tensors of shape x_shape.

    Parameters
    ----------
    a : torch.Tensor
        1-d Tensor containing the parameters (alphas, betas, etc.) for all time
        steps.
    t : torch.Tensor
        1-d Tensor containing [batch_size] number of time step values.
    x_shape : torch.Tensor
        Tensor containing any number b of images, shape (b, c, h, w)

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch_size, 1, 1, 1) containing the extracted 
        parameters. Shape is chosen such that the output can be used for 
        operations on images.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *(1,) * (len(x_shape) - 1)).to(t.device)


class Diffusion():
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas:
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        # Shift everything to the right, crop last place,
        # insert 1 at first place:
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0),
                                         value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others:
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod
        )
        # calculations for posterior q(x_{t-1} | x_t, x_0))
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev)
            / (1. - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # Get parameters for given t:
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        # Sample noisy image:
        out = (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return out

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Eq. 11 in paper:
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm2, line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]

        # Start from pure noise:
        img = torch.randn(shape, device=device)
        imgs = []

        for t in tqdm(reversed(range(0, self.timesteps)),
                      desc="Sampling loop time step", total=self.timesteps):

            sample_ts = torch.full((b,), t, device=device, dtype=torch.long)
            img = self.p_sample(model, img, sample_ts, t)
            imgs.append(img.cpu().numpy())

        return torch.Tensor(imgs)

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1):
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(model, shape=shape)
