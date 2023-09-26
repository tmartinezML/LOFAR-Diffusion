import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.model_utils import construct_from_config


def cosine_beta_schedule(timesteps, s=8e-3):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = (x / timesteps + s) / (1 + s) * torch.pi * 0.5
    alphas_cumprod = torch.cos(alphas_cumprod)**2
    alphas_cumprod /= alphas_cumprod.clone()[0]
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

scheduler = {
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sigmoid": sigmoid_beta_schedule,
}


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

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between two normal distributions.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # Force variances to be tensors.
    kl = 0.5 * (
        logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2)**2 * torch.exp(-logvar2)
        - 1.
    )
    return kl

def approx_standard_normal_cdf(x):
    """
    Approximation of the standard normal CDF.
    """
    return 0.5 * (1. + torch.tanh(np.sqrt(2. / np.pi) * (x + 0.044715 * x**3)))

def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Used to calculate L_0 term in vlb loss.
    See Ho et al. (2020), Section 3.3.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min,
            torch.log(torch.clamp(cdf_delta, min=1e-12))
        )
    )
    return log_probs

class Diffusion():
    def __init__(self, timesteps, schedule="linear", learn_variance=False):
        self.timesteps = timesteps
        self.betas = scheduler[schedule](timesteps=timesteps)
        self.learn_variance = learn_variance

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
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod
        )
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1. / self.alphas_cumprod - 1
        ) 
        # calculations for posterior q(x_{t-1} | x_t, x_0))
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev)
            / (1. - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((self.posterior_variance[1].unsqueeze(0),
                      self.posterior_variance[1:]))
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev)
            / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)
            / (1. - self.alphas_cumprod)
        )

    @classmethod
    def from_config(cls, conf):
        return construct_from_config(cls, conf)
        

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract(self.posterior_variance, t, x_t.shape)
        log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return mean, variance, log_variance_clipped
    
    def p_mean_variance(self, model, x, t):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)

        if self.learn_variance:
            assert model_output.shape == (B, 2*C, *x.shape[2:])
            predicted_noise, model_log_var = torch.split(model_output, C, dim=1)
            model_var = torch.exp(model_log_var)
        else:
            predicted_noise = model_output
            model_var = extract(
                self.posterior_variance, t, x.shape
            )
            model_log_var = extract(
                self.posterior_log_variance_clipped, t, x.shape
            )

        x0_pred = self._x0_from_epsilon(x_t=x, t=t, eps=predicted_noise)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=x0_pred, x_t=x, t=t
        )
        return model_mean, model_var, model_log_var, x0_pred

    def _x0_from_epsilon(self, x_t, t, eps):
        assert eps.shape == x_t.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _epsilon_from_x0(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def _vb_terms_bpd(
        self, model, x_start, x_t, t, 
    ):
        # True and predicted mean and variance
        true_mean, _, true_log_var_clipped = self.q_posterior_mean_variance(
            x_start, x_t, t
        )
        model_mean, _, model_log_var, x0_pred = self.p_mean_variance(
            model, x_t, t
        )

        # KL divergence for normal distributions with above parameters
        kl = normal_kl(
            true_mean, true_log_var_clipped, model_mean, model_log_var
        )
        kl = kl.mean(dim=list(range(1, kl.dim())))  # Average over batch
        kl /= np.log(2.)  # Convert to bits per dimension
        
        # L0 term
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, model_mean, model_log_var*0.5
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = decoder_nll.mean(dim=list(range(1, decoder_nll.dim())))
        decoder_nll /= np.log(2.)

        # Assign L0 term to output with t=0, KL to all other timesteps
        output = torch.where((t == 0), decoder_nll, kl)
        return output, x0_pred

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

        model_output = denoise_model(x_noisy, t)

        if loss_type == 'hybrid':
            B, C = x_noisy.shape[:2]
            assert model_output.shape == (B, 2*C, *x_noisy.shape[2:])
            predicted_noise, model_log_var = torch.split(model_output, C, dim=1)
            frozen_out = torch.cat(
                [predicted_noise.detach(), model_log_var.detach()], dim=1
            )
            frozen_model = lambda x, t: frozen_out
            # Variational bound loss
            vb, _ = self._vb_terms_bpd(
                frozen_model, x_start, x_noisy, t
            )
            vb = vb * self.timesteps * 1e-3  # 1e-3 for scaling loss term
            # MSE loss
            mse = F.mse_loss(noise, predicted_noise)
            return mse + vb.mean()
        
        predicted_noise = model_output
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
    def p_sample_step(self, model, x, t, t_index):
        model_mean, _, model_log_var, _ = self.p_mean_variance(
            model, x, t
        )

        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            # Algorithm2, line 4:
            return model_mean + torch.exp(0.5 * model_log_var) * noise

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
            img = self.p_sample_step(model, img, sample_ts, t)
            imgs.append(img.cpu().numpy())

        return torch.Tensor(np.array(imgs))

    @torch.no_grad()
    def p_sampling(self, model, image_size, batch_size=16, channels=1):
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(model, shape=shape)

    @torch.no_grad()
    def ddim_sample_step(self, model, x, t, t_next):

        model_mean, _, model_log_var, x0_pred = self.p_mean_variance(
            model, x, t
        )
        noise_pred = self._epsilon_from_x0(x, t, x0_pred)
        
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            x0_pred * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * noise_pred
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return sample
    
    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, inference_steps):
        device = next(model.parameters()).device

        b = shape[0]

        # Start from pure noise:
        img = torch.randn(shape, device=device)
        imgs = []

        step = self.timesteps // inference_steps
        loop_range = range(step, self.timesteps, step)

        for t in tqdm(reversed(loop_range),
                      desc="DDIM loop step", total=inference_steps):

            sample_ts = torch.full((b,), t, device=device, dtype=torch.long)
            sample_t_nexts = torch.full((b,), t-step, device=device, 
                                        dtype=torch.long)
            img = self.ddim_sample_step(model, img, sample_ts, sample_t_nexts)
            imgs.append(img.cpu().numpy())

        return torch.Tensor(np.array(imgs))
    
    @torch.no_grad()
    def ddim_sampling(self, model, image_size, batch_size=16, channels=1,
                    inference_steps=100):
        shape = (batch_size, channels, image_size, image_size)
        return self.ddim_sample_loop(model, shape=shape, 
                                       inference_steps=inference_steps)