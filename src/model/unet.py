"""
Created following the tutorial on 
https://huggingface.co/blog/annotated-diffusion

"""

import math
from pathlib import Path
from functools import partial
from abc import abstractmethod

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from utils.config_utils import construct_from_config

from utils.config_utils import configModuleBase

DEBUG_DIR = Path("/home/bbd0953/diffusion/results/debug")


def clamp_tensor(x):
    """
    https://discuss.huggingface.co/t/t5-fp16-issue-is-fixed/3139
    """
    dtype = torch.half if torch.is_autocast_enabled() else x.dtype
    clamp_value = torch.finfo(dtype).max - 1000
    x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def zero_module(module):
    """
    Sets all parameters of a module to zero. Used for initializing the
    optimizers.

    Parameters
    ----------
    module : nn.Module
        Module to be zeroed.

    Returns
    -------
    nn.Module
        Zeroed module.
    """
    for param in module.parameters():
        param.detach().zero_()
    return module


def upsample(dim, dim_out=None, use_conv=True):
    """
    Upsampling layer, NxN --> 2Nx2N, using nearest neighbor algorithm.
    Basically, every pixel is quadrupled, and the new pixels are filled with
    the value of the original pixel.

    Parameters
    ----------
    dim : int
        Input channels
    dim_out : int, optional
        Output channels, by default None

    Returns
    -------
    nn.Sequential
        Upsampling layer.
    """
    return nn.Sequential(
        # Upsampling quadruples each pixel.
        nn.Upsample(scale_factor=2, mode="nearest"),
        # Convolution leaves image size unchanged.
        (nn.Conv2d(dim, (dim_out or dim), 3, padding=1) if use_conv else nn.Identity()),
    )


def downsample(dim, dim_out=None):
    """
    Downsampling layer, NxN -> N/2 x N/2. Works by splitting image into 4, then
    doing 1x1 convolution using the 4 sub-images as input channels.
    In the original U-Net, this is done by max pooling.

    Parameters
    ----------
    dim : int
        Input channels
    dim_out : int, optional
        Output channels, by default None

    Returns
    -------
    nn.Sequential
        Downsampling layer.
    """
    return nn.Sequential(
        # Rearrange: Split each image into 4 smaller and concatenate along
        # channel dimension.
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # Convolution leaves image sizes unchanged, changes channel dimensions
        # back to original (or specified dim_out).
        nn.Conv2d(dim * 4, (dim_out or dim), 1),
    )


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential module where forward() takes timestep embeddings as a second
    argument.
    """

    def forward(self, x, emb):

        for layer in self:

            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)

            else:
                x = layer(x)

        return x


class SinusoidalEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, 1) corresponding to the time values of
    the noised images, and returns embedding of shape (batch_size, dim).
    For a good explanation of the embedding formula, see:
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        freqs = math.log(1e5) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freqs)
        embeddings = time[:, None] * freqs[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FourierEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, 1) corresponding to values of a
    continuous context feature, and returns
    embedding of shape (batch_size, dim).
    For a good explanation of the embedding formula, see:
    https://bmild.github.io/fourfeat/
    """

    def __init__(self, dim, scale=16):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x):
        x = x.outer((2 * np.pi * self.freqs.to(x.device)).to(x.dtype))
        embeddings = torch.cat([x.cos(), x.sin()], dim=-1)
        return embeddings


class ContextEmbedding(nn.Module):
    """
    Takes input t of shape (batch_size, context_dim) corresponding to the
    values of the entire context, and returns embedding
    of shape (batch_size, dim), which is the sum of fourier embeddings
    of all single features.
    """

    def __init__(self, context_dim, emb_cls, **cls_kwargs):
        super().__init__()
        self.emb_layers = [emb_cls(**cls_kwargs) for _ in range(context_dim)]
        for i, layer in enumerate(self.emb_layers):
            self.add_module(f"emb_{i}", layer)

    def forward(self, x):
        print(f"Context on {x.device}")
        return sum(
            layer(x[:, i].view(-1, 1)) for i, layer in enumerate(self.emb_layers)
        )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Weight-standardized 2d convolutional layer, built from a standard
    conv2d layer. Works better with group normalization.
    https://arxiv.org/abs/1903.10520
    https://kushaj.medium.com/weight-standardization-a-new-normalization-in-town-54b2088ce355 #noqa
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3  # Epsilon
        weight = self.weight

        # Tensors with mean and variance, same shape as weight tensors
        # for subsequent operations.
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")  # o = outp. channels
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        out = F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        # Clamp inf values to avoid Infs/NaNs
        if torch.isinf(out).any():
            out = clamp_tensor(out)

        return out


class ResidualLinearAttention(nn.Module):
    """
    Basically the same as regular multi-head attention, but this implementation
    is more efficient, (linear vs quadratic).
    To be exact, when using softmax this is not precisely mathematically
    equivalent, but a very good approximation.
    https://arxiv.org/abs/1812.01243

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.pre_norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=True)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, bias=True), nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        res = x
        _, _, h, w = x.shape

        # qkv: Tuple of 3 tensors of shape [b, dim_head*heads, h, w]:
        x = self.pre_norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # Reshape to three tensors of shape [b, heads, dim_head, h*w]:
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q * self.scale

        # Trick to prevent overflow in softmax
        q_shift = q - q.amax(dim=-1, keepdim=True).detach()
        k_shift = k - k.amax(dim=-2, keepdim=True).detach()

        q_norm = q_shift.softmax(dim=-1)
        k_norm = k_shift.softmax(dim=-2)

        context = torch.einsum("b h d n, b h e n -> b h d e", k_norm, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q_norm)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out) + res


class ResidualBlock(TimestepBlock):
    """
    BigGAN Block implementation
    """

    def __init__(
        self,
        dim,
        dim_out,
        time_emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
        up=False,
        down=False,
        use_conv=False,
    ):
        super().__init__()

        self.resize = up or down
        self.do_res_conv = dim != dim_out

        # Input layers
        self.in_layers = nn.Sequential(
            nn.GroupNorm(norm_groups, dim),
            nn.SiLU(),
            WeightStandardizedConv2d(dim, dim_out, 3, padding=1, bias=True),
        )

        # Resampling layers
        if up:
            self.h_upd = upsample(dim, use_conv=False)
            self.x_upd = upsample(dim, use_conv=False)

        elif down:
            self.h_upd = downsample(dim)
            self.x_upd = downsample(dim)

        # Time embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2),
        )

        # Output layers
        self.out_layers = nn.Sequential(
            nn.GroupNorm(norm_groups, dim_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                WeightStandardizedConv2d(dim_out, dim_out, 3, padding=1, bias=True)
            ),
        )

        # Residual layers
        if self.do_res_conv:
            self.res_conv = (
                WeightStandardizedConv2d(dim, dim_out, 3)
                if use_conv
                else nn.Conv2d(dim, dim_out, 1)
            )

    def forward(self, x, time_emb):

        # Input Layers
        if self.resize:
            # Split input layers into Norm+SiLU and Conv
            # (up/downsample will happen in between)
            in_pre, in_conv = self.in_layers[:-1], self.in_layers[-1]

            # Hidden state
            h = in_pre(x)  # Norm+SiLU
            h = self.h_upd(h)  # Up/downsample
            h = in_conv(h)  # Conv

            # Residual
            x = self.x_upd(x)  # Up/downsample residual

        else:
            h = self.in_layers(x)

        # Time embedding
        time_emb = self.emb_layers(time_emb)  # SiLU and Linear
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale, shift = time_emb.chunk(2, dim=1)

        # Output layers
        # Split output layers into Norm and SiLU+Dropout+Conv
        # (time embedding will be applied in between)
        out_norm, out_post = self.out_layers[0], self.out_layers[1:]
        # Apply Norm and time embedding
        h = out_norm(h) * (scale + 1) + shift
        # Apply SiLU+Dropout+Conv
        h = out_post(h)
        # Residual layer
        if self.do_res_conv:
            x = self.res_conv(x)

        return clamp_tensor(h + x)


class ResidualBlockAttention(nn.Module):
    """
    Sequential module that applies a residual block and an attention block.
    """

    def __init__(self, resBlock, attnBlock):
        super().__init__()
        self.resBlock = resBlock
        self.attnBlock = attnBlock

    def forward(self, x, time_emb):
        x = self.resBlock(x, time_emb)
        x = self.attnBlock(x)
        return x


class DownsampleBlock(ResidualBlock):
    def __init__(
        self,
        channels,
        time_emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
    ):
        super().__init__(
            channels,
            channels,
            time_emb_dim,
            dropout=dropout,
            norm_groups=norm_groups,
            up=False,
            down=True,
            use_conv=False,
        )


class UpsampleBlock(ResidualBlock):
    def __init__(
        self,
        channels,
        time_emb_dim,
        *,
        dropout=0.0,
        norm_groups=32,
    ):
        super().__init__(
            channels,
            channels,
            time_emb_dim,
            dropout=dropout,
            norm_groups=norm_groups,
            up=True,
            down=False,
            use_conv=False,
        )


class FeatureEmbedding(nn.Module):
    """
    Time embedding module, which is used to inject time information into the
    model.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, dim_out)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        return x


class Unet(configModuleBase):
    # See:
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L396
    def __init__(
        self,
        init_channels,
        out_channels=None,
        channel_mults=(1, 2, 4, 8),
        image_channels=1,
        n_labels=0,
        context_dim=0,
        label_dropout=0,
        context_dropout=0,
        norm_groups=32,
        dropout=0,
        num_res_blocks=2,
        attention_levels=3,
        attention_heads=4,
        attention_head_channels=32,
    ):
        super().__init__()

        # Determine channels
        self.input_channels = image_channels
        self.out_channels = out_channels or image_channels
        self.init_channels = init_channels

        # Time and label embeddings
        emb_dim = init_channels * 4
        self.time_emb = SinusoidalEmbedding(emb_dim)
        self.label_emb = nn.Linear(n_labels, emb_dim, bias=False) if n_labels else None
        self.label_dropout = label_dropout
        self.n_labels = n_labels

        # Context embedding
        self.context_emb = (
            FeatureEmbedding(context_dim, emb_dim) if context_dim else None
        )
        self.context_dim = context_dim
        self.context_dropout = context_dropout

        # Feature embedding
        self.feature_emb = FeatureEmbedding(emb_dim, emb_dim)

        # Initial convolution layer
        self.init_conv = WeightStandardizedConv2d(
            self.input_channels, self.init_channels, 3, padding=1
        )
        input_block_chans = [self.init_channels]

        # Create lists of down- and up-blocks
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        ch = self.init_channels
        n_levels = len(channel_mults)

        # Fill down-block list
        for res_level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                # Create residual block
                block = ResidualBlock(
                    ch,
                    int(mult * init_channels),
                    emb_dim,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                ch = int(mult * init_channels)

                # Add attention layer to block if necessary
                if n_levels - res_level <= attention_levels:
                    attn = ResidualLinearAttention(
                        ch, heads=attention_heads, dim_head=attention_head_channels
                    )
                    block = ResidualBlockAttention(block, attn)

                # Add residual(-attention) block to down-list
                self.down_blocks.append(block)
                input_block_chans.append(ch)

            # Add downsample block if not last resolution level
            if res_level != n_levels - 1:
                block = DownsampleBlock(
                    ch, emb_dim, dropout=dropout, norm_groups=norm_groups
                )
                self.down_blocks.append(block)
                input_block_chans.append(ch)

        # Create middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, emb_dim, dropout=dropout, norm_groups=norm_groups),
            ResidualLinearAttention(
                ch, heads=attention_heads, dim_head=attention_head_channels
            ),
            ResidualBlock(ch, ch, emb_dim, dropout=dropout, norm_groups=norm_groups),
        )

        # Fill up-blocks (loop in reverse, hence [::-1])
        for res_level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                # Get residual input channels from list
                ich = input_block_chans.pop()

                # Create residual block
                block = ResidualBlock(
                    ch + ich,
                    int(mult * init_channels),
                    emb_dim,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                ch = int(mult * init_channels)

                # Add attention layer to block if necessary
                if n_levels - res_level <= attention_levels:
                    attn = ResidualLinearAttention(
                        ch, heads=attention_heads, dim_head=attention_head_channels
                    )
                    block = ResidualBlockAttention(block, attn)

                # Add residual(-attention) block to up-list
                self.up_blocks.append(block)

                # Add upsample block if not last resolution level
                if res_level and i == num_res_blocks:
                    block = UpsampleBlock(
                        ch, emb_dim, dropout=dropout, norm_groups=norm_groups
                    )
                    self.up_blocks.append(block)

        # Final residual block
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, self.out_channels, 3, padding=1)),
        )

    def forward(self, x, time, context=None, class_labels=None):
        # Time embedding
        emb = self.time_emb(time)

        # Class label embedding
        if self.label_emb is not None and class_labels is not None:
            labels_enc = F.one_hot(class_labels, num_classes=self.n_labels).to(x.dtype)

            # Apply label dropout
            if self.training and self.label_dropout:
                mask = torch.rand([x.shape[0], 1], device=x.device)
                mask = mask >= self.label_dropout
                labels_enc = labels_enc * mask.to(labels_enc.dtype)

            # Add label embedding to time embedding & apply activation
            emb = emb + self.label_emb(labels_enc)

        # Context embedding
        if self.context_emb is not None and context is not None:
            context_emb = self.context_emb(context.to(x.dtype))

            if self.training and self.context_dropout:
                mask = torch.rand([x.shape[0], 1], device=x.device)
                mask = mask >= self.context_dropout
                context_emb = context_emb * mask.to(context_emb.dtype)

            emb = emb + context_emb

        # Feature embedding
        emb = self.feature_emb(emb)

        # Initial convolution
        x = self.init_conv(x)
        h = [x]

        # Encoder
        for module in self.down_blocks:
            x = module(x, emb)
            h.append(x)

        # Middle block
        x = self.middle_block(x, emb)

        # Decoder
        for module in self.up_blocks:
            if not isinstance(module, UpsampleBlock):
                x = torch.cat((x, h.pop()), dim=1)
            x = module(x, emb)
        return self.out(x)


class EDMPrecond(configModuleBase):
    """
    Wrapper for Unet to apply preconditioning as introduced in EDM paper.
    """

    def __init__(
        self,
        model,
        sigma_min=0,
        sigma_max=torch.inf,
        sigma_data=0.5,
    ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    @classmethod
    def from_config(cls, config):
        model = Unet.from_config(config)
        return construct_from_config(cls, config, model=model)

    def forward(self, x, sigma, context=None, class_labels=None):

        # Expand sigma to shape [batch_size, 1, 1, 1]
        sigma = sigma.view([-1, 1, 1, 1])

        # Weight coefficients for each term
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        # Apply inner model
        F_x = self.model(
            c_in * x, c_noise.flatten(), context=context, class_labels=class_labels
        )

        # Generate denoiser output
        D_x = c_skip * x + c_out * F_x

        return D_x
