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
from model.config_utils import construct_from_config

from model.config_utils import configModuleBase

DEBUG_DIR = Path("/home/bbd0953/diffusion/results/debug")


def upsample(x):
    c = x.shape[1]
    return F.conv_transpose2d(
        x, torch.ones(c, 1, 2, 2).to(x.device, dtype=x.dtype), stride=2, groups=c
    ).to(x.device)


def downsample(x):
    c = x.shape[1]
    return F.conv2d(
        x,
        (1 / 4 * torch.ones(c, 1, 2, 2)).to(x.device, dtype=x.dtype),
        stride=2,
        groups=c,
    ).to(x.device)


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def mp_silu(x):
    return F.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


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


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        kernel = [kernel_size, kernel_size] if kernel_size else []
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight = torch.nn.Parameter(
                    normalize(w)
                )  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class LinearAttention(nn.Module):
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

    def __init__(self, dim, dim_head=32, balance=0.3):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = dim // dim_head
        self.balance = balance
        self.to_qkv = MPConv(dim, dim * 3, 1)
        self.to_out = MPConv(dim, dim, 1)

    def forward(self, x):
        # qkv in single tensor of shape [b, 3*dim_head*heads, h, w]:
        y = self.to_qkv(x)

        # Reshape to shape [b, heads, dim_head, 3, h*w]:
        y = y.reshape(y.shape[0], self.heads, -1, 3, y.shape[-2] * y.shape[-1])

        # Pixel norm & split into q, k, v:
        q, k, v = normalize(y, dim=2).unbind(3)

        # Attention weight matrix:
        w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)

        # Apply weight matrix to keys to obtain context:
        y = torch.einsum("nhqk,nhck->nhcq", w, v)

        # Reshape back to [b, dim, h, w] and apply out conv:
        y = self.to_out(y.reshape(*x.shape))

        # Weighted sum with residual connection
        x = mp_sum(x, y, t=self.balance)
        return x


class Block(torch.nn.Module):
    """
    BigGAN Block implementation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        *,
        block_type="enc",  # "enc" or "dec"
        resample=None,  # "up", "down", None
        res_balance=0.3,
        attention=False,
        attn_balance=0.3,
        attn_head_dim=32,
        dropout=0.0,
        clip_act=256,
    ):
        super().__init__()

        # Assign attributes
        assert block_type in ["enc", "dec"], f"Invalid block type: {block_type}."
        self.block_type = block_type
        assert resample in [None, "up", "down"], f"Invalid resample type: {resample}."
        self.resample = resample
        self.out_channels = out_channels
        self.dropout = dropout
        self.res_balance = res_balance
        self.clip_act = clip_act

        # Initialize modules
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv0 = MPConv(
            out_channels if block_type == "enc" else in_channels, out_channels, 3
        )
        self.emb_linear = MPConv(time_emb_dim, out_channels, 0)
        self.conv1 = MPConv(out_channels, out_channels, 3)
        self.conv_skip = (
            MPConv(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.attn = (
            LinearAttention(out_channels, dim_head=attn_head_dim, balance=attn_balance)
            if attention
            else None
        )

    def forward(self, x, emb):
        # Resample:
        match self.resample:
            case "up":
                x = upsample(x)
            case "down":
                x = downsample(x)
            case None:
                pass

        # For encoder block, apply input convolution & normalize:
        if self.block_type == "enc":
            x = self.conv_skip(x) if self.conv_skip is not None else x
            x = normalize(x, dim=1)

        # Apply first convolution & embedding:
        y = mp_silu(x)
        y = self.conv0(y)
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv1(y)

        # Skip connection:
        if self.block_type == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention:
        if self.attn is not None:
            x = self.attn(x)

        # Clip activations
        if self.clip_act is not None:
            x = torch.clamp(x, -self.clip_act, self.clip_act)

        return x


class MPFeatureEmbedding(nn.Module):
    """
    Time embedding module, which is used to inject time information into the
    model.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin1 = MPConv(dim_in, dim_out, 0)
        self.lin2 = MPConv(dim_out, dim_out, 0)

    def forward(self, x):
        x = self.lin1(x)
        x = mp_silu(x)
        x = self.lin2(x)
        x = mp_silu(x)
        return x


class Unet(configModuleBase):
    # See:
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L396
    def __init__(
        self,
        init_channels,
        out_channels=None,
        channel_mults=(1, 2, 2, 2),
        image_res=80,
        image_channels=1,
        n_labels=0,
        context_dim=0,
        label_dropout=0,
        context_dropout=0,
        dropout=0,
        num_blocks=2,
        attn_levels=2,
        attn_head_dim=32,
        context_balance=0.5,
        concat_balance=0.5,
    ):
        super().__init__()

        # Determine channels
        self.input_channels = image_channels
        self.out_channels = out_channels or image_channels
        self.init_channels = init_channels
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))
        chan_block = [init_channels * m for m in channel_mults]

        # Noise and label embeddings
        emb_dim = init_channels * 4
        self.noise_emb = MPFourier(emb_dim)
        self.label_emb = MPConv(n_labels, emb_dim, 0) if n_labels else None
        self.label_dropout = label_dropout
        self.n_labels = n_labels

        # Context embedding
        self.context_emb = (
            MPFeatureEmbedding(context_dim, emb_dim) if context_dim else None
        )
        self.context_dim = context_dim
        self.context_dropout = context_dropout
        self.context_balance = context_balance

        # Feature embedding
        self.feature_emb = MPFeatureEmbedding(emb_dim, emb_dim)

        # Encoder
        self.enc = torch.nn.ModuleDict()
        chan_out = image_channels + 1
        for level, channels in enumerate(chan_block):
            res = image_res >> level  # Binary shift --> division by 2**level

            # Initial convolution
            if level == 0:
                chan_in = chan_out
                chan_out = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(chan_in, chan_out, 3)

            # Downsample
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    chan_out,
                    chan_out,
                    emb_dim,
                    block_type="enc",
                    resample="down",
                    dropout=dropout,
                )

            # Blocks
            for block in range(num_blocks):
                chan_in = chan_out
                chan_out = channels
                self.enc[f"{res}x{res}_block{block}"] = Block(
                    chan_in,
                    chan_out,
                    emb_dim,
                    block_type="enc",
                    attention=(len(chan_block) - level) <= attn_levels,
                    dropout=dropout,
                    attn_head_dim=attn_head_dim,
                )

        # Decoder
        self.dec = torch.nn.ModuleDict()
        chan_skips = [block.out_channels for block in self.enc.values()]

        for level, channels in reversed(list(enumerate(chan_block))):
            res = image_res >> level  # Binary shift --> division by 2**level

            # Middle block level
            if level == len(chan_block) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(
                    chan_out,
                    chan_out,
                    emb_dim,
                    block_type="dec",
                    attention=True,
                    dropout=dropout,
                    attn_head_dim=attn_head_dim,
                )
                self.dec[f"{res}x{res}_in1"] = Block(
                    chan_out,
                    chan_out,
                    emb_dim,
                    block_type="dec",
                    dropout=dropout,
                    attn_head_dim=attn_head_dim,
                )

            # Upsample
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    chan_out,
                    chan_out,
                    emb_dim,
                    block_type="dec",
                    resample="up",
                    dropout=dropout,
                    attn_head_dim=attn_head_dim,
                )

            # Blocks
            for block in range(num_blocks + 1):
                chan_in = chan_out + chan_skips.pop()
                chan_out = channels
                self.dec[f"{res}x{res}_block{block}"] = Block(
                    chan_in,
                    chan_out,
                    emb_dim,
                    block_type="dec",
                    attention=(len(chan_block) - level) <= attn_levels,
                    dropout=dropout,
                    attn_head_dim=attn_head_dim,
                )

        # Output convolution
        self.out_conv = MPConv(chan_out, image_channels, 3)

    def forward(self, x, noise, context=None, class_labels=None):
        # Time embedding
        emb = self.noise_emb(noise)

        emb_list = []
        # Class label embedding
        if self.label_emb is not None and class_labels is not None:
            labels_enc = F.one_hot(class_labels, num_classes=self.n_labels).to(x.dtype)

            # Apply label dropout
            if self.training and self.label_dropout:
                mask = torch.rand([x.shape[0], 1], device=x.device)
                mask = mask >= self.label_dropout
                labels_enc = labels_enc * mask.to(labels_enc.dtype)

            label_emb = self.label_emb(
                labels_enc * np.sqrt(self.n_labels), t=self.context_balance
            )
            emb_list.append(label_emb)

        # Context embedding
        if self.context_emb is not None and context is not None:
            context_emb = self.context_emb(context.to(x.dtype))

            if self.training and self.context_dropout:
                mask = torch.rand([x.shape[0], 1], device=x.device)
                mask = mask >= self.context_dropout
                context_emb = context_emb * mask.to(context_emb.dtype)

            emb_list.append(context_emb)

        # Combine embeddings
        if len(emb_list) == 1:
            emb = mp_sum(emb, emb_list[0], t=self.context_balance)
        elif len(emb_list) == 2:
            emb = mp_sum(
                emb, mp_sum(emb_list[0], emb_list[1], t=0.5), t=self.context_balance
            )

        # Feature embedding
        emb = self.feature_emb(emb)

        # Encoder
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)

        # Output
        x = self.out_conv(x, gain=self.out_gain)
        return x


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
        sigma = sigma.reshape(-1, 1, 1, 1)

        # Weight coefficients for each term
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Apply inner model
        F_x = self.model(c_in * x, c_noise, context=context, class_labels=class_labels)

        # Generate denoiser output
        D_x = c_skip * x + c_out * F_x

        return D_x
