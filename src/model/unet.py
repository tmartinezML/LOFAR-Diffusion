"""
Created following the tutorial on 
https://huggingface.co/blog/annotated-diffusion

"""
import math
from inspect import isfunction, signature
from functools import partial
from abc import abstractmethod
import logging


import torch
from torch import Tensor, einsum
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from utils.model_utils import customModelClass, construct_from_config


def default(val, d):
    """
    If val is None, returns d (or d()), otherwise returns val. Used for optional
    kwargs in return values of several functions.

    Parameters
    ----------
    val : 
        If not None, return value of the function
    d : 
        Return value of the function if val is None. d itself can be a 
        function, in which case d() will be returned.

    Returns
    -------
        val, d or d()
    """
    if val is not None:
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    """
    Adds the input to the output of a particular function. This class can be
    used to add a residual connection to any module. 
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        Forward pass, simply adding inpout to output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of function with input tensor added.
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None, use_conv=True):
    """
    Upsampling layer, NxN --> 2Nx2N

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
        (
            nn.Conv2d(dim, default(dim_out, dim), 3, padding=1) if use_conv
            else nn.Identity()
        ),
    )


def Downsample(dim, dim_out=None):
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
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
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
        embeddings = math.log(1e5) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


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
        var = reduce(weight, "o ... -> o 1 1 1",
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    """
    Basic building block contained in a resnet block.
    Consists of weight-standardized Conv2d, followed by group norm and SiLU
    activation. Also, if scale_shift parameters are passed to the forward()
    function, a FiLM modulation is applied. In the context of Diffusion, this 
    represents the injection of time information.
    For FiLM, see:
    https://distill.pub/2018/feature-wise-transformations/
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # FiLM layer if time information is available:
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    ResNet block, which is the core building block of the U-Net.
    https://arxiv.org/abs/1512.03385
    Time conditioning is done with FiLM, see
    https://distill.pub/2018/feature-wise-transformations/
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()

        # FiLM generator is constructed if time embedding dimension is passed:
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None

        # Pass time embedding through FiLM generator if available:
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """
    Multi-head self-attention in its regular implementation, as used in the 
    transformer. Evaluates relevance between pixels. 
    By default, it has 4 heads with dimension of 32.
    For an excellent explanation, see
    https://jalammar.github.io/illustrated-transformer/,
    In this case, the words in a sequence correspond to pixels in an image.
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # After to_qkv: [b, dim_head*heads*3, h, w]
        # After chunk: 3 Tensors (q, k and v) of [b, dim_head*heads, h, w]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv
        )
        q = q * self.scale

        # q, k, and v have dim [b, heads, dim_heads, h*w]
        # Dot product: Sum over channels
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        # Now: [b, heads, h*w, h*w], one sum for each q-v pixel pair.
        # Max trick for numerical stability: Subtraction does not change
        # softmax output.
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


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

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y",
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """
    Group normalization layer with only one group, which effectively makes it a 
    layer normalization if I understand all this stuff correctly.
    Applied before attention layer.

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Unet(customModelClass):
    def __init__(
        self,
        init_channels,
        out_channels=None,
        channel_mults=(1, 2, 4, 8),
        image_channels=3,
        self_condition=False,
        resnet_norm_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = image_channels
        self.self_condition = self_condition
        image_channels = image_channels * (2 if self_condition else 1)

        self.init_channels = init_channels
        self.init_conv = nn.Conv2d(image_channels, self.init_channels,
                                    1, padding=0)

        dims = [self.init_channels, *map(lambda m: self.init_channels * m,
                                          channel_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_norm_groups)

        # time embeddings
        time_dim = self.init_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.init_channels),
            nn.Linear(self.init_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out,
                                    time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out,
                                    time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1)

                    ]
                )
            )

        self.out_dim = default(out_channels, image_channels)

        self.final_res_block = block_klass(self.init_channels * 2, 
                                           self.init_channels, 
                                           time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(self.init_channels, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

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
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

# Create a class representing a BigGAN residual block consistent with the other
# implementations of this file.
class BigGANBlock(TimestepBlock):
    def __init__(self,
                dim,
                dim_out,
                time_emb_dim,
                dropout,
                *,
                norm_groups=32,
                up=False,
                down=False,
                use_conv=False,):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(norm_groups, dim),
            nn.SiLU(),
            WeightStandardizedConv2d(dim, dim_out, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(dim, use_conv=False)
            self.x_upd = Upsample(dim, use_conv=False)
        elif down:
            self.h_upd = Downsample(dim)
            self.x_upd = Downsample(dim)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(norm_groups, dim_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            WeightStandardizedConv2d(dim_out, dim_out, 3, padding=1),
        )

        if dim == dim_out:
            self.res_conv = nn.Identity()
        elif use_conv:
            self.res_conv = WeightStandardizedConv2d(dim, dim_out, 3)
        else:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x, time_emb):
        if self.updown:
            in_pre, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_pre(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        time_emb = self.emb_layers(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale, shift = time_emb.chunk(2, dim=1)

        out_norm, out_post = self.out_layers[0], self.out_layers[1:]
        h = out_norm(h) * (scale + 1) + shift
        h = out_post(h)
        return h + self.res_conv(x)
    
    
class ImprovedUnet(customModelClass):
    def __init__(
        self,
        init_channels,
        learn_variance=False,
        out_channels=None,
        channel_mults=(1, 2, 4, 8),
        image_channels=1,
        self_condition=False,
        norm_groups=32,
        dropout=0,
        num_res_blocks=2,
        attention_levels=3,
        attention_heads=4,
        attention_head_channels=32,

    ):
        super().__init__()

        # determine dimensions
        self.input_channels = image_channels * (2 if self_condition else 1)
        self.self_condition = self_condition
        self.out_channels = default(out_channels, image_channels)
        if learn_variance:
            self.out_channels *= 2

        self.init_channels = init_channels

        # time embeddings
        time_dim = init_channels * 4

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(init_channels),
            nn.Linear(init_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.input_block = nn.ModuleList([
            # Initial convolution
            TimestepEmbedSequential(
                nn.Conv2d(self.input_channels, self.init_channels, 3, padding=1)
            )
        ])
        self.output_block = nn.ModuleList([])

        input_block_chans = [self.init_channels]
        ch = self.init_channels

        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [
                    BigGANBlock(ch, int(mult * init_channels),
                                      time_dim, dropout, 
                                      norm_groups=norm_groups)
                ]
                ch = int(mult * init_channels)
                if len(channel_mults) - level <= attention_levels:
                    layers.append(Residual(LinearAttention(
                        ch, heads=attention_heads,
                        dim_head=attention_head_channels
                    )))

                self.input_block.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mults) - 1:  # Downsample
                self.input_block.append(BigGANBlock(ch, ch, time_dim,
                                                dropout=dropout, down=True,
                                                norm_groups=norm_groups))
                input_block_chans.append(ch)
                
        self.middle_block = TimestepEmbedSequential(
            BigGANBlock(ch, ch, time_dim, dropout, norm_groups=norm_groups),
            Residual(LinearAttention(
                ch, heads=attention_heads,
                dim_head=attention_head_channels
            )),
            BigGANBlock(ch, ch, time_dim, dropout, norm_groups=norm_groups),
        )

        for level, mult in list(enumerate(channel_mults))[::-1]:
            # See:
            # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py#L396
            # l. 568
            for i in range(num_res_blocks+1):  # WHY +1???
                ich = input_block_chans.pop()
                layers = [
                    BigGANBlock(ch + ich, int(mult * init_channels),
                                time_dim, dropout)
                ]
                ch = int(mult * init_channels)
                if len(channel_mults) - level <= attention_levels:
                    layers.append(Residual(LinearAttention(
                        ch, heads=attention_heads,
                        dim_head=attention_head_channels
                )))

                if level and i == num_res_blocks:
                    layers.append(BigGANBlock(ch, ch, time_dim,
                                              dropout=dropout, up=True,
                                              norm_groups=norm_groups))

                self.output_block.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.out_channels, 3, padding=1),
        )

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        t = self.time_emb(time)
        h = []
        for i, module in enumerate(self.input_block):
            x = module(x, t)
            h.append(x)
        x = self.middle_block(x, t)
        for i, module in enumerate(self.output_block):
            x = torch.cat((x, h.pop()), dim=1)
            x = module(x, t)
        return self.out(x)


class EDMPrecond(customModelClass):
    def __init__(
        self,
        model,
        sigma_min = 0,
        sigma_max = torch.inf,
        sigma_data = 0.5,
        ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    @classmethod
    def from_config(cls, config):
        model = ImprovedUnet.from_config(config)
        return construct_from_config(cls, config, model=model)

    def forward(self, x, sigma):
        sigma = sigma.view([-1, 1, 1, 1])
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x), c_noise.flatten())
        D_x = c_skip * x + c_out * F_x
        return D_x



        

        