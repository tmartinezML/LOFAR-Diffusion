"""
Created following the tutorial on 
https://huggingface.co/blog/annotated-diffusion

"""
import math
from inspect import isfunction
from functools import partial

import torch
from torch import Tensor, einsum
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


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


def Upsample(dim, dim_out=None):
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
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
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


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
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

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

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
