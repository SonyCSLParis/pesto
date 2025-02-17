r"""Cached convolutions. Original code from Antoine Caillon (former IRCAM).
See https://github.com/acids-ircam/cached_conv/tree/master"""
from typing import Tuple

import torch
import torch.nn as nn


class RefillPad1d(nn.Module):
    def __init__(self, padding: Tuple[int, int]):
        super(RefillPad1d, self).__init__()
        self.right_padding = padding[1]

    def forward(self, x):
        return torch.cat((x, x[..., -self.right_padding:]), dim=-1)


class CachedPadding1d(nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of
    the previous tensor.

    Compared to original implementation, we only consider mono signals and batch size 1.
    """

    def __init__(self, padding, max_batch_size: int = 1, crop=False):
        super().__init__()
        self.padding = padding
        self.max_batch_size = max_batch_size
        self.crop = crop

        self.init_cache()

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self):
        self.register_buffer("pad", torch.zeros(self.max_batch_size, 1, self.padding), persistent=False)

    def forward(self, x):
        bs = x.size(0)
        if self.padding:
            x = torch.cat((self.pad[:bs], x), -1)
            self.pad[:bs].copy_(x[..., -self.padding:])

        return x


class CachedConv1d(nn.Conv1d):
    """
    Implementation of a Conv1d operation with cached padding
    """
    def __init__(self, *args, **kwargs):
        padding = kwargs.get("padding", 0)
        max_batch_size = kwargs.pop("max_batch_size", 1)
        mirror = kwargs.pop("mirror", 0)
        mirror_fn = kwargs.pop("mirror_fn", "zeros")
        cumulative_delay = kwargs.pop("cumulative_delay", 0)

        kwargs["padding"] = 0

        super(CachedConv1d, self).__init__(*args, **kwargs)

        if isinstance(padding, int):
            r_pad = padding
        elif isinstance(padding, list) or isinstance(padding, tuple):
            r_pad = padding[1]
            padding = padding[0] + padding[1]
        else:
            raise TypeError("padding must be int or list or tuple")

        s = self.stride[0]
        cd = cumulative_delay

        stride_delay = (s - ((r_pad + cd) % s)) % s

        self.cumulative_delay = (r_pad + stride_delay + cd) // s

        self.cache = CachedPadding1d(padding, max_batch_size=max_batch_size)
        # self.downsampling_delay = CachedPadding1d(stride_delay, crop=True)

        if mirror == 0:
            mirroring_fn = nn.Identity
        elif mirror_fn == "reflection":
            mirroring_fn = nn.ReflectionPad1d
        elif mirror_fn == "zeros":
            mirroring_fn = nn.ZeroPad1d
        elif mirror_fn == "refill":
            mirroring_fn = RefillPad1d
        else:
            mirroring_fn = nn.Identity

        self.mirror = mirroring_fn((0, mirror))

    def forward(self, x):
        # x = self.downsampling_delay(x)  NOTE: not sure we actually need this thing
        x = self.cache(x)
        x = self.mirror(x)
        return super(CachedConv1d, self).forward(x)


class CachedConvTranspose1d(nn.ConvTranspose1d):
    """
    Implementation of a ConvTranspose1d operation with cached padding
    """

    def __init__(self, *args, **kwargs):
        cd = kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        stride = self.stride[0]
        self.initialized = 0
        self.cumulative_delay = self.padding[0] + cd * stride

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self, x):
        b, c, _ = x.shape
        self.register_buffer(
            "cache",
            torch.zeros(1, c, 2 * self.padding[0]).to(x))
        self.initialized += 1

    def forward(self, x):
        x = nn.functional.conv_transpose1d(
            x,
            self.weight,
            None,
            self.stride,
            0,
            self.output_padding,
            self.groups,
            self.dilation,
        )

        if not self.initialized:
            self.init_cache(x)

        padding = 2 * self.padding[0]

        x[..., :padding] += self.cache[:x.shape[0]]
        self.cache[:x.shape[0]].copy_(x[..., -padding:])

        x = x[..., :-padding]

        bias = self.bias
        if bias is not None:
            x = x + bias.unsqueeze(-1)
        return x


class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("cumulative_delay", 0)
        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0


class Conv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        self._pad = kwargs.get("padding", (0, 0))
        kwargs.pop("cumulative_delay", 0)
        kwargs["padding"] = 0

        super().__init__(*args, **kwargs)
        self.cumulative_delay = 0

    def forward(self, x):
        x = nn.functional.pad(x, self._pad)
        return nn.functional.conv1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
