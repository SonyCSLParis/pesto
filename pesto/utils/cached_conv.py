r"""Cached convolutions. Original code from Antoine Caillon (former IRCAM).
See https://github.com/acids-ircam/cached_conv/tree/master"""
import torch
import torch.nn as nn


class CachedPadding1d(nn.Module):
    """
    Cached Padding implementation, replace zero padding with the end of
    the previous tensor.

    Compared to original implementation, we only consider mono signals and batch size 1.
    """

    def __init__(self, padding, crop=False):
        super().__init__()
        self.padding = padding
        self.crop = crop

        self.init_cache()

    @torch.jit.unused
    @torch.no_grad()
    def init_cache(self):
        self.register_buffer("pad", torch.zeros(1, 1, self.padding), persistent=False)

    def forward(self, x):
        if self.padding:
            x = torch.cat((self.pad, x), -1)
            self.pad.copy_(x[..., -self.padding:])

        return x


class CachedConv1d(nn.Conv1d):
    """
    Implementation of a Conv1d operation with cached padding
    """
    def __init__(self, *args, **kwargs):
        padding = kwargs.get("padding", 0)
        mirror = kwargs.pop("mirror", 0)
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

        self.cache = CachedPadding1d(padding)
        self.downsampling_delay = CachedPadding1d(stride_delay, crop=True)
        self.mirror = nn.ReflectionPad1d((0, mirror)) if mirror > 0 else nn.Identity()

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
