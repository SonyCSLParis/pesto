r"""The implementation of the CQT comes from the nnAudio repository: https://github.com/KinWaiCheuk/nnAudio
Due to conflicts between some versions of NumPy and nnAudio, we use the implementation as is instead of adding nnAudio
to the requirements of this project. Compared to the original implementation, some minor modifications have been done
in the code, however the behaviour remains the same.
"""
from typing import Optional

import pesto.utils.cached_conv as cc
import numpy as np
from scipy.signal import get_window

import torch
import torch.nn as nn
import torch.nn.functional as F


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    return x


def nextpow2(A):
    """A helper function to calculate the next nearest number to the power of 2.

    Parameters
    ----------
    A : float
        A float number that is going to be rounded up to the nearest power of 2

    Returns
    -------
    int
        The nearest power of 2 to the input number ``A``

    Examples
    --------

    >>> nextpow2(6)
    3
    """

    return int(np.ceil(np.log2(A)))


def get_window_dispatch(window, N, fftbins=True):
    if isinstance(window, str):
        return get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning(
            "You are using Kaiser window with beta factor "
            + str(window)
            + ". Correct behaviour not checked."
        )
    else:
        raise Exception(
            "The function get_window from scipy only supports strings, tuples and floats."
        )


def create_cqt_kernels(
        Q,
        fs,
        fmin,
        n_bins=84,
        bins_per_octave=12,
        norm=1,
        window="hann",
        fmax=None,
        topbin_check=True,
        gamma=0,
        pad_fft=True
):
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** nextpow2(np.ceil(Q * fs / fmin))
    # minWin = 2**nextpow2(np.ceil(Q * fs / fmax))

    if (fmax != None) and (n_bins == None):
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))

    elif (fmax == None) and (n_bins != None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(
            bins_per_octave * np.log2(fmax / fmin)
        )  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.double(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check == True:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, \
                          please reduce the n_bins".format(
                np.max(freqs)
            )
        )

    alpha = 2.0 ** (1.0 / bins_per_octave) - 1.0
    lengths = np.ceil(Q * fs / (freqs + gamma / alpha))

    # get max window length depending on gamma value
    max_len = int(max(lengths))
    fftLen = int(2 ** (np.ceil(np.log2(max_len))))

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)
    specKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    for k in range(0, int(n_bins)):
        freq = freqs[k]
        l = lengths[k]

        # Centering the kernels
        if l % 2 == 1:  # pad more zeros on RHS
            start = int(np.ceil(fftLen / 2.0 - l / 2.0)) - 1
        else:
            start = int(np.ceil(fftLen / 2.0 - l / 2.0))

        window_dispatch = get_window_dispatch(window, int(l), fftbins=True)
        sig = window_dispatch * np.exp(np.r_[-l // 2: l // 2] * 1j * 2 * np.pi * freq / fs) / l

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start: start + int(l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start: start + int(l)] = sig
        # specKernel[k, :] = fft(tempKernel[k])

    # return specKernel[:,:fftLen//2+1], fftLen, torch.tensor(lenghts).float()
    return tempKernel, fftLen, torch.tensor(lengths).float(), freqs


class BaseCQT(nn.Module):
    def __init__(
            self,
            sr=22050,
            hop_length=512,
            fmin=32.70,
            fmax=None,
            n_bins=84,
            bins_per_octave=12,
            gamma=0,
            filter_scale=1,
            norm=1,
            window="hann",
            center: bool = True,
            trainable=False,
            output_format="Magnitude"
    ):

        super(BaseCQT, self).__init__()

        self.trainable = trainable
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.center = center
        self.output_format = output_format

        # creating kernels for CQT
        Q = float(filter_scale) / (2 ** (1 / bins_per_octave) - 1)

        cqt_kernels, self.kernel_width, lengths, freqs = create_cqt_kernels(
            Q, sr, fmin, n_bins, bins_per_octave, norm, window, fmax, gamma=gamma
        )

        self.register_buffer("sqrt_lengths", lengths.sqrt_().unsqueeze_(-1))
        self.frequencies = freqs

        self.cqt_kernels = torch.from_numpy(cqt_kernels).unsqueeze(1)

    @torch.no_grad()
    def init_weights(self):
        # initialize convolution layers
        self.conv.weight.copy_(torch.cat((self.cqt_kernels.real, -self.cqt_kernels.imag), dim=0))
        self.conv.weight.requires_grad = self.trainable

    def forward(self, x, output_format=None, normalization_type="librosa"):
        r"""Computes the Constant-Q Transform

        Args:
            x (torch.Tensor): input audio waveform, shape (batch_size?, num_samples)
            output_format (str, optional): "Magnitude" or "Complex"
            normalization_type (str, optional): "librosa" or "convolutional"

        Returns:
            torch.Tensor: CQT, shape (batch_size, num_freqs, num_timesteps, 2?)
        """
        output_format = output_format or self.output_format

        x = broadcast_dim(x)

        # CQT
        cqt = self.conv(x).view(x.size(0), 2, self.n_bins, -1)

        if normalization_type == "librosa":
            cqt *= self.sqrt_lengths
        elif normalization_type == "convolutional":
            pass
        elif normalization_type == "wrap":
            cqt *= 2
        else:
            raise ValueError(
                "The normalization_type %r is not part of our current options."
                % normalization_type
            )

        if output_format == "Magnitude":
            margin = 1e-8 if self.trainable else 0
            return cqt.pow(2).sum(-3).add(margin).sqrt()

        if output_format == "Complex":
            return cqt.permute(0, 2, 3, 1)  # shape: (batch_size, n_bins, n_timesteps, 2)

        cqt_real, cqt_imag = cqt.split(self.n_bins, dim=-2)
        if output_format == "Phase":
            phase_real = torch.cos(torch.atan2(cqt_imag, cqt_real))
            phase_imag = torch.sin(torch.atan2(cqt_imag, cqt_real))
            return torch.stack((phase_real, phase_imag), -1)

        raise ValueError(f"Invalid output format: {output_format}.")


class RegularCQT(BaseCQT):
    def __init__(self, *args, pad_mode="reflect", **kwargs):
        super().__init__(*args, **kwargs)

        padding = self.kernel_width // 2 if self.center else 0

        self.conv = nn.Conv1d(1,
                              2 * self.n_bins,  # we handle real and imaginary part in parallel
                              kernel_size=self.kernel_width,
                              stride=self.hop_length,
                              padding=padding,
                              padding_mode=pad_mode,
                              bias=False)

        self.init_weights()


class StreamingCQT(BaseCQT):
    def __init__(self,
                 *args,
                 mirror: float = 0.,
                 max_batch_size: int = 1,
                 **kwargs):
        super(StreamingCQT, self).__init__(*args, **kwargs)

        if self.center:
            mirrored_samples = int(mirror * (self.kernel_width - self.hop_length) / 2)
            padding = self.kernel_width - self.hop_length - mirrored_samples
        else:  # no padding
            mirrored_samples = 0
            padding = 0

        self.conv = cc.CachedConv1d(1,
                                    2 * self.n_bins,
                                    kernel_size=self.kernel_width,
                                    stride=self.hop_length,
                                    padding=padding,
                                    mirror=mirrored_samples,
                                    max_batch_size=max_batch_size,
                                    bias=False)

        self.init_weights()


class CQT:
    regular_only_kwargs = ["pad_mode"]
    streaming_only_kwargs = ["mirror", "max_batch_size"]

    def __new__(cls, *args, **kwargs):
        streaming = kwargs.pop("streaming", False)
        if streaming:
            for kwarg in cls.regular_only_kwargs:
                kwargs.pop(kwarg, None)
            return StreamingCQT(*args, **kwargs)

        for kwarg in cls.streaming_only_kwargs:
            kwargs.pop(kwarg, None)
        return RegularCQT(*args, **kwargs)


class HarmonicCQT(nn.Module):
    r"""Harmonic CQT layer, as described in Bittner et al. (20??)"""
    def __init__(
            self,
            harmonics,
            sr: int = 22050,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: Optional[float] = None,
            bins_per_semitone: int = 1,
            n_bins: int = 84,
            center_bins: bool = True,
            gamma: int = 0,
            center: bool = True,
            streaming: bool = False,
            mirror: float = 0.,
            max_batch_size: int = 1
    ):
        super(HarmonicCQT, self).__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt_kernels = nn.ModuleList([
            CQT(sr=sr, hop_length=hop_length, fmin=h * fmin, fmax=fmax, n_bins=n_bins,
                bins_per_octave=12*bins_per_semitone, gamma=gamma, center=center,
                streaming=streaming, mirror=mirror, max_batch_size=max_batch_size,
                output_format="Complex")
            for h in harmonics
        ])

    def forward(self, audio_waveforms: torch.Tensor):
        r"""Converts a batch of waveforms into a batch of HCQTs.

        Args:
            audio_waveforms (torch.Tensor): Batch of waveforms, shape (batch_size, num_samples)

        Returns:
            Harmonic CQT, shape (batch_size, num_harmonics, num_freqs, num_timesteps, 2)
        """
        return torch.stack([cqt(audio_waveforms) for cqt in self.cqt_kernels], dim=1)
