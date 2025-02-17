from typing import Optional

import torch
import torch.nn as nn

from .utils import HarmonicCQT


class ToLogMagnitude(nn.Module):
    def __init__(self):
        super(ToLogMagnitude, self).__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        x = x.abs()
        x.clamp_(min=self.eps).log10_().mul_(20)
        return x



class Preprocessor(nn.Module):
    r"""

    Args:
        hop_size (float): step size between consecutive CQT frames (in milliseconds)
    """
    def __init__(self,
                 hop_size: float,
                 sampling_rate: Optional[int] = None,
                 **hcqt_kwargs):
        super(Preprocessor, self).__init__()

        # HCQT
        self.hcqt_sr = None
        self.hcqt_kernels = None
        self.hop_size = hop_size

        self.hcqt_kwargs = hcqt_kwargs

        # log-magnitude
        self.to_log = ToLogMagnitude()

        # register a dummy tensor to get implicit access to the module's device
        self.register_buffer("_device", torch.zeros(()), persistent=False)

        # if the sampling rate is provided, instantiate the CQT kernels
        if sampling_rate is not None:
            self.hcqt_sr = sampling_rate
            self._reset_hcqt_kernels()

    def forward(self, x: torch.Tensor, sr: Optional[int] = None) -> torch.Tensor:
        r"""

        Args:
            x (torch.Tensor): audio waveform or batch of audio waveforms, any sampling rate,
                shape (batch_size?, num_samples)
            sr (int, optional): sampling rate

        Returns:
            torch.Tensor: log-magnitude CQT of batch of CQTs,
                shape (batch_size, num_timesteps, num_harmonics, num_freqs)
        """
        # compute CQT from input waveform, and invert dims for (time_steps, num_harmonics, freq_bins)
        # in other words, time becomes the batch dimension, enabling efficient processing for long audios.
        complex_cqt = torch.view_as_complex(self.hcqt(x, sr=sr)).permute(0, 3, 1, 2)

        # convert to dB
        return self.to_log(complex_cqt)

    def hcqt(self, audio: torch.Tensor, sr: Optional[int] = None) -> torch.Tensor:
        r"""Compute the Harmonic CQT of the input audio after eventually recreating the kernels
        (in case the sampling rate has changed).

        Args:
            audio (torch.Tensor): mono audio waveform, shape (batch_size, num_samples)
            sr (int): sampling rate of the audio waveform.
                If not specified, we assume it is the same as the previous processed audio waveform.

        Returns:
            torch.Tensor: Complex Harmonic CQT (HCQT) of the input,
                shape (batch_size, num_harmonics, num_freqs, num_timesteps, 2)
        """
        # compute HCQT kernels if it does not exist or if the sampling rate has changed
        if sr is not None and sr != self.hcqt_sr:
            self.hcqt_sr = sr
            self._reset_hcqt_kernels()

        return self.hcqt_kernels(audio)

    def _reset_hcqt_kernels(self) -> None:
        hop_length = int(self.hop_size * self.hcqt_sr / 1000 + 0.5)
        self.hcqt_kernels = HarmonicCQT(sr=self.hcqt_sr,
                                        hop_length=hop_length,
                                        **self.hcqt_kwargs).to(self._device.device)
