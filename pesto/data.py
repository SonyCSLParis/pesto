import torch
import torch.nn as nn

from .cqt import CQT


class DataProcessor(nn.Module):
    r"""

    Args:
        step_size (float): step size between consecutive CQT frames (in SECONDS)
    """
    def __init__(self,
                 step_size: float,
                 bins_per_semitone: int = 3,
                 device: torch.device = torch.device("cpu"),
                 **cqt_kwargs):
        super(DataProcessor, self).__init__()
        self.bins_per_semitone = bins_per_semitone

        # CQT-related stuff
        self.cqt_kwargs = cqt_kwargs
        self.cqt_kwargs["bins_per_octave"] = 12 * bins_per_semitone
        self.cqt = None

        # log-magnitude
        self.eps = torch.finfo(torch.float32).eps

        # cropping
        self.lowest_bin = int(11 * self.bins_per_semitone / 2 + 0.5)
        self.highest_bin = self.lowest_bin + 88 * self.bins_per_semitone

        # handling different sampling rates
        self._sampling_rate = None
        self.step_size = step_size
        self.device = device

    def forward(self, x: torch.Tensor):
        r"""

        Args:
            x: audio waveform, any sampling rate, shape (num_samples)

        Returns:
            log-magnitude CQT, shape (
        """
        # compute CQT from input waveform, and invert dims for (batch_size, time_steps, freq_bins)
        complex_cqt = torch.view_as_complex(self.cqt(x)).transpose(1, 2)

        # reshape and crop borders to fit training input shape
        complex_cqt = complex_cqt[..., self.lowest_bin: self.highest_bin]

        # flatten eventual batch dimensions so that batched audios can be processed in parallel
        complex_cqt = complex_cqt.flatten(0, 1).unsqueeze(1)

        # convert to dB
        log_cqt = complex_cqt.abs().clamp_(min=self.eps).log10_().mul_(20)
        return log_cqt

    def _init_cqt_layer(self, sr: int, hop_length: int):
        self.cqt = CQT(sr=sr, hop_length=hop_length, **self.cqt_kwargs).to(self.device)

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sr: int):
        if sr == self._sampling_rate:
            return

        hop_length = int(self.step_size * sr + 0.5)
        self._init_cqt_layer(sr, hop_length)
        self._sampling_rate = sr
