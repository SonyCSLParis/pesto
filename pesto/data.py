import torch
import torch.nn as nn

from nnAudio.features.cqt import CQT


class DataProcessor(nn.Module):
    def __init__(self, bins_per_semitone: int = 3, **cqt_kwargs):
        super(DataProcessor, self).__init__()
        self.bins_per_semitone = bins_per_semitone

        # CQT-related stuff
        self.cqt_kwargs = cqt_kwargs
        self.cqt_kwargs["bins_per_octave"] = 12 * bins_per_semitone
        self.cqt = None
        self.step_size = None

        # log-magnitude
        self.eps = torch.finfo(torch.float32).eps

        # cropping
        self.lowest_bin = int(11 * self.bins_per_semitone / 2 + 0.5)
        self.highest_bin = self.lowest_bin + 88 * self.bins_per_semitone

    def forward(self, x: torch.Tensor):
        r"""

        Args:
            x: audio waveform, any sampling rate, shape (num_samples)

        Returns:
            log-magnitude CQT, shape (
        """
        # compute CQT from input waveform
        complex_cqt = torch.view_as_complex(self.cqt(x)).permute(2, 0, 1)

        # reshape and crop borders to fit training input shape
        complex_cqt = complex_cqt[..., self.lowest_bin: self.highest_bin]

        log_cqt = complex_cqt.abs().clamp_(min=self.eps).log10_().mul_(20)
        return log_cqt

    def init_cqt_layer(self, sr: int, hop_length: int, device):
        self.step_size = hop_length / sr
        self.cqt = CQT(sr=sr, hop_length=hop_length, **self.cqt_kwargs).to(device)
