import torch
import torch.nn as nn


class CropCQT(nn.Module):
    def __init__(self, min_steps: int, max_steps: int):
        super(CropCQT, self).__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps

        # lower bin
        self.lower_bin = self.max_steps

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        # WARNING: didn't check that it works, it may be dangerous
        return spectrograms[..., self.max_steps: self.min_steps]

        # old implementation
        batch_size, _, input_height = spectrograms.size()

        output_height = input_height - self.max_steps + self.min_steps
        assert output_height > 0, \
            f"With input height {input_height:d} and output height {output_height:d}, impossible " \
            f"to have a range of {self.max_steps - self.min_steps:d} bins."

        return spectrograms[..., self.lower_bin: self.lower_bin + output_height]
