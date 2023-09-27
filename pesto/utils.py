import os

import torch

from pesto.config import model_args, cqt_args, bins_per_semitone
from pesto.data import DataProcessor
from pesto.model import PESTOEncoder


def load_dataprocessor(device: torch.device | None = None):
    return DataProcessor(**cqt_args).to(device)


def load_model(model_name: str, device: torch.device | None = None) -> PESTOEncoder:
    model = PESTOEncoder(**model_args).to(device)
    model.eval()

    model_path = os.path.join(os.path.dirname(__file__), "weights", model_name + ".pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def reduce_activation(activations: torch.Tensor, reduction: str) -> torch.Tensor:
    r"""Computes the pitch predictions from the activation outputs of the encoder.
    Pitch predictions are returned in semitones, NOT in frequencies.

    Args:
        activations: tensor of probability activations, shape (num_frames, num_bins)
        reduction:

    Returns:
        torch.Tensor: pitch predictions, shape (num_frames,)
    """
    bps = bins_per_semitone
    if reduction == "argmax":
        pred = activations.argmax(dim=1)
        return pred.float() / bps

    all_pitches = (torch.arange(activations.size(1), dtype=torch.float, device=activations.device)) / bps
    if reduction == "mean":
        return torch.mm(activations, all_pitches)

    if reduction == "alwa":  # argmax-local weighted averaging, see https://github.dev/marl/crepe
        center_bin = activations.argmax(dim=1, keepdim=True)
        window = torch.arange(-bps+1, bps, device=activations.device)
        indices = window + center_bin
        cropped_activations = activations.gather(1, indices)
        cropped_pitches = all_pitches.unsqueeze(0).expand_as(activations).gather(1, indices)
        return (cropped_activations * cropped_pitches).sum(dim=1) / cropped_activations.sum(dim=1)

    raise ValueError


def format_time(t: float):
    if t < 60:
        return f"{t:.3f}s"
    m, s = divmod(t, 60)
    return f"{m:d}min {s:.3f}s"
