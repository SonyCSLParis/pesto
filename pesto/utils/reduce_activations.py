import torch


def reduce_activations(activations: torch.Tensor, reduction: str = "alwa") -> torch.Tensor:
    r"""

    Args:
        activations: tensor of probability activations, shape (*, num_bins)
        reduction (str): reduction method to compute pitch out of activations,
            choose between "argmax", "mean", "alwa".

    Returns:
        torch.Tensor: pitches as fractions of MIDI semitones, shape (*)
    """
    device = activations.device
    num_bins: int = activations.size(-1)

    if torch.is_tensor(num_bins):  # JIT shit
        num_bins = num_bins.item()

    bps, r = divmod(num_bins, 128)
    assert r == 0, f"Activations should have output size 128*bins_per_semitone, got {num_bins}."

    if reduction == "argmax":
        pred = activations.argmax(dim=-1)
        return pred.float() / bps

    all_pitches = torch.arange(num_bins, dtype=torch.float, device=device).div_(bps)
    if reduction == "mean":
        return torch.matmul(activations, all_pitches)

    if reduction == "alwa":  # argmax-local weighted averaging, see https://github.com/marl/crepe
        center_bin = activations.argmax(dim=-1, keepdim=True)
        window = torch.arange(1, 2 * bps, device=device) - bps  # [-bps+1, -bps+2, ..., bps-2, bps-1]
        indices = (center_bin + window).clip_(min=0, max=num_bins - 1)
        cropped_activations = activations.gather(-1, indices)
        cropped_pitches = all_pitches.unsqueeze(0).expand_as(activations).gather(-1, indices)
        return (cropped_activations * cropped_pitches).sum(dim=-1) / cropped_activations.sum(dim=-1)

    raise ValueError
