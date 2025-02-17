import torch


def mid_to_hz(pitch: int):
    return 440 * 2 ** ((pitch - 69) / 12)


def generate_synth_data(pitch: int, num_harmonics: int = 5, duration=2, sr=16000):
    f0 = mid_to_hz(pitch)
    t = torch.arange(0, duration, 1/sr)
    harmonics = torch.stack([
        torch.cos(2 * torch.pi * k * f0 * t + torch.rand(()))
        for k in range(1, num_harmonics+1)
    ], dim=1)
    # volume = torch.randn(()) * torch.arange(num_harmonics).neg().div(0.5).exp()
    volume = 0.5 * torch.rand(num_harmonics)
    volume[0] = 1
    volume *= torch.randn(()).clip(min=0.1)
    audio = torch.sum(volume * harmonics, dim=1)
    return audio