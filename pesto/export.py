import warnings

import numpy as np
try:
    MATPLOTLIB_AVAILABLE = True
    import matplotlib.pyplot as plt
except (IndexError, ModuleNotFoundError):
    MATPLOTLIB_AVAILABLE = False

import torch


def export(fmt, output_file, timesteps, pitch, confidence, activations):
    output_file = output_file + '.' + fmt
    if fmt == "csv":
        export_csv(output_file, timesteps, pitch, confidence)

    elif fmt == "npz":
        export_npy(output_file, timesteps, pitch, confidence, activations)

    elif fmt == "png":
        export_png(output_file, timesteps, pitch, confidence, activations)

    else:
        raise ValueError(f"Invalid export type detected, choose either `csv`, `npz` or `png`. Got {fmt}.")


def export_csv(output_file, timesteps, pitch, confidence):
    data = torch.stack((timesteps, pitch, confidence), dim=1).cpu().numpy()
    header = "time,frequency,confidence"

    np.savetxt(output_file, data, delimiter=',', fmt='%.3f', header=header, comments="")


def export_npy(output_file, timesteps, pitch, confidence, activations):
    np.savez(output_file, timesteps=timesteps, pitch=pitch, confidence=confidence, activations=activations)


def export_png(output_file: str, timesteps, confidence, activations, lims=(21, 109)) -> None:
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Exporting in PNG format requires Matplotlib to be installed. Skipping...")
        return
    print(activations.shape)
    bps = activations.size(1) // 128
    activations = activations[:, bps*lims[0]: bps*lims[1]]
    activations = activations * confidence.unsqueeze(1)
    plt.imshow(activations.t().cpu().numpy(),
               aspect='auto', origin='lower', cmap='inferno',
               extent=(timesteps[0], timesteps[-1]) + lims)

    plt.title(output_file.rsplit('.', 2)[0])
    plt.tight_layout()
    plt.savefig(output_file)
