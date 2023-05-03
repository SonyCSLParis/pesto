import matplotlib.pyplot as plt
import numpy as np

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
        raise ValueError


def export_csv(output_file, timesteps, pitch, confidence):
    data = torch.stack((timesteps, pitch, confidence), dim=1).cpu().numpy()
    header = "time,frequency,confidence"

    np.savetxt(output_file, data, delimiter=',', fmt='%.3f', header=header, comments="")


def export_npy(output_file, timesteps, pitch, confidence, activations):
    np.savez(output_file, timesteps=timesteps, pitch=pitch, confidence=confidence, activations=activations)


def export_png(output_file, timesteps, pitch, confidence, activations, lims=(21, 109)):
    bps = activations.size(1) // 128
    activations = activations[:, bps*lims[0]: bps*lims[1]]
    activations = activations * confidence.unsqueeze(1)
    plt.imshow(activations.t().cpu().numpy(),
               aspect='auto', origin='lower', cmap='inferno',
               extent=(timesteps[0], timesteps[-1]) + lims)

    mask = confidence > 0.9
    plt.scatter(timesteps[mask], pitch[mask], c='w', s=0.1)
    plt.title(output_file.rsplit('.', 2)[0])
    plt.tight_layout()
    plt.savefig(output_file)
