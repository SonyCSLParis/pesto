import logging
import pathlib

import numpy as np
try:
    CREPE_NOTES_AVAILABLE = True
    from crepe_notes.crepe_notes import process as process_crepe_notes
except (ImportError, ModuleNotFoundError):
    CREPE_NOTES_AVAILABLE = False

try:
    MATPLOTLIB_AVAILABLE = True
    import matplotlib.pyplot as plt
except (IndexError, ModuleNotFoundError):
    MATPLOTLIB_AVAILABLE = False


log = logging.getLogger(__name__)


def export(fmt, output_file, timesteps, pitch, confidence, activations):
    output_file = output_file + '.' + fmt
    if fmt == "csv":
        export_csv(output_file, timesteps, pitch, confidence)

    elif fmt == "mid":
        export_mid(output_file, timesteps, pitch, confidence)

    elif fmt == "npz":
        export_npz(output_file, timesteps, pitch, confidence, activations)

    elif fmt == "png":
        export_png(output_file, timesteps, confidence, activations)

    else:
        raise ValueError(f"Invalid export type detected, choose either `csv`, `npz` or `png`. Got {fmt}.")


def export_csv(output_file, timesteps, pitch, confidence):
    data = np.stack((timesteps, pitch, confidence), axis=1)
    header = "time,frequency,confidence"

    np.savetxt(output_file, data, delimiter=',', fmt='%.3f', header=header, comments="")


def export_mid(output_file, timesteps, pitch, confidence):
    if not CREPE_NOTES_AVAILABLE:
        log.error("Exporting in MIDI format requires `crepe_notes` to be installed. "
                      "Please check https://github.com/xavriley/crepe_notes for more information.")
        return

    stem, fmt, ext = output_file.split(".")
    assert fmt == "f0", \
        "To be compatible with the CREPE Notes API, you have to convert PESTO predictions to frequencies."

    # CREPE Notes requires the input audio file as a `pathlib.Path` object
    # In order to avoid messing up the whole API we just retrieve it from the output file
    # It may break if you set a custom output_dir or non ".wav" files, maybe it'll be fixed later...
    audio_path = pathlib.Path(stem + ".wav")

    # We disable splitting because madmom has shitty deprecated dependencies making it incompatible with Python 3.10
    process_crepe_notes(pitch,
                        confidence,
                        audio_path,
                        sensitivity=0.005,
                        min_duration=0.05,
                        disable_splitting=True,
                        use_cwd=False,
                        default_sample_rate=44100)


def export_npz(output_file, timesteps, pitch, confidence, activations):
    np.savez(output_file, timesteps=timesteps, pitch=pitch, confidence=confidence, activations=activations)


def export_png(output_file: str, timesteps, confidence, activations, lims=(21, 109)) -> None:
    if not MATPLOTLIB_AVAILABLE:
        log.error("Exporting in PNG format requires Matplotlib to be installed. Skipping...")
        return

    bps = activations.shape[1] // 128
    activations = activations[:, bps*lims[0]: bps*lims[1]]
    activations = activations * confidence[:, None]
    plt.imshow(activations.T,
               aspect='auto', origin='lower', cmap='inferno',
               extent=(timesteps[0] / 1000., timesteps[-1] / 1000.) + lims)

    plt.xlabel("Time (s)")
    plt.title(output_file.rsplit('.', 2)[0])
    plt.tight_layout()
    plt.savefig(output_file)
