import os
import time
import warnings
from typing import Sequence

import torch
import torchaudio

from pesto.utils import load_model, load_dataprocessor, reduce_activation, format_time
from pesto.export import export


def predict(
        x: torch.Tensor,
        model,
        data_preprocessor=None,
        reduction: str = "argmax",
        sr: int = None,
        hop_length: int = None,
        convert_to_freq: bool = False
):
    # convert to mono
    assert x.ndim == 2, f"Audio file should have two dimensions, but found shape {x.size()}"
    x = x.mean(dim=0)

    if data_preprocessor is None:
        data_preprocessor = load_dataprocessor(device=x.device)

    if isinstance(model, str):
        model = load_model(model, device=x.device)
        assert sr and hop_length, \
            "You must specify the sampling rate and hop length when calling directly `pesto.predict`"
        data_preprocessor.init_cqt_layer(sr=sr, hop_length=hop_length, device=x.device)

    # apply model
    cqt = data_preprocessor(x)
    activations = model(cqt)

    # shift activations as it should
    activations = activations.roll(model.abs_shift.cpu().item(), dims=1)

    # convert model predictions to pitch values
    pitch = reduce_activation(activations, reduction=reduction)
    if convert_to_freq:
        pitch = 440 * 2 ** ((pitch - 69) / 12)

    # for now, confidence is computed very naively just based on volume
    confidence = cqt.squeeze(1).max(dim=1).values
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())

    timesteps = torch.arange(len(pitch), device=x.device) * data_preprocessor.step_size

    return timesteps, pitch, confidence, activations


@torch.inference_mode()
def predict_from_files(
        audio_files: Sequence[str],
        model_name: str,
        output: str | None = None,
        step_size: float = 10.,
        reduction: str = "argmax",
        export_format: Sequence[str] = ("csv",),
        no_convert_to_freq: bool = False,
        gpu: int = -1
):
    r"""

    Args:
        audio_files: audio files to process
        model_name: name of the model. Currently only `mir-1k` is supported.
        output:
        step_size: hop length in milliseconds
        reduction:
        export_format:
        no_convert_to_freq: whether convert output values to Hz or keep fractional MIDI pitches
        gpu: index of GPU to use (-1 for CPU)

    Returns:
        Pitch predictions, see `predict` for more details.
    """
    t0 = time.time()

    if gpu >= 0 and not torch.cuda.is_available():
        warnings.warn("You're trying to use the GPU but no GPU has been found. Using CPU instead...")
        gpu = -1
    device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")

    # define data preprocessing
    data_preprocessor = load_dataprocessor(device=device)
    current_sr = None

    # define model
    model = load_model(model_name, device=device)
    predictions = None

    n_files = len(audio_files)
    len_last_info = 0
    for i, file in enumerate(audio_files):
        msg = f"[{i+1}/{n_files}] {file}"
        spaces = min(len_last_info - len(msg), 0) * ' '
        len_last_info = len(msg)
        print(msg + spaces, end='\r')

        # load audio file
        try:
            x, sr = torchaudio.load(file)
        except Exception as e:
            print(e, f"Skipping {file}...")
            continue

        x = x.to(device)

        # if the sampling rate has changed, recompute the CQT kernels
        if sr != current_sr:
            hop_length = int(step_size * sr / 1000 + 0.5)
            data_preprocessor.init_cqt_layer(sr, hop_length, device)
            current_sr = sr

        # compute the predictions
        predictions = predict(
            x,
            model,
            data_preprocessor,
            reduction=reduction,
            convert_to_freq=not no_convert_to_freq
        )

        output_file = file.rsplit('.', 1)[0] + ".f0"
        if output is not None:
            os.makedirs(output, exist_ok=True)
            output_file = os.path.join(output, os.path.basename(output_file))

        predictions = [p.cpu().numpy() for p in predictions]
        for fmt in export_format:
            export(fmt, output_file, *predictions)

    t1 = time.time()

    print(f"Successfully predicted pitch for {len(audio_files)} files in {format_time(t1-t0)}.")

    return predictions
