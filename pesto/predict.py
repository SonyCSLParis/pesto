import os
import logging
from typing import Sequence

import torch
import torchaudio

import utils
from export import export


def predict(
        x: torch.Tensor,
        model,
        data_preprocessor=None,
        reduction: str = "argmax",
        sr: int = None,
        hop_length: int = None,
        chunk_length: int = None,
        convert_to_freq: bool = False
):
    # convert to mono
    assert x.ndim == 2, f"Audio file should have two dimensions, but found shape {x.size()}"
    x = x.mean(dim=0)

    if data_preprocessor is None:
        data_preprocessor = utils.load_dataprocessor(device=x.device)

    if isinstance(model, str):
        model = utils.load_model(model, device=x.device)
        assert sr and hop_length and chunk_length, \
            "You must specify the sampling rate and hop length when calling directly `pesto.predict`"
        data_preprocessor.init_cqt_layer(sr=sr, hop_length=hop_length, device=x.device)

    # apply model
    cqt = data_preprocessor(x)
    activations = torch.cat([
        model(chunk) for chunk in cqt.split(chunk_length)
    ])

    # shift activations as it should
    activations = activations.roll(model.abs_shift.cpu().item(), dims=1)

    # convert model predictions to pitch values
    if reduction is None:
        pitch = None
    else:
        pitch = utils.reduce_activation(activations, reduction=reduction)
        if convert_to_freq:
            pitch = 440 * 2 ** ((pitch - 69) / 12)

    # for now, confidence is computed very naively just based on volume
    confidence = cqt.squeeze(1).max(dim=1).values
    confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())

    timesteps = torch.arange(len(confidence)) * data_preprocessor.step_size

    return timesteps, pitch, confidence, activations


@torch.inference_mode()
def predict_from_file(
        audio_file: str,
        model: torch.nn.Module,
        data_preprocessor: torch.nn.Module,
        device=torch.device("cpu"),
        step_size: float = 10.,
        chunk_size: float = 30.,
        reduction: str = "argmax",
        convert_to_freq: bool = False
):
    # load audio file (maybe this thing can be faster)
    x, sr = torchaudio.load(audio_file)
    x = x.to(device)

    # if the sampling rate has changed, recompute the CQT kernels
    if sr != data_preprocessor.sr:
        hop_length = int(step_size * sr / 1000 + 0.5)
        data_preprocessor.init_cqt_layer(sr, hop_length, device)

    # compute the predictions
    predictions = predict(
        x,
        model,
        data_preprocessor,
        reduction=reduction,
        chunk_length=int(chunk_size * sr),
        convert_to_freq=convert_to_freq
    )
    return predictions, data_preprocessor


def predict_from_files(
        audio_files: Sequence[str],
        model_name: str,
        output: str | None = None,
        step_size: float = 10.,
        reduction: str = "argmax",
        export_format: Sequence[str] = ("csv",),
        no_convert_to_freq: bool = False,
        no_overwrite: bool = False,
        chunk_size: float = 30,
        gpu: int = -1
):
    r"""

    Args:
        audio_files:
        model_name:
        output:
        step_size: hop length in milliseconds
        reduction:
        export_format:
        no_convert_to_freq: whether convert output values to Hz or keep fractional MIDI pitches
        no_overwrite: skip pitch tracking if output file already exists
        chunk_size:
        gpu:

    Returns:

    """
    device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")

    # define data preprocessing
    data_preprocessor = utils.load_dataprocessor(device=device)

    # define model
    model = utils.load_model(model_name, device=device)
    predictions = None

    n_files = len(audio_files)
    for i, file in enumerate(audio_files):
        print(f"[{i+1}/{n_files}]", file, end='\r')

        output_file = file.rsplit('.', 1)[0] + ".f0"
        if output is not None:
            os.makedirs(output, exist_ok=True)
            output_file = os.path.join(output, os.path.basename(output_file))

        if no_overwrite:
            if all([os.path.exists(output_file + '.' + fmt) for fmt in export_format]):
                continue

        predictions, data_processor = predict_from_file(
            file,
            model,
            data_preprocessor,
            device=torch.device("cpu"),
            reduction=reduction,
            step_size=step_size,
            chunk_size=chunk_size,
            convert_to_freq=not no_convert_to_freq
        )

        for fmt in export_format:
            export(fmt, output_file, *predictions)
        logging.info(f"Pitch predictions saved in {output_file}.{export_format}.")

    return predictions
