import os
import warnings
from typing import Optional, Sequence, Union

import torch
import torchaudio
from tqdm import tqdm

from .utils import load_model, load_dataprocessor, reduce_activation
from .export import export


@torch.inference_mode()
def predict(
        x: torch.Tensor,
        sr: Optional[int] = None,
        model: Union[torch.nn.Module, str] = "mir-1k",
        data_preprocessor=None,
        step_size: Optional[float] = None,
        reduction: str = "argmax",
        convert_to_freq: bool = False
):
    r"""Main prediction function.

    Args:
        x (torch.Tensor): input audio tensor, shape (num_channels, num_samples)
        sr (int, optional): sampling rate. If not specified, uses the current sampling rate of the model.
        model: PESTO model. If a string is passed, it will load the model with the corresponding name.
            Otherwise, the actual nn.Module will be used for doing predictions.
        data_preprocessor: Module handling the data processing pipeline (waveform to CQT, cropping, etc.)
        step_size (float, optional): step size between each CQT frame in milliseconds.
            If the data_preprocessor is passed, its value will be used instead.
        reduction (str):
        convert_to_freq (bool): whether predictions should be converted to frequencies or not.
    """
    # convert to mono
    assert x.ndim == 2, f"Audio file should have two dimensions, but found shape {x.size()}"
    x = x.mean(dim=0)

    if data_preprocessor is None:
        assert step_size is not None, \
            "If you don't use a predefined data preprocessor, you must at least indicate a step size."
        data_preprocessor = load_dataprocessor(step_size=step_size, device=x.device)

    # If the sampling rate has changed, change the sampling rate accordingly
    # It will automatically recompute the CQT kernels if needed
    data_preprocessor.sampling_rate = sr

    if isinstance(model, str):
        model = load_model(model, device=x.device)

    # apply model
    cqt = data_preprocessor(x)
    activations = model(cqt)

    # shift activations as it should (PESTO predicts pitches up to an additive constant)
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


def predict_from_files(
        audio_files: Sequence[str],
        model_name: str = "mir-1k",
        output: Optional[str] = None,
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
    if gpu >= 0 and not torch.cuda.is_available():
        warnings.warn("You're trying to use the GPU but no GPU has been found. Using CPU instead...")
        gpu = -1
    device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")

    # define data preprocessing
    data_preprocessor = load_dataprocessor(step_size / 1000., device=device)

    # define model
    model = load_model(model_name, device=device)
    predictions = None

    pbar = tqdm(audio_files)
    for file in pbar:
        pbar.set_description(file)

        # load audio file
        try:
            x, sr = torchaudio.load(file)
        except Exception as e:
            print(e, f"Skipping {file}...")
            continue

        x = x.to(device)

        # compute the predictions
        predictions = predict(
            x,
            sr,
            model=model,
            data_preprocessor=data_preprocessor,
            reduction=reduction,
            convert_to_freq=not no_convert_to_freq
        )

        output_file = file.rsplit('.', 1)[0] + "." + ("semitones" if no_convert_to_freq else "f0")
        if output is not None:
            os.makedirs(output, exist_ok=True)
            output_file = os.path.join(output, os.path.basename(output_file))

        predictions = [p.cpu().numpy() for p in predictions]
        for fmt in export_format:
            export(fmt, output_file, *predictions)

    return predictions
