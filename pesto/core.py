import os
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torchaudio
from tqdm import tqdm

from .loader import load_model
from .model import PESTO
from .utils import export


def _predict(x: torch.Tensor,
             sr: int,
             model: PESTO,
             num_chunks: int = 1,
             convert_to_freq: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, confidence, activations = [], [], []
    try:
        for chunk in x.chunk(chunks=num_chunks):
            pred, conf, vol, act = model(chunk, sr=sr, convert_to_freq=convert_to_freq, return_activations=True)
            preds.append(pred)
            confidence.append(conf)
            activations.append(act)
    except torch.cuda.OutOfMemoryError:
        raise torch.cuda.OutOfMemoryError("Got an out-of-memory error while performing pitch estimation. "
                                          "Please increase the number of chunks with option `-c`/`--chunks` "
                                          "to reduce GPU memory usage.")

    preds = torch.cat(preds, dim=0)
    confidence = torch.cat(confidence, dim=0)
    activations = torch.cat(activations, dim=0)

    # compute timesteps
    timesteps = torch.arange(preds.size(-1), device=x.device) * model.hop_size

    return timesteps, preds, confidence, activations


def predict(x: torch.Tensor,
            sr: int,
            step_size: float = 10.,
            model_name: str = "mir-1k_g7",
            reduction: str = "alwa",
            num_chunks: int = 1,
            convert_to_freq: bool = True,
            inference_mode: bool = True,
            no_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Main prediction function.

    Args:
        x (torch.Tensor): input audio tensor, can be provided as a batch but should be mono,
            shape (num_samples) or (batch_size, num_samples)
        sr (int, optional): sampling rate. If not specified, uses the current sampling rate of the model.
        step_size (float, optional): step size between each CQT frame in milliseconds.
            If a `PESTO` object is passed as `model`, this will be ignored.
        model_name: name of PESTO model. Can be a path to a custom PESTO checkpoint or the name of a standard model.
        reduction (str): reduction method for converting activation probabilities to log-frequencies.
        num_chunks (int): number of chunks to split the input audios in.
            Default is 1 (all CQT frames in parallel) but it can be increased to reduce memory usage
            and prevent out-of-memory errors.
        convert_to_freq (bool): whether predictions should be converted to frequencies or not.
        inference_mode (bool): whether to run with `torch.inference_mode`.
        no_grad (bool): whether to run with `torch.no_grad`. If set to `False`, argument `inference_mode` is ignored.

    Returns:
        timesteps (torch.Tensor): timesteps corresponding to each pitch prediction, shape (num_timesteps)
        preds (torch.Tensor): pitch predictions in SEMITONES, shape (batch_size?, num_timesteps)
            where `num_timesteps` ~= `num_samples` / (`self.hop_size` * `sr`)
        confidence (torch.Tensor): confidence of whether frame is voiced or unvoiced in [0, 1],
            shape (batch_size?, num_timesteps)
        activations (torch.Tensor): activations of the model, shape (batch_size?, num_timesteps, output_dim)
    """
    # sanity checks
    assert x.ndim <= 2, \
        f"Audio file should have shape (num_samples) or (batch_size, num_samples), but found shape {x.size()}."

    inference_mode = inference_mode and no_grad
    with torch.no_grad() if no_grad and not inference_mode else torch.inference_mode(mode=inference_mode):
        model = load_model(model_name, step_size, sampling_rate=sr).to(x.device)
        model.reduction = reduction

        return _predict(x, sr, model, num_chunks=num_chunks, convert_to_freq=convert_to_freq)


def predict_from_files(
        audio_files: Union[str, Sequence[str]],
        model_name: str = "mir-1k_g7",
        output: Optional[str] = None,
        step_size: float = 10.,
        reduction: str = "alwa",
        export_format: Sequence[str] = ("csv",),
        no_convert_to_freq: bool = False,
        num_chunks: int = 1,
        gpu: int = -1):
    r"""

    Args:
        audio_files: audio files to process
        model_name: name of the model. Currently only `mir-1k` is supported.
        output:
        step_size: hop length in milliseconds
        reduction:
        export_format (Sequence[str]): format to export the predictions to.
            Currently formats supported are: ["csv", "npz", "json"].
        no_convert_to_freq: whether convert output values to Hz or keep fractional MIDI pitches
        num_chunks (int): number of chunks to divide the inputs into. Increase this value if you encounter OOM errors.
        gpu: index of GPU to use (-1 for CPU)
    """
    if isinstance(audio_files, str):
        audio_files = [audio_files]

    if gpu >= 0 and not torch.cuda.is_available():
        warnings.warn("You're trying to use the GPU but no GPU has been found. Using CPU instead...")
        gpu = -1
    device = torch.device(f"cuda:{gpu:d}" if gpu >= 0 else "cpu")

    # define model
    model = load_model(model_name, step_size=step_size).to(device)
    model.reduction = reduction

    pbar = tqdm(audio_files)

    with torch.inference_mode():  # here the purpose is to write results in disk, so there is no point storing gradients
        for file in pbar:
            pbar.set_description(file)

            # load audio file
            try:
                x, sr = torchaudio.load(file)
            except Exception as e:
                print(e, f"Skipping {file}...")
                continue

            x = x.mean(dim=0).to(device)  # convert to mono then pass to the right device

            # compute the predictions
            predictions = _predict(x, sr, model=model, convert_to_freq=not no_convert_to_freq, num_chunks=num_chunks)

            output_file = file.rsplit('.', 1)[0] + "." + ("semitones" if no_convert_to_freq else "f0")
            if output is not None:
                os.makedirs(output, exist_ok=True)
                output_file = os.path.join(output, os.path.basename(output_file))

            predictions = [p.cpu().numpy() for p in predictions]
            for fmt in export_format:
                export(fmt, output_file, *predictions)
