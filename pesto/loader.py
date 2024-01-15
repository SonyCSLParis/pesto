import os
from typing import Optional

import torch

from .data import Preprocessor
from .model import PESTO, Resnet1d


def load_model(checkpoint: str,
               step_size: float,
               sampling_rate: Optional[int] = None) -> PESTO:
    r"""Load a trained model from a checkpoint file.
    See https://github.com/SonyCSLParis/pesto-full/blob/master/src/models/pesto.py for the structure of the checkpoint.

    Args:
        checkpoint (str): path to the checkpoint or name of the checkpoint file (if using a provided checkpoint)
        step_size (float): hop size in milliseconds
        sampling_rate (int, optional): sampling rate of the audios.
            If not provided, it can be inferred dynamically as well.
    Returns:
        PESTO: instance of PESTO model
    """
    if os.path.exists(checkpoint):  # handle user-provided checkpoints
        model_path = checkpoint
    else:
        model_path = os.path.join(os.path.dirname(__file__), "weights", checkpoint + ".pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"You passed an invalid checkpoint file: {checkpoint}.")

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    hparams = checkpoint["hparams"]
    hcqt_params = checkpoint["hcqt_params"]
    state_dict = checkpoint["state_dict"]

    # instantiate preprocessor
    preprocessor = Preprocessor(hop_size=step_size, sampling_rate=sampling_rate, **hcqt_params)

    # instantiate PESTO encoder
    encoder = Resnet1d(**hparams["encoder"])

    # instantiate main PESTO module and load its weights
    model = PESTO(encoder,
                  preprocessor=preprocessor,
                  crop_kwargs=hparams["pitch_shift"],
                  reduction=hparams["reduction"])
    model.load_state_dict(state_dict)
    model.eval()

    return model
