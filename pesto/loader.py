import logging
import os
from typing import Optional

import torch

from .data import Preprocessor
from .model import PESTO, Resnet1d


log = logging.getLogger(__name__)

harmless_message = ('Error(s) in loading state_dict for PESTO:\n'
                    '\tMissing key(s) in state_dict: "preprocessor.hcqt_kernels.cqt_kernels.0.sqrt_lengths", "preprocessor.hcqt_kernels.cqt_kernels.0.conv.weight". ')


def load_model(checkpoint: str,
               step_size: float,
               sampling_rate: Optional[int] = None,
               **hcqt_kwargs) -> PESTO:
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
        model_path = os.path.join(os.path.dirname(__file__), "weights", checkpoint + ".ckpt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"You passed an invalid checkpoint file: {checkpoint}.")

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    hparams = checkpoint["hparams"]
    state_dict = checkpoint["state_dict"]
    hcqt_params = checkpoint["hcqt_params"]
    hcqt_params.update(hcqt_kwargs)

    # instantiate preprocessor
    preprocessor = Preprocessor(hop_size=step_size, sampling_rate=sampling_rate, **hcqt_params)

    # instantiate PESTO encoder
    encoder = Resnet1d(**hparams["encoder"])

    # instantiate main PESTO module and load its weights
    model = PESTO(encoder,
                  preprocessor=preprocessor,
                  crop_kwargs=hparams["pitch_shift"],
                  reduction=hparams["reduction"])
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if str(e) != harmless_message:
            log.warning(repr(e))
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model
