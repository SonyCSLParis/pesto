import itertools

import pytest

import torch

from pesto import load_model
from .utils import generate_synth_data


@pytest.fixture
def model():
    return load_model('mir-1k_g7', step_size=1.)


@pytest.mark.parametrize('pitch, sr, reduction',
                         itertools.product(range(50, 80), [16000, 44100, 48000], ["argmax", "alwa"]))
def test_performances(model, pitch, sr, reduction):
    x = generate_synth_data(pitch, sr=sr)

    preds, *_ = model(x, sr=sr, return_activations=False)

    # remove boundary effects
    preds = preds[10:-10]

    torch.testing.assert_close(preds, torch.full_like(preds, pitch), atol=0.1, rtol=0.1)
