import itertools

import pytest

import torch

from pesto import load_model
from .utils import generate_synth_data


SAMPLING_RATE = 48000
HOP = 128


@pytest.fixture
def model():
    return load_model('mir-1k_g7',
                      step_size=1000 * HOP / SAMPLING_RATE,
                      streaming=True,
                      sampling_rate=SAMPLING_RATE,
                      max_batch_size=4,
                      mirror=1.)


@pytest.mark.parametrize('pitch, reduction',
                         itertools.product(range(50, 80), ["argmax", "alwa"]))
def test_performances(model, pitch, reduction):
    x = generate_synth_data(pitch, duration=1000 * HOP / SAMPLING_RATE, sr=SAMPLING_RATE)
    x = x.unsqueeze(0).repeat(3, 1)

    preds = []
    for chunk in x.split(HOP, dim=-1):
        p, *_ = model(chunk)
        preds.append(p)

    preds = torch.cat(preds, dim=1)[5:]

    torch.testing.assert_close(preds, torch.full_like(preds, pitch), atol=0.1, rtol=0.1)
