import os.path

import pytest

import torch

from .utils import generate_synth_data


DATE = "250217"
SAMPLE_RATE = 48000
HOP = 256
BATCHED = True
MARGIN = 8192 // (2 * HOP)
SCRIPT_PATH = f"realtime/{DATE}_sr{SAMPLE_RATE // 1000:d}k_h{HOP:d}.pt"


@pytest.fixture
def model():
    return torch.jit.load(SCRIPT_PATH)


@pytest.mark.skipif(not os.path.exists(SCRIPT_PATH), reason="Script path does not exist")
@pytest.mark.parametrize('pitch', range(50, 80))
def test_performances(model, pitch):
    x = generate_synth_data(pitch, sr=SAMPLE_RATE)
    if BATCHED:
        x.unsqueeze_(0)

    preds, *_ = model(x)

    # remove first elems of the audio
    preds = preds[..., MARGIN:]

    torch.testing.assert_allclose(preds, torch.full_like(preds, pitch), atol=0.1, rtol=0.1)


@pytest.mark.skipif(not os.path.exists(SCRIPT_PATH), reason="Script path does not exist")
@pytest.mark.parametrize('pitch', range(50, 80))
def test_streaming(model, pitch):
    x = generate_synth_data(pitch, duration=1000 * HOP / SAMPLE_RATE, sr=SAMPLE_RATE)
    if BATCHED:
        x.unsqueeze_(0)

    preds = []
    for chunk in x.split(HOP, dim=-1):
        p, *_ = model(chunk)
        preds.append(p)

    preds = torch.cat(preds, dim=int(BATCHED))
    preds = preds[..., MARGIN:]

    torch.testing.assert_allclose(preds, torch.full_like(preds, pitch), atol=0.1, rtol=0.1)
