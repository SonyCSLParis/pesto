import os.path

import pytest

import numpy as np
import onnxruntime as ort

from .utils import generate_synth_data


SAMPLE_RATE = 44100
HOP = 1024
BATCHED = True
MARGIN = 8192 // (2 * HOP)
ONNX_PATH = f"mir-1k_g7_{SAMPLE_RATE}_{HOP}.onnx"


@pytest.fixture
def model():
    session = ort.InferenceSession(ONNX_PATH)
    return session


@pytest.mark.skipif(not os.path.exists(ONNX_PATH), reason="ONNX path does not exist")
@pytest.mark.parametrize("pitch", range(50, 80))
def test_performances(model, pitch):
    x = generate_synth_data(pitch, sr=SAMPLE_RATE)
    if BATCHED:
        x = x.unsqueeze(0)

    # Initialize cache state for the ONNX model
    cache_state = np.zeros((1, model.get_inputs()[1].shape[1]), dtype=np.float32)

    outputs = model.run(None, {"audio": x.numpy(), "cache": cache_state})
    preds = outputs[0]  # First output is predictions

    # remove first elems of the audio
    preds = preds[..., MARGIN:]

    expected = np.full_like(preds, pitch)
    np.testing.assert_allclose(preds, expected, atol=0.1, rtol=0.1)


@pytest.mark.skipif(not os.path.exists(ONNX_PATH), reason="ONNX path does not exist")
@pytest.mark.parametrize("pitch", range(50, 80))
def test_streaming(model, pitch):
    x = generate_synth_data(pitch, duration=1000 * HOP / SAMPLE_RATE, sr=SAMPLE_RATE)
    if BATCHED:
        x = x.unsqueeze(0)

    # Initialize cache state for streaming
    cache_state = np.zeros((1, model.get_inputs()[1].shape[1]), dtype=np.float32)

    preds = []
    for chunk in x.split(HOP, dim=-1):
        outputs = model.run(None, {"audio": chunk.numpy(), "cache": cache_state})
        p = outputs[0]  # predictions
        cache_state = outputs[4]  # cache_out (5th output)
        preds.append(p)

    preds = np.concatenate(preds, axis=int(BATCHED))
    preds = preds[..., MARGIN:]

    expected = np.full_like(preds, pitch)
    np.testing.assert_allclose(preds, expected, atol=0.1, rtol=0.1)
