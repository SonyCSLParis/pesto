import itertools

import pytest

import torch

from pesto import load_model, predict
from .utils import generate_synth_data


@pytest.fixture
def model():
    return load_model('mir-1k_g7', step_size=10.)


@pytest.fixture
def synth_data_16k():
    return generate_synth_data(pitch=69, duration=5., sr=16000), 16000


@pytest.mark.parametrize('reduction', ["argmax", "mean", "alwa"])
def test_shape_no_batch(model, synth_data_16k, reduction):
    x, sr = synth_data_16k

    model.reduction = reduction

    num_samples = x.size(-1)

    num_timesteps = int(num_samples * 1000 / (model.hop_size * sr)) + 1

    preds, conf, amplitude, activations = model(x, sr=sr, return_activations=True)

    assert preds.shape == (num_timesteps,)
    assert conf.shape == (num_timesteps,)
    assert amplitude.shape == (num_timesteps,)
    assert activations.shape == (num_timesteps, 128 * model.bins_per_semitone)


@pytest.mark.parametrize('sr, reduction',
                         itertools.product([16000, 44100, 48000], ["argmax", "mean", "alwa"]))
def test_shape_batch(model, sr, reduction):
    model.reduction = reduction

    batch_size = 13

    batch = torch.stack([
        generate_synth_data(pitch=p, duration=5., sr=sr)
        for p in range(50, 50+batch_size)
    ])

    num_timesteps = int(batch.size(-1) * 1000 / (model.hop_size * sr)) + 1

    preds, conf, amplitude, activations = model(batch, sr=sr, return_activations=True)

    assert preds.shape == (batch_size, num_timesteps)
    assert conf.shape == (batch_size, num_timesteps)
    assert activations.shape == (batch_size, num_timesteps, 128 * model.bins_per_semitone)


@pytest.mark.parametrize('step_size, reduction',
                         itertools.product([10., 20., 50., 100], ["argmax", "mean", "alwa"]))
def test_predict_shape_no_batch(synth_data_16k, step_size, reduction):
    x, sr = synth_data_16k

    num_samples = x.size(-1)

    num_timesteps = int(num_samples * 1000 / (step_size * sr)) + 1

    timesteps, preds, conf, activations = predict(x,
                                                  sr,
                                                  step_size=step_size,
                                                  reduction=reduction)

    assert timesteps.shape == (num_timesteps,)
    assert preds.shape == (num_timesteps,)
    assert conf.shape == (num_timesteps,)


@pytest.mark.parametrize('sr, step_size, reduction',
                         itertools.product([16000, 44100, 48000], [10., 20., 50., 100.], ["argmax", "mean", "alwa"]))
def test_predict_shape_batch(sr, step_size, reduction):
    batch_size = 13

    batch = torch.stack([
        generate_synth_data(pitch=p, duration=5., sr=sr)
        for p in range(50, 50+batch_size)
    ])

    num_timesteps = int(batch.size(-1) * 1000 / (step_size * sr)) + 1

    timesteps, preds, conf, activations = predict(batch,
                                                  sr=sr,
                                                  step_size=step_size,
                                                  reduction=reduction)

    assert timesteps.shape == (num_timesteps,)
    assert preds.shape == (batch_size, num_timesteps)
    assert conf.shape == (batch_size, num_timesteps)
