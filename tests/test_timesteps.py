import pytest

import torch

from pesto import predict
from .utils import generate_synth_data


@pytest.fixture
def synth_data_16k():
    return generate_synth_data(pitch=69, duration=5., sr=16000), 16000


@pytest.mark.parametrize('step_size', [10., 20., 50., 100])
def test_build_timesteps(synth_data_16k, step_size):
    timesteps, *_ = predict(*synth_data_16k, step_size=step_size)
    diff = timesteps[1:] - timesteps[:-1]
    torch.testing.assert_close(diff, torch.full_like(diff, step_size))
