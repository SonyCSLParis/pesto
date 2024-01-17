import glob
import itertools
import os

import pytest

import torch


AUDIOS_DIR = os.path.join(os.path.dirname(__file__), "audios")


@pytest.mark.parametrize("file, fmt, convert_to_freq",
                         itertools.product(glob.glob(AUDIOS_DIR + "/*.wav"), ["csv", "npz", "png"], [True, False]))
def test_cli(file, fmt, convert_to_freq):
    if convert_to_freq:
        suffix = ".f0." + fmt
        option = ""
    else:
        suffix = ".semitones." + fmt
        option = " -F"
    os.system(f"pesto {file} --export_format " + fmt + option)
    out_file = file.rsplit('.', 1)[0] + suffix
    assert os.path.isfile(out_file)
    os.unlink(out_file)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
@pytest.mark.parametrize("file, fmt, convert_to_freq",
                         itertools.product(glob.glob(AUDIOS_DIR + "/*.wav"), ["csv", "npz", "png"], [True, False]))
def test_cli_gpu(file, fmt, convert_to_freq):
    if convert_to_freq:
        suffix = ".f0." + fmt
        option = ""
    else:
        suffix = ".semitones." + fmt
        option = " -F"
    os.system(f"pesto {file} --gpu 0 --export_format " + fmt + option)
    out_file = file.rsplit('.', 1)[0] + suffix
    assert os.path.isfile(out_file)
    os.unlink(out_file)
