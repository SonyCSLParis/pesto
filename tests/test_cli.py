import glob
import itertools
import os

import pytest


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
    assert os.path.isfile(file.rsplit('.', 1)[0] + suffix)
