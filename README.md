# PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective

**tl;dr**: Fast and powerful pitch estimator based on machine learning

**Disclaimer:** This repository contains minimal code and should be used for inference only.
If you want full implementation details or want to use PESTO for research purposes, take a look at ~~[this repository](https://github.com/aRI0U/pesto-full)~~ (work in progress).


## Installation

```shell
pip install pesto
```

### Common issues

- When 

### Dependencies

This repository is implemented in [PyTorch](https://pytorch.org/) and has the following additional dependencies:
- `matplotlib` and `numpy` for basic I/O and plotting operations
- [torchaudio](https://pytorch.org/audio/stable/) for audio loading
- [nnAudio](https://github.com/KinWaiCheuk/nnAudio) for computing the Constant-Q Transform (CQT)

## Usage

### Command-line interface

This package includes a CLI as well as pretrained models.
In order to use it, type in a terminal:
```shell
pesto my_file.wav
```
or
```shell
python -m pesto my_file.wav
```

#### Output formats

The output format can be specified with option `-e`/`--export_format`.
By default, the predicted pitch is saved in a `.csv` file that looks like this:
```
time,frequency,confidence
0.00,185.616,0.907112
0.01,186.764,0.844488
0.02,188.356,0.798015
0.03,190.610,0.746729
0.04,192.952,0.771268
0.05,195.191,0.859440
0.06,196.541,0.864447
0.07,197.809,0.827441
0.08,199.678,0.775208
...
```
This structure is voluntarily the same as in [CREPE](https://github.com/marl/crepe) repo for enabling easy comparison between both methods.

Alternatively, one can choose to save timesteps, pitch, confidence and activation outputs as a `.npz` file.

Finally, you can also visualize the pitch predictions by exporting them as a `png` file. Here is an example:
<p align="center">
  <img src="https://github.com/SonyCSLParis/pesto/blob/master/examples/example.f0.png?raw=true">
</p>
Multiple formats can be specified after the `-e` option.

#### Batch processing

Any number of audio files can be passed in the command for convenient batch processing.
For example, for estimating the pitch of a whole folder of MP3 files, just type:
```shell
pesto my_folder/*.mp3
```

#### Audio format

Internally, this repository uses [torchaudio](https://pytorch.org/audio/stable/backend.html) for loading audio files.
Most audio formats should be supported, refer to torchaudio's documentation for more information. 
Additionally, audio files can have any sampling rate, no resampling is required.

#### Pitch prediction from output probability distribution

By default, the model returns a probability distribution over all pitch bins.
To convert it to a proper pitch, by default we use Argmax-Local Weighted Averaging as in CREPE:

<p align="center">
  <img width="360" src="https://github.com/SonyCSLParis/pesto/blob/master/images/alwa.png?raw=true">
</p>

Alternatively, one can use basic argmax of weighted average with option `-r`/`--reduction`.

#### Miscellaneous

- The step size of the pitch predictions can be specified (in milliseconds) with option `-s`. Note that this number is then converted to an integer hop length for the CQT so there might be a slight difference between the step size you specify and the actual one.
- You can use `-F` option to return directly pitch predictions in semitones instead of frequency.
- If you have access to a GPU, inference speed can be further improved with option `--gpu <gpu_id>`. `--gpu -1` (the default) corresponds to CPU.


### Python API

Alternatively, the functions defined in `pesto/predict.py` can directly be called within another Python code.
In particular, function `predict_from_files` is the one that is directly called by the CLI.

#### Basic usage

```python
import torchaudio
import pesto

# predict the pitch of your audio tensors directly within your own Python code
x, sr = torchaudio.load("my_file.wav")
timesteps, pitch, confidence, activations = pesto.predict(x, sr, step_size=10.)

# you can also predict pitches from audio files directly
pesto.predict_from_files(["example1.wav", "example2.mp3", "example3.ogg"], step_size=10., export_format=["csv"])
```

#### Advanced usage

If not provided,  `pesto.predict` will first load the CQT kernels and the model before performing 
any pitch estimation. If you want to process a significant number of files, calling `predict` several times will then 
re-initialize the same model for each tensor.

To avoid this time-consuming step, one can manually instantiate  the model   and data processor, then pass them directly 
as args to the `predict` function. To do so, one has to use the underlying methods from `pesto.utils`:
```python
import torch

from pesto import predict
from pesto.utils import load_model, load_dataprocessor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("mir-1k", device=device)
data_processor = load_dataprocessor(step_size=0.01, device=device)

for x, sr in ...:
    data_processor.sampling_rate = sr  # The data_processor handles waveform->CQT conversion so it must know the sampling rate
    predictions = predict(x, sr, model=model, data_processor=data_processor)
    ...
```
Note that when passing a list of files to `pesto.predict_from_files(...)` or the CLI directly, the model  is loaded only
once so you don't have to bother with that in general.

## Benchmark

On [MIR-1K]() and [MDB-stem-synth](), PESTO outperforms other self-supervised baselines.
Its performances are close to CREPE's ones, that has 800x more parameters and was trained in a supervised way on a huge dataset containing MIR-1K and MDB-stem-synth, among others.

<p align="center">
  <img width="360" src="https://github.com/SonyCSLParis/pesto/blob/master/images/results.png?raw=true">
</p>

## Speed

TODO

## Cite

If you want to cite this work, 