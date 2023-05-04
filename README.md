# PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective

**tl;dr**: Fast and powerful pitch estimator based on machine learning

**Disclaimer:** This repository contains minimal code and should be used for inference only.
If you want full implementation details or want to use PESTO for research purposes, take a look at [this repository](https://github.com/aRI0U/pesto-full).


## Installation

```shell
pip install pesto
```
*not possible yet*

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
**TODO**

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

$$\textcent = \dfrac{\sum^{m+2}_{i=m-2} y_i \textcent_i}{\sum^{m+2}_{i=m-2} y_i \textcent_i} \text{, with } m = \arg\max_i y_i$$

Alternatively, one can use basic argmax of weighted average with option `-r`/`--reduction`.

#### Miscellaneous

- The step size of the pitch predictions can be specified (in milliseconds) with option `-s`. Note that this number is then converted to an integer hop length for the CQT so there might be a slight difference between the step size you specify and the actual one.
- You can use `-F` option to return directly pitch predictions in semitones instead of frequency.
- If you have access to a GPU, inference speed can be further improved with option `--gpu <gpu_id>`. `--gpu -1` (the default) corresponds to CPU.

## Benchmark

TODO

## Speed

TODO

## TODO

- add licence
- add confidence
- fill sections in README