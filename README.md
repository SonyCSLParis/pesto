# PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective
 
**tl;dr**: Fast and powerful pitch estimator based on machine learning

This code is the implementation of the [PESTO paper](https://arxiv.org/abs/2309.02265),
that received the Best Paper Award at [ISMIR 2023](https://ismir2023.ismir.net/).

**Disclaimer:** This repository contains minimal code and should be used for inference only.
If you want full implementation details or want to use PESTO for research purposes, take a look at [this repository](https://github.com/SonyCSLParis/pesto-full).

## Installation

```shell
pip install pesto-pitch
```
That's it!

### Dependencies

This repository is implemented in [PyTorch](https://pytorch.org/) and has the following additional dependencies:
- `numpy` for basic I/O  operations
- [torchaudio](https://pytorch.org/audio/stable/) for audio loading
- `matplotlib` for exporting pitch predictions as images (optional)

**Warning:** If installing in a clean environment, it may be safer to first install PyTorch [the recommended way](https://pytorch.org/get-started/locally/) before PESTO.

## Usage

### Command-line interface

This package includes a CLI as well as pretrained models.
To use it, type in a terminal:
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
This structure is voluntarily the same as in [CREPE](https://github.com/marl/crepe) repo for easy comparison between both methods.

Alternatively, one can save timesteps, pitch, confidence and activation outputs as a `.npz` file.

Finally, you can also visualize the pitch predictions by exporting them as a `png` file (you need `matplotlib` to be installed for PNG export). Here is an example:

![example f0](https://github.com/SonyCSLParis/pesto/assets/36546630/5aa18c23-0154-4d2d-8021-2c23277b27a3)


Multiple formats can be specified after the `-e` option.

#### Batch processing

Any number of audio files can be passed in the command for convenient batch processing.
For example, to estimate the pitch of a whole folder of MP3 files, just type:
```shell
pesto my_folder/*.mp3
```

#### Audio format

Internally, this repository uses [torchaudio](https://pytorch.org/audio/stable/backend.html) for loading audio files.
Most audio formats should be supported; refer to torchaudio's documentation for more information. 
Additionally, audio files can have any sampling rate; no resampling is required.

#### Pitch prediction from the output probability distribution

By default, the model returns a probability distribution over all pitch bins.
To convert it to a proper pitch, by default, we use Argmax-Local Weighted Averaging as in CREPE:
![image](https://github.com/SonyCSLParis/pesto/assets/36546630/3138c33f-672a-477f-95a9-acaacf4418ab)


Alternatively, one can use basic argmax of weighted average with option `-r`/`--reduction`.

#### Miscellaneous

- The step size of the pitch predictions can be specified (in milliseconds) with option `-s`. Note that this number is then converted to an integer hop length for the CQT so there might be a slight difference between the step size you specify and the actual one.
- You can use `-F` option to return directly pitch predictions in semitones instead of frequency.
- If you can access a GPU, inference speed can be further improved with option `--gpu <gpu_id>`. `--gpu -1` (the default) corresponds to CPU.


### Python API

Alternatively, the functions defined in `pesto/predict.py` can directly be called within another Python code.
In particular, the function `predict_from_files` is the one that the CLI directly calls.

#### Basic usage

```python
import torchaudio
import pesto

# predict the pitch of your audio tensors directly within your own Python code
x, sr = torchaudio.load("my_file.wav")
x = x.mean(dim=0)  # PESTO takes mono audio as input
timesteps, pitch, confidence, activations = pesto.predict(x, sr)

# or predict using your own custom checkpoint
predictions = pesto.predict(x, sr, model_name="/path/to/checkpoint.ckpt")

# You can also predict pitches from audio files directly
pesto.predict_from_files(["example1.wav", "example2.mp3", "example3.ogg"], export_format=["csv"])
```
`pesto.predict` supports batched inputs, which should then have shape `(batch_size, num_samples)`.

**Warning:** If you forget to convert a stereo audio in mono, channels will be treated as batch dimensions and you will 
get predictions for each channel separately.

#### Advanced usage

`pesto.predict` will first load the CQT kernels and the model before performing 
any pitch estimation. If you want to process a significant number of files, calling `predict` several times will then 
re-initialize the same model for each tensor.

To avoid this time-consuming step, one can manually instantiate  the model with `load_model`,
then call the forward method of the model instead:

```python
import torch

from pesto import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pesto_model = load_model("mir-1k_g7", step_size=20.).to(device)

for x, sr in ...:
    x = x.to(device)
    predictions, confidence, amplitude, activations = pesto_model(x, sr)
    ...
```

Note that the `PESTO` object returned by `load_model` is a subclass of `nn.Module` 
and its `forward` method also supports batched inputs.
One can therefore easily integrate PESTO within their own architecture by doing:
```python
import torch
import torch.nn as nn

from pesto import load_model


class MyGreatModel(nn.Module):
    def __init__(self, step_size, sr=44100, *args, **kwargs):
        super(MyGreatModel, self).__init__()
        self.f0_estimator = load_model("mir-1k_g7", step_size, sampling_rate=sr)
        ...

    def forward(self, x):
        with torch.no_grad():
            f0, conf, amp = self.f0_estimator(x, convert_to_freq=True, return_activations=False)
        ...
```

### NEW: Streaming implementation

PESTO can now process directly audio streams instead of full audio files.
To use a streamable version of PESTO, just add `streaming=True` to the `load_model` function.

```python
import time
import torch

from pesto import load_model


pesto_model = load_model('mir-1k_g7',
                         step_size=5.,
                         sampling_rate=48000,
                         streaming=True,  # that's it
                         max_batch_size=4)

while True:
    buffers = torch.randn(3, 240)  # acquire a batch of 3 audio buffers
    pitch, conf, amp = pesto_model(buffers, return_activations=False)
    print(pitch)
    time.sleep(0.005)
```

**WARNING:** The streaming implementation of PESTO uses an internal circular buffer in the CQT module for each batched dimension.
Do not use the same model for different streams sequentially.

### NEW: Export compiled model

For reducing further latency and/or integrating PESTO in anotaher application, the trained model can be compiled.
To do so, we provide a script `realtime/export_jit.py`.

A drawback of this method is that the parameters of the `load_model` function (sampling rate, step size...) are frozen 
and cannot be modified after, which makes this method less flexible.

## Performances

On [MIR-1K](https://zenodo.org/record/3532216#.ZG0kWhlBxhE) and [MDB-stem-synth](https://zenodo.org/records/1481172), PESTO outperforms other self-supervised baselines.
Its performances are close to CREPE's, which has 800x more parameters and was trained in a supervised way on a vast 
dataset containing MIR-1K and MDB-stem-synth, among others.

![image](https://github.com/SonyCSLParis/pesto/assets/36546630/d6ae0306-ba8b-465a-8ca7-f916479a0ba5)


## Speed benchmark

PESTO is a very lightweight model and is therefore very fast at inference time.
As CQT frames are processed independently, the actual speed of the pitch estimation process mainly depends on the 
granularity of the predictions, which can be controlled with the `--step_size` parameter (10ms by default).

Here is a speed comparison between CREPE and PESTO, averaged over 10 runs on the same machine.

![speed](https://github.com/SonyCSLParis/pesto/assets/36546630/612b1850-c2cf-4df1-9824-b8460a2f9148)


- Audio file: `wav` format, 2m51s
- Hardware: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz, 8 cores

Note that the *y*-axis is in log-scale: with a step size of 10ms (the default),
PESTO would perform pitch estimation of the file in 13 seconds (~12 times faster than real-time) while CREPE would take 12 minutes!
It is therefore more suited to applications that need very fast pitch estimation without relying on GPU resources.

### Inference on GPU

The underlying PESTO pitch estimator is a standard PyTorch module and can therefore use the GPU,
if available, by setting option `--gpu` to the id of the device you want to use for pitch estimation.

Under the hood, the input is passed to the model as a single batch of CQT frames,
so pitch is estimated for the whole track in parallel, making inference extremely fast.

However, when dealing with very large audio files, processing the whole track at once can lead to OOM errors. 
To circumvent this, one can split the batch of CQT frames into multiple chunks by setting option `-c`/`--num_chunks`.
Chunks will be processed sequentially, thus reducing memory usage.

As an example, a 48kHz audio file of 1 hour can be processed in 20 seconds only on a single GTX 1080 Ti when split into 10 chunks.

## Contributing

- Currently, only a single model trained on [MIR-1K](https://zenodo.org/record/3532216#.ZG0kWhlBxhE) is provided.
Feel free to play with the architecture, training data or hyperparameters and to submit your checkpoints as PRs if you get better performances than
the provided pretrained model. 
- Despite PESTO being significantly faster than real-time, it is currently implemented as standard PyTorch and may be further accelerated.
Any suggestions for improving speed are more than welcome!

More generally, do not hesitate to contact me if you have ideas to improve PESTO's recipe!

## Cite

If you want to use this work, please cite:
```
@inproceedings{PESTO,
    author = {Riou, Alain and Lattner, Stefan and Hadjeres, Gaëtan and Peeters, Geoffroy},
    booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023},
    publisher = {International Society for Music Information Retrieval},
    title = {PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective},
    year = {2023}
}
```

## Credits

- [nnAudio](https://github.com/KinWaiCheuk/nnAudio) for the original CQT implementation
- [multipitch-architectures](https://github.com/christofw/multipitch_architectures) for the original architecture of the model

```
@ARTICLE{9174990,
    author={K. W. {Cheuk} and H. {Anderson} and K. {Agres} and D. {Herremans}},
    journal={IEEE Access}, 
    title={nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks}, 
    year={2020},
    volume={8},
    number={},
    pages={161981-162003},
    doi={10.1109/ACCESS.2020.3019084}}
@ARTICLE{9865174,
    author={Weiß, Christof and Peeters, Geoffroy},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    title={Comparing Deep Models and Evaluation Strategies for Multi-Pitch Estimation in Music Recordings}, 
    year={2022},
    volume={30},
    number={},
    pages={2814-2827},
    doi={10.1109/TASLP.2022.3200547}}
```
