# 🌿 PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective

**tl;dr**: 🌿 PESTO is a fast and powerful pitch estimator based on machine learning. 

This repository provides a minimal code implementation for inference only. For full training details and research experiments, please refer to the [full implementation](https://github.com/SonyCSLParis/pesto-full) or the [paper 📝(arXiv)](https://arxiv.org/abs/2309.02265).

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
   - [Command-line Interface](#command-line-interface)
     - [Output Formats](#output-formats)
     - [Batch Processing](#batch-processing)
     - [Audio Format](#audio-format)
     - [Pitch Prediction Options](#pitch-prediction-options)
     - [Miscellaneous](#miscellaneous)
   - [Python API](#python-api)
     - [Basic Usage](#basic-usage)
     - [Advanced Usage](#advanced-usage)
3. [🚀NEW: Streaming Implementation](#streaming-implementation)
4. [🚀NEW: Export Compiled Model](#export-compiled-model)
5. [Performance and Speed Benchmarks](#performance-and-speed-benchmarks)
6. [Contributing](#contributing)
7. [Citation](#citation)
8. [Credits](#credits)

---

## Installation

```shell
pip install pesto-pitch
```
That's it!

### Dependencies
- **PyTorch:** For model inference.
- **numpy:** For basic I/O operations.
- **torchaudio:** For audio loading.
- **matplotlib:** For exporting pitch predictions as images (optional).

**Note:** It is recommended to install PyTorch before PESTO following the [official guide](https://pytorch.org/get-started/locally/) if you're working in a clean environment.

---

## Usage

### Command-line Interface

To use the CLI, run the following command in a terminal to estimate the pitch of an audio file using a pretrained model:
```shell
pesto my_file.wav
```
or
```shell
python -m pesto my_file.wav
```

**Note:** You can use `pesto -h` to get help on output formats and additional options.

#### Output Formats

By default, the predicted pitch is saved in a `.csv` file:
```csv
time,frequency,confidence
0.00,185.616,0.907112
0.01,186.764,0.844488
0.02,188.356,0.798015
0.03,190.610,0.746729
0.04,192.952,0.771268
0.05,195.191,0.859440
...
```
This structure is voluntarily the same as in [CREPE](https://github.com/marl/crepe) repo for easy comparison between both methods.

Alternatively, you can specify the output format with the `-e`/`--export_format` option:
- **.npz:** Timesteps, pitch, confidence, and activations.
- **.png:** Visualization of pitch predictions (requires matplotlib).
Here is an example output of using the '-e .png' option:
![example f0](https://github.com/SonyCSLParis/pesto/assets/36546630/5aa18c23-0154-4d2d-8021-2c23277b27a3)


Multiple formats can be specified after the `-e` option.

#### Batch Processing

To estimate the pitch for multiple files:
```shell
pesto my_folder/*.mp3
```

#### Audio Format

PESTO uses [torchaudio](https://pytorch.org/audio/stable/backend.html) for audio loading, supporting various formats and sampling rates without needing resampling.

#### Pitch Decoding Options

By default, the model outputs a probability distribution over pitch bins, which is converted to pitch using Argmax-Local Weighted Averaging decoding (similar to CREPE)
![Weighted Averaging](https://github.com/SonyCSLParis/pesto/assets/36546630/3138c33f-672a-477f-95a9-acaacf4418ab)

Alternatively, basic argmax or weighted average can be used with the `-r`/`--reduction` option.

#### Miscellaneous

- **Step Size:** Set with `-s` (in milliseconds). The actual hop length might differ slightly due to integer conversion.
- **Output Units:** Use `-F` to return pitch in semitones instead of frequency.
- **GPU Inference:** Specify the GPU with `--gpu <gpu_id>`. The default `--gpu -1` uses the CPU.

---

### Python API

You can also call functions defined in `pesto/predict.py` directly in your Python code.  `predict_from_files` is the the function used by default when running the CLI.

#### Basic Usage

```python
import torchaudio
import pesto

# Load audio (ensure mono; stereo channels are treated as separate batch dimensions)
x, sr = torchaudio.load("my_file.wav")
x = x.mean(dim=0)  # PESTO takes mono audio as input

# Predict pitch. x can be (num_samples) or (batch, num_samples)
timesteps, pitch, confidence, activations = pesto.predict(x, sr)

# Using a custom checkpoint:
predictions = pesto.predict(x, sr, model_name="/path/to/checkpoint.ckpt")

# Predicting from multiple files:
pesto.predict_from_files(["example1.wav", "example2.mp3"], export_format=["csv"])
```

**Note:**  If you forget to convert a stereo to mono, channels will be treated as batch dimensions and you will get predictions for each channel separately.

#### Advanced Usage

`pesto.predict` will first load the CQT kernels and the model before performing 
any pitch estimation. To avoid reloading the model for every prediction, instantiate the model once using `load_model`:

```python
import torch
from pesto import load_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pesto_model = load_model("mir-1k_g7", step_size=20.).to(device)

for x, sr in your_audio_loader:
    x = x.to(device)
    predictions, confidence, amplitude, activations = pesto_model(x, sr)
    # Process predictions as needed
```

Since PESTO is a subclass of `nn.Module`, it supports batched inputs, making integration into custom architectures straightforward.

Example:
```python
import torch
import torch.nn as nn
from pesto import load_model

class MyGreatModel(nn.Module):
    def __init__(self, step_size, sr=44100, **kwargs):
        super(MyGreatModel, self).__init__()
        self.f0_estimator = load_model("mir-1k_g7", step_size, sampling_rate=sr)
    
    def forward(self, x):
        with torch.no_grad():
            f0, conf, amp = self.f0_estimator(x, convert_to_freq=True, return_activations=False)
        # Further processing with f0, conf, and amp
        return f0, conf, amp
```

---

## NEW: Streaming Implementation

PESTO now supports streaming audio processing. To enable streaming, load the model with `streaming=True`:

```python
import time
import torch
from pesto import load_model

pesto_model = load_model(
    'mir-1k_g7',
    step_size=5.,
    sampling_rate=48000,
    streaming=True,  # Enable streaming mode
    max_batch_size=4
)

while True:
    buffers = torch.randn(3, 240)  # Acquire a batch of 3 audio buffers
    pitch, conf, amp = pesto_model(buffers, return_activations=False)
    print(pitch)
    time.sleep(0.005)
```

**Note:** The streaming implementation uses an internal circular buffer per batch dimension. Do not use the same model for different streams sequentially.

---

## NEW:  Export Compiled Model

For reduced latency and easier integration into other applications, you can export a compiled version of the model:
```shell
python -m realtime.export_jit --help
```
**Note:** When using the compiled model, parameters like sampling rate and step size become fixed and cannot be modified later.

---

## Performance and Speed Benchmarks

PESTO outperforms other self-supervised baselines on datasets such as MIR-1K and MDB-stem-synth and achieves performance close to supervised methods like CREPE (with 800x more parameters).

![image](https://github.com/SonyCSLParis/pesto/assets/36546630/d6ae0306-ba8b-465a-8ca7-f916479a0ba5)

**Benchmark Example:**
- **Audio File:** `wav` (2m51s)
- **Hardware:** Intel i7-1185G7 (11th Gen, 8 cores)
- **Speed:** With a 10ms step size, PESTO would perform pitch estimation of the file in 13 seconds (~12 times faster than real-time) while CREPE would take 12 minutes!.

![speed](https://github.com/SonyCSLParis/pesto/assets/36546630/612b1850-c2cf-4df1-9824-b8460a2f9148)


The y-axis in the speed graph is log-scaled.

### Inference on GPU

Under the hood, the input is passed to the model as a single batch of CQT frames,
so pitch is estimated for the whole track in parallel, making inference extremely fast.

However, when dealing with very large audio files, processing the whole track at once can lead to OOM errors. 
To circumvent this, one can split the batch of CQT frames into multiple chunks by setting option `-c`/`--num_chunks`.
Chunks will be processed sequentially, thus reducing memory usage.

As an example, a 48kHz audio file of 1 hour can be processed in 20 seconds only on a single GTX 1080 Ti when split into 10 chunks.

## Contributing

- Currently, only a single model trained on [MIR-1K](https://zenodo.org/record/3532216#.ZG0kWhlBxhE) is provided.
- Contributions to model architectures, training data, hyperparameters, or additional pretrained checkpoints are welcome.
- Despite PESTO being significantly faster than real-time, it is currently implemented as standard PyTorch and may be further accelerated.
- Suggestions for improving inference speed aree more than welcome!

Feel free to contact us with any ideas to enhance PESTO.

---

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{PESTO,
    author = {Riou, Alain and Lattner, Stefan and Hadjeres, Gaëtan and Peeters, Geoffroy},
    booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023},
    publisher = {International Society for Music Information Retrieval},
    title = {PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective},
    year = {2023}
}
```

---

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

