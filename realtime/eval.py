import datetime
import os
import sys
from functools import partial
from math import ceil


import torch
import torch.nn.functional as F
import torchaudio
import time
import matplotlib.pyplot as plt
import numpy as np

# import pyaudio
# from pythonosc import udp_client


from pesto import load_model


@torch.inference_mode()
def evaluate_nostream(model, wav_file):
    x, sr = torchaudio.load(wav_file)

    f0, conf, vol = model(x, convert_to_freq=True, return_activations=False)
    f0.squeeze_(0)
    t = torch.arange(len(f0)).float() * 0.02

    out_file = wav_file.replace('.wav', '_nostream.csv')
    np.savetxt(out_file, torch.stack([t, f0], dim=1).numpy(), header='t,f', fmt='%.3f', delimiter=',')


@torch.inference_mode()
def evaluate_stream(model, wav_file):
    kw, hl = 4096, 320

    model = model()
    model.confidence = Dummy()

    x, sr = torchaudio.load(wav_file)
    x = x.mean(dim=0)
    x = F.pad(x, ((hl//2) - (kw // 2) % (hl//2), kw // 2))
    x = F.pad(x, (0, hl - len(x) % hl))
    f0 = []  # model(x, convert_to_freq=True, return_activations=False)[0]
    for chunk in x.split(hl):
        f = model(chunk, convert_to_freq=True, return_activations=False)[0]
        f0.append(f.item())

    f0 = torch.tensor(f0[:-6])
    t = torch.arange(len(f0)).float() * 0.02

    out_file = wav_file.replace('.wav', '_stream.csv')
    np.savetxt(out_file, torch.stack([t, f0], dim=1).numpy(), header='t,f', fmt='%.3f', delimiter=',')
    if False:
        plt.plot(t, f0.log())
        plt.plot(t, np.log(
            np.loadtxt(wav_file.replace('.wav', '-pitch.csv'), delimiter=',')[:, 1])
        )
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)
    else:
        audio = None
        sr = 48000

    # FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = sr
    STEP_SIZE = 20.
    CHUNK_SIZE = int(STEP_SIZE * RATE / 1000 + 0.5)
    BUFFER_SIZE = CHUNK_SIZE
    N_BUF = np.ceil(BUFFER_SIZE / CHUNK_SIZE)
    XLIM = int(len(audio) / CHUNK_SIZE + 0.5 if audio is not None else 3 * 100 * STEP_SIZE)

    device = "cpu"
    pesto_model = partial(load_model, "mir-1k_g7_conf",  #"/home/alain/code/pesto/logs/HCQT/checkpoints/HCQT-0227cc80/last.ckpt",
                          step_size=STEP_SIZE,
                          sampling_rate=RATE,
                          streaming=True,
                          mirror=1.)

    if True:
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(())

        pesto_model.confidence = Dummy()

    for wav_file in sys.argv[1:]:
        print(wav_file)
        evaluate_stream(pesto_model, wav_file)
