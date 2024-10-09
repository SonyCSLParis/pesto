import datetime
import os
import sys
from functools import partial
sys.path.append('/opt/homebrew/opt/ffmpeg@6/lib')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torchaudio
import time
import numpy as np

# import pyaudio
# from pythonosc import udp_client


from pesto import load_model

@torch.inference_mode()
def iterator(audio=None):
    t0 = time.time()
    i = 0
    buffer = bytearray(BUFFER_SIZE)

    t_limit = audio.size(-1) // CHUNK_SIZE if audio is not None else float("inf")
    while i < t_limit:
        # chunk = os.urandom(CHUNK_SIZE)
        #
        # buffer[:] = chunk

        tbuffer = audio[i * CHUNK_SIZE: (i+1) * CHUNK_SIZE] if audio is not None else torch.randn(CHUNK_SIZE)
        # tbuffer.div_(256).sub_(0.5)
        tbuffer = tbuffer.to(device)

        if i % 100 == 0:
            print(f"{i / (time.time() - t0):.2f} FPS")

        yield i, pesto_model(tbuffer, convert_to_freq=False, return_activations=False)
        i += 1


def update_ani(data):
    i, (f0, conf, vol) = data
    # scat.set_offsets([i % XLIM, f0.item()])
    # scat.set_sizes([vol.clip(min=0).item()])
    x_data.append(i % XLIM)
    y_data.append(f0.item())
    c_data.append(conf.item())
    print(c_data[-1])
    s_data.append(vol.clip(min=0).item())
    scat.set_offsets(list(zip(x_data, y_data)))
    scat.set_sizes(s_data)
    scat.set_array(np.array(c_data) / max(max(c_data), 1e-8))


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
    RATE = 48000
    STEP_SIZE = 20.
    CHUNK_SIZE = int(STEP_SIZE * RATE / 1000 + 0.5)
    BUFFER_SIZE = CHUNK_SIZE
    N_BUF = np.ceil(BUFFER_SIZE / CHUNK_SIZE)
    XLIM = int(len(audio) / CHUNK_SIZE + 0.5 if audio is not None else 3 * 100 * STEP_SIZE)

    device = "cpu"
    pesto_model = load_model("/home/alain/code/pesto/logs/HCQT/checkpoints/HCQT-0227cc80/last.ckpt",
                             step_size=STEP_SIZE,
                             sampling_rate=RATE,
                             streaming=True,
                             mirror=1.).to(device)

    fig, ax = plt.subplots(figsize=(20, 4))
    plt.tight_layout()
    scat = ax.scatter([], [], cmap='inferno')

    ax.set_xlim(0, XLIM)
    ax.set_ylim(20, 100)

    x_data = []
    y_data = []
    c_data = []
    s_data = []

    try:
        ani = animation.FuncAnimation(fig,
                                      update_ani,
                                      frames=partial(iterator, audio=audio),
                                      interval=1,
                                      cache_frame_data=False)
        plt.colorbar(scat, ax=ax)
        plt.show()
    finally:
        outfile = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_trace.pdf"
        print(f"Saving trace in {outfile}...")
        plt.figure(figsize=(20, 4))
        plt.scatter(x_data, y_data, s=s_data, c=np.array(s_data) / max(s_data), cmap='viridis')
        plt.ylim(20, 100)
        plt.tight_layout()
        plt.title(outfile.split(".")[0])
        plt.savefig(outfile)

        plt.clf()
        plt.hist([s for s in s_data if s > 0], bins=200)
        plt.savefig(outfile.replace("trace", "hist"))


        print("Done.")

    # start = time.time()
    # for i, (vol, f0, conf) in iterator():
    #         # log frequencies and speed in FPS
    #         if i % 1 == 0:
    #             print(*[f'{s:.3f}' for s in (vol.item(), f0.item(), i / (time.time() - start))], sep='   ')
    #         if i == 500:
    #             end = time.time()
    #             print(end - start)
    #             break
