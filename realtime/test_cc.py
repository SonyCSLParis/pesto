import os
import sys
sys.path.append('/opt/homebrew/opt/ffmpeg@6/lib')
import torch
import torchaudio
import time
import numpy as np

# import pyaudio
# from pythonosc import udp_client


from pesto import load_model


if __name__ == "__main__":
    # FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 48000
    STEP_SIZE = 10.
    CHUNK_SIZE = int(STEP_SIZE * RATE / 1000 + 0.5)
    BUFFER_SIZE = CHUNK_SIZE
    N_BUF = np.ceil(BUFFER_SIZE / CHUNK_SIZE)

    device = "cpu"
    pesto_model = load_model("mir-1k_g5",
                             step_size=STEP_SIZE,
                             sampling_rate=RATE,
                             streaming=True,
                             mirror=0.).to(device)

    # p = pyaudio.PyAudio()
    buffer = bytearray(BUFFER_SIZE)

    # devices = p.get_default_input_device_info()
    # print(devices)
    # stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    # client = udp_client.SimpleUDPClient("127.0.0.1", 3333)

    with torch.inference_mode():
        print('Recording...')
        start = time.time()
        i = 0
        while True:
            # print(stream.get_read_available())
            # while stream.get_read_available() > CHUNKSIZE:
            #     chunk = stream.read(CHUNKSIZE,exception_on_overflow=False)
            #     if len(buffers) == N_BUF:
            #         buffers.pop(0)
            #
            #     #print(chunk)
            #
            #
            #     #compute energy of signal
            #     buffers.append(chunk)
            #     so_far = 0
            chunk = os.urandom(CHUNK_SIZE)

            buffer[:] = chunk

            #amplitude of last_buffer
            amp = 0
            # to f32
            fa = np.frombuffer(chunk, dtype='float32')
            #print(fa)

            # for k in range(len(fa)):
            #     val = fa[k]
            #     amp += val*val

            #tbuffer = torch.frombuffer(buffer,dtype=torch.int16).type(torch.float32)
            tbuffer = torch.frombuffer(buffer, dtype=torch.uint8).to(torch.float32)
            tbuffer.div_(256).sub_(0.5)
            # print(buffer, tbuffer.shape)
            #predictions,confidence,activations = pesto_model(buffer,RATE)
            tbuffer = tbuffer.to(device)
            #a,b = pesto_model(tbuffer,RATE)
            vol, f0, conf = pesto_model(tbuffer, convert_to_freq=True, return_activations=False)

            # log frequencies and speed in FPS
            i += 1
            if i % 1 == 0:
                print(*[f'{s:.3f}' for s in (vol.item(), f0.item(), i / (time.time() - start))], sep='   ')

            if i == 500:
                end = time.time()
                print(end - start)
                break
