import os
import sys
sys.path.append('/opt/homebrew/opt/ffmpeg@6/lib')
import torch
import torchaudio
import time
import numpy as np

# import cached_conv as cc

# import pyaudio
# from pythonosc import udp_client


from pesto import load_model


if __name__ == "__main__":
    CHUNKSIZE = 480
    # FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 48000
    BUFFER_SIZE = 4096
    N_BUF = int(BUFFER_SIZE/CHUNKSIZE) + 1

    device = "cpu"
    # cc.use_cached_conv(True)
    pesto_model = load_model("vqt-v2", step_size=10., sampling_rate=RATE).to(device)

    # p = pyaudio.PyAudio()
    buffer = bytearray(BUFFER_SIZE)
    buffers = []

    # devices = p.get_default_input_device_info()
    # print(devices)
    # stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    # client = udp_client.SimpleUDPClient("127.0.0.1", 3333)


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
        chunk = os.urandom(CHUNKSIZE)
        if len(buffers) == N_BUF:
            buffers.pop(0)

        buffers.append(chunk)
        # time.sleep(0.1*CHUNKSIZE/RATE)
        so_far = 0

        if len(buffers) == N_BUF:
            for k in range(N_BUF):
                buffer[so_far:so_far+len(buffers[k])]=buffers[k]
                so_far += len(buffers[k])

            #amplitude of last_buffer
            amp = 0
            last = N_BUF-1
            # to f32
            fa = np.frombuffer(buffers[last], dtype='float32')
            #print(fa)

            for k in range(len(fa)):
                val = fa[k]
                amp += val*val


            #we have our N buffers
            #print(chunk)

            #tbuffer = torch.frombuffer(buffer,dtype=torch.int16).type(torch.float32)
            tbuffer = torch.frombuffer(buffer, dtype=torch.uint8).to(torch.float32)
            tbuffer.div_(256).sub_(0.5)
            # print(buffer, tbuffer.shape)
            #predictions,confidence,activations = pesto_model(buffer,RATE)
            tbuffer = tbuffer.to(device)
            #a,b = pesto_model(tbuffer,RATE)
            print(tbuffer.shape)
            f0, conf = pesto_model(tbuffer, convert_to_freq=True, return_activations=False)
            i += 1
            print(f0[-1].item(), i / (time.time() - start))#,conf)
            #print(amp)
            # client.send_message("/note", [f0[len(f0)-1].item(), amp])
            # break
