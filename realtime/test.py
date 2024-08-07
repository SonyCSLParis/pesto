import sys
sys.path.append('/opt/homebrew/opt/ffmpeg@6/lib')
import torch
import torchaudio
import time
import pesto
import numpy as np

import pyaudio
from pythonosc import udp_client


from pesto import load_model



if __name__ == "__main__":
    CHUNKSIZE = 512
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 48000
    N_BUF = int(65536/CHUNKSIZE)+1

    device = "cpu"
    pesto_model = load_model("mir-1k", step_size = 200.,sampling_rate=RATE).to(device)

    p = pyaudio.PyAudio()
    buffer = bytearray(65536)
    buffers = []

    devices = p.get_default_input_device_info()
    print(devices)
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    client = udp_client.SimpleUDPClient("127.0.0.1", 3333)


    print('Recording...')
    while True:
        #print(stream.get_read_available())
        while stream.get_read_available() > CHUNKSIZE:
            chunk = stream.read(CHUNKSIZE,exception_on_overflow=False)
            if len(buffers) == N_BUF:
                buffers.pop(0)

            #print(chunk)


            #compute energy of signal
            buffers.append(chunk)
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
            tbuffer = torch.frombuffer(buffer,dtype=torch.float32)
            #print(buffer)
            #predictions,confidence,activations = pesto_model(buffer,RATE)
            tbuffer = tbuffer.to(device)
            #a,b = pesto_model(tbuffer,RATE)
            f0, conf = pesto_model(tbuffer, convert_to_freq=True, return_activations=False)
            print(f0[len(f0)-1].item())#,conf)
            #print(amp)
            client.send_message("/note", [f0[len(f0)-1].item(), amp])





