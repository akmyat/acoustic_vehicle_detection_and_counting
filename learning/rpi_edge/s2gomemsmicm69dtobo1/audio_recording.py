import time
from datetime import datetime

import wave
import pyaudio

sample_rate = 48000
win_length = 1024
hop_length = 512
duration = 20
channels = 2

frames = []

p = pyaudio.PyAudio()

fps = int(sample_rate / win_length * duration)

def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return in_data, pyaudio.paContinue

stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=win_length,
        stream_callback=callback,
        input_device_index=13,
        )

stream.start_stream()

try:
    while True:
        if len(frames) > fps:
            clip = []
            for i in range(0, fps):
                clip.append(frames[i])
            fname = "".join(["./", "clip-", datetime.utcnow().strftime("%Y%m%d%H%M%S"), ".wav"])
            print(fname)
            
            wavefile = wave.open(fname, "wb")
            wavefile.setnchannels(channels)
            wavefile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wavefile.setframerate(sample_rate)

            wavefile.writeframes(b''.join(clip))
            wavefile.close()

            frames = frames[(duration - hop_length -1):]
except KeyboardInterrupt as e:
    stream.stop_stream()
    print("Ending")