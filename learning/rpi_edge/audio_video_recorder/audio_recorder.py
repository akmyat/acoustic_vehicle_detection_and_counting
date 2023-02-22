import pyaudio
import wave

import time
from datetime import datetime

# Audio
sample_rate = 48000
channels = 2
frames_per_buffer = 1024
duration = 20
audio_input_device = 13

audio_file_num = 0
audio_file_name = "audio_" + str(audio_file_num) + ".wav"
pa  = pyaudio.PyAudio()

wavefile = wave.open(audio_file_name, "wb")
wavefile.setnchannels(channels)
wavefile.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
wavefile.setframerate(sample_rate)


def callback(in_data, frame_count, time_info, status):
    global wavefile
    wavefile.writeframes(in_data)
    return in_data, pyaudio.paContinue


stream = pa.open(
    format=pyaudio.paInt16,
    channels=channels,
    rate=sample_rate,
    input=True,
    frames_per_buffer=frames_per_buffer,
    input_device_index=audio_input_device,
    stream_callback=callback
)

stream.start_stream()

start_time = time.time()

try:
    while True:
        if time.time() - start_time > 20:
            wavefile.close()
            audio_file_name = "audio_" + str(audio_file_num) + ".wav"
            print(audio_file_name)

            wavefile = wave.open(audio_file_name, "wb")
            wavefile.setnchannels(channels)
            wavefile.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wavefile.setframerate(sample_rate)

            start_time = time.time()
            audio_file_num += 1
        time.sleep(0.1)

except KeyboardInterrupt as e:
    wavefile.close()
    stream.close()
    pa.terminate()