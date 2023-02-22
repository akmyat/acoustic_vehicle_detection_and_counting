import pyaudio
import wave

import cv2

import time

# ---------------------------------------------------- Audio ---------------------------------------------------- 
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

# ---------------------------------------------------- Video ---------------------------------------------------- 
fps = 30
frame_size = (640, 480)
fourcc = "MJPG"
video_input_device = 0

video_file_num = 0
video_file_name = "video_" + str(video_file_num) + ".avi"

video_cap = cv2.VideoCapture(video_input_device)
video_codec = cv2.VideoWriter_fourcc(*fourcc)
video_writer = cv2.VideoWriter(video_file_name, video_codec, fps, frame_size)

# ---------------------------------------------------- Start Recording ----------------------------------------------------
stream.start_stream()
start_time = time.time()

try:
    while True:
        if time.time() - start_time > 20:

            # Audio save file
            print(audio_file_name)
            audio_file_num += 1
            audio_file_name = "audio_" + str(audio_file_num) + ".wav"

            wavefile = wave.open(audio_file_name, "wb")
            wavefile.setnchannels(channels)
            wavefile.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wavefile.setframerate(sample_rate)

            # Video save file
            print(video_file_name)
            video_file_num += 1
            video_file_name = "video_" + str(video_file_num) + ".avi"
            video_writer = cv2.VideoWriter(video_file_name, video_codec, fps, frame_size)

            start_time = time.time()

        # time.sleep(0.1)
        ret, frame = video_cap.read()
        if ret:
            video_writer.write(frame)        

# ---------------------------------------------------- End Recording ----------------------------------------------------
except KeyboardInterrupt as e:
    wavefile.close()
    stream.close()
    pa.terminate()

    video_writer.release()
    video_cap.release()
    cv2.destroyAllWindows()    