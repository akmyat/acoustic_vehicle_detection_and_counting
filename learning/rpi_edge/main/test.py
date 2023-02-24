from AVrecorder import AVrecorder
import time

sample_rate = 48000
channels = 2
frames_per_buffer = 1024
duration = 20
audio_input_device = 13

fps = 30
width = 640
height = 480
fourcc = "MJPG"
video_input_device = 0

avrecorder = AVrecorder(sample_rate, channels, frames_per_buffer, duration, audio_input_device, fps, width, height, fourcc, video_input_device)

avrecorder.start_audio_stream()
avrecorder.start_video_stream()
start_time = time.time()

try:
    while True:
        if time.time() - start_time > 20:

            avrecorder.update_timestamp()
            avrecorder.prepare_new_audio_file()
            avrecorder.prepare_new_video_file()

            start_time = time.time()
        else:
            avrecorder.read_audio_frame()
            avrecorder.read_video_frame()
except KeyboardInterrupt:
    avrecorder.stop_audio_stream()
    avrecorder.stop_video_stream()
