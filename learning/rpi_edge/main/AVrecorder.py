import cv2
import wave
import pyaudio
import time
from datetime import datetime

class AVrecorder:
    def __init__(self, sample_rate, channels, fpb, duration, audio_input_device, fps, img_width, img_height, encoding, video_input_device, output_path):
        self.sample_rate = sample_rate
        self.channels = channels
        self.fpb = fpb
        self.duration = duration
        self.audio_input_device = audio_input_device

        self.fps = fps
        self.frame_size = (img_width, img_height)
        self.encoding = cv2.VideoWriter_fourcc(*encoding)
        self.video_input_device = video_input_device

        self.time_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._video_cap = None
        
        self.audio_file = None
        self.video_file = None
        self.output_path = output_path

    def prepare_new_audio_file(self):
        self.audio_file_name = self.output_path + "audio_" + self.time_stamp + ".wav"

        wavefile = wave.open(self.audio_file_name, "wb")
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.sample_rate)
        self.audio_file = wavefile

    def prepare_new_video_file(self):
        self.video_file_name = self.output_path + "video_" + self.time_stamp + ".mp4"

        video_writer = cv2.VideoWriter(self.video_file_name, self.encoding, self.fps, self.frame_size)
        self.video_file = video_writer

    def audio_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.audio_file.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback

    def start_audio_stream(self):
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.fpb,
            input_device_index=self.audio_input_device,
            stream_callback=self.audio_callback()
        )
        self.prepare_new_audio_file()
        self._stream.start_stream()
    
    def stop_audio_stream(self):
        self.audio_file.close()
        self._stream.close()
        self._pa.terminate()

    def start_video_stream(self):
        self._video_cap = cv2.VideoCapture(self.video_input_device)
        self.prepare_new_video_file()

    def stop_video_stream(self):
        self.video_file.release()
        self._video_cap.release()
        cv2.destroyAllWindows()

    def update_timestamp(self):
        self.time_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    def read_video_frame(self):
        ret, frame = self._video_cap.read()
        if ret:
            self.video_file.write(frame)
    
    def read_audio_frame(self):
        time.sleep(0.1)