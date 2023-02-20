#!/usr/bin/python3
import sys
import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

encoder = H264Encoder(10000000)

output_filename = sys.argv[1]
duration = sys.argv[2]

picam2.start_recording(encoder, output_filename)
time.sleep(duration)
picam2.stop_recording()