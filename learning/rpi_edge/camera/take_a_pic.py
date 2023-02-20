from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
# picam2.start_preview(Preview.QTGL)

# Option 1
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")

# Option 2
picam2.start_and_capture_file("test.jpg")