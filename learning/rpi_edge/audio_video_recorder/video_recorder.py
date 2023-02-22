import cv2
import time

# Video
fps = 30
frame_size = (640, 480)
fourcc = "MJPG"
video_input_device = 0

video_file_num = 0
video_file_name = "video_" + str(video_file_num) + ".avi"

video_cap = cv2.VideoCapture(video_input_device)
video_codec = cv2.VideoWriter_fourcc(*fourcc)
video_writer = cv2.VideoWriter(video_file_name, video_codec, fps, frame_size)

start_time = time.time()
try:
    while True:
        ret, frame = video_cap.read()
        if ret:
            video_writer.write(frame)

        if time.time() - start_time > 20:
            print(video_file_name)
            video_writer.release()
            video_file_num += 1
            video_file_name = "video_" + str(video_file_num) + ".avi"
            video_writer = cv2.VideoWriter(video_file_name, video_codec, fps, frame_size)
            start_time = time.time()
        
except KeyboardInterrupt as e:
    video_writer.release()
    video_cap.release()
    cv2.destroyAllWindows()