# !/usr/bin/env python3

import time
from mqtt_client import MQTTClient              # For MQTT
from enviro_plus import EnviroPlus                 # For Enviro+ board
from AVrecorder import AVrecorder               # For Audio and Video recorder

debug = True

deviceID = "VCD0001-0000000018e10413"
prefix = "dt"

# For MQTT
tls_version = "tlsv1.2"
ca_crt_path = "./certs/ca.crt"
client_crt_path = "./certs/client.crt"
client_key_path = "./certs/client.key"

host = "aungkaungmyat.engineer"
port = 1883
keepalive = 60

mqtt_client = MQTTClient(tls_version, ca_crt_path, client_crt_path, client_key_path)
mqtt_client.connect(host, port, keepalive)

# For Enviro+
enviro_sensor = EnviroPlus()

topic_base = "/" + prefix + "/" + deviceID + "/"
topic_dict = {
    "temperature": topic_base + "3303/0/5700",
    "pressure": topic_base + "3315/0/5700",
    "humidity": topic_base + "3304/0/5700",
    "illuminance": topic_base + "3301/0/5700",
    "oxidising": topic_base + "3300/0/5700",
    "reducing": topic_base + "3300/1/5700",
    "nh3": topic_base + "3300/2/5700",
}

def read_sensor_values():
    temperature = enviro_sensor.measure_temperature()
    pressure = enviro_sensor.measure_pressure()
    humidity = enviro_sensor.measure_humidity()
    illuminance = enviro_sensor.measure_lux()
    oxidising = enviro_sensor.get_oxidising()
    reducing = enviro_sensor.get_reducing()
    nh3 = enviro_sensor.get_nh3()

    value_dict = {
        "temperature": str(temperature),
        "pressure": str(pressure),
        "humidity": str(humidity),
        "illuminance": str(illuminance),
        "oxidising": str(oxidising),
        "reducing": str(reducing),
        "nh3": str(nh3),        
    }

    return value_dict

# For Audio
sample_rate = 48000
channels = 2
frames_per_buffer = 1024
duration = 20
audio_input_device = 1

# For Video
fps = 6
width = 640
height = 480
fourcc = "MJPG"
video_input_device = 0

avrecorder = AVrecorder(sample_rate, channels, frames_per_buffer, duration, audio_input_device, fps, width, height, fourcc, video_input_device)

avrecorder.start_audio_stream()
avrecorder.start_video_stream()
mqtt_client.start_loop()

if debug:
    frame_num = 0

start_time = time.time()
print("Start Recording...")
try:
    while True:
        if time.time() - start_time > 20:
            if debug:
                print(frame_num)
                frame_num = 0

            avrecorder.update_timestamp()
            avrecorder.prepare_new_audio_file()
            avrecorder.prepare_new_video_file()
            print("Write Audio and Video files")

            value_dict = read_sensor_values()
            for key in value_dict.keys():
                mqtt_client.publish(topic_dict[key], value_dict[key], qos=1)
            print("Send data via MQTT")

            start_time = time.time()
        else:
            avrecorder.read_audio_frame()
            avrecorder.read_video_frame()
            if debug:
                frame_num += 1

except KeyboardInterrupt:
    avrecorder.stop_audio_stream()
    avrecorder.stop_video_stream()
    mqtt_client.stop_loop()
    print("Stop Recording...")
