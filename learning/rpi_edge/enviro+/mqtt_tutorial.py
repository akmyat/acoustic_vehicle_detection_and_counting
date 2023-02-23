# !/usr/bin/env python3

import time
import ssl
import paho.mqtt.client as mqtt

def on_connect(mqttc, obj, flags, rc):
    print("connect rc: " + str(rc))

def on_message(mqttc, obj, msg):
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))

def on_publish(mqttc, obj, mid):
    print("mid: " + str(mid))

def on_subscribe(mqttc, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_log(mqttc, obj, level, string):
    print(string)

topic = "hello"
msg = "hello there"

mqttc = mqtt.Client(clean_session=True)

tls_version = ssl.PROTOCOL_TLSv1_2
cert_required = ssl.CERT_REQUIRED
mqttc.tls_set(ca_certs="./certs/ca.crt", certfile="./certs/client.crt", keyfile="./certs/client.key", cert_reqs=cert_required, tls_version=tls_version)

mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe

host = "aungkaungmyat.engineer"
port = 1883
keepalive = 60
mqttc.connect(host, port, keepalive)

mqttc.loop_start()

start_time = time.time()
try:
    while True:
        if time.time() - start_time > 20:
            print("Publishing... ")
            infot = mqttc.publish(topic, msg)
            infot.wait_for_publish()

            time.sleep(0.1)

            start_time = time.time()
except KeyboardInterrupt as e:
    mqttc.disconnect()