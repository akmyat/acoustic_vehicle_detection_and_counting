# !/usr/bin/env python3

import time
import ssl
import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, tls_version, ca_crt_path, client_crt_path, client_key_path, log=False):
        self.mqttc = mqtt.Client(clean_session=True)

        if tls_version == "tlsv1":
            self.tls_version = ssl.PROTOCOL_TLSv1
        elif tls_version == "tlsv1.1":
            self.tls_version = ssl.PROTOCOL_TLSv1_1
        elif tls_version == "tlsv1.2":
            self.tls_version = ssl.PROTOCOL_TLSv1_2
        else:
            print("Unknown TLS version - ignoring")
            self.tls_version = None
        self.cert_required = ssl.CERT_REQUIRED

        self.mqttc.tls_set(
            ca_certs=ca_crt_path, 
            certfile=client_crt_path, 
            keyfile=client_key_path, 
            cert_reqs=self.cert_required, 
            tls_version=self.tls_version)

    def connect(self, host: str, port: int, keepalive: int):
        self.mqttc.connect(host, port, keepalive)

    def start_loop(self):
        self.mqttc.loop_start()

    def stop_loop(self):
        self.mqttc.loop_stop()
    
    def publish(self, topic: str, msg: str, qos: int = 0):
        infot = self.mqttc.publish(topic, msg, qos)
        infot.wait_for_publish()
        time.sleep(0.1)

    def subscribe(self, topic: str, qos: int = 0):
        self.mqttc.subscribe(topic, qos)

    def unsubscribe(self, topic):
        self.mqttc.unsubscribe(topic)