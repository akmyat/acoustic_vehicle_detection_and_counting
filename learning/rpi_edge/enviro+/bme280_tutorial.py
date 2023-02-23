#!/usr/bin/env python3

import time
from bme280 import BME280
from smbus import SMBus

bus = SMBus(1)
bme280 = BME280(i2c_dev=bus)

while True:
    temperature = bme280.get_temperature()
    pressure = bme280.get_pressure()
    humidity = bme280.get_humidity()

    print("Temp:\t", temperature)
    print("Pressure:\t", pressure)
    print("Humidity:\t", humidity)

    time.sleep(1)