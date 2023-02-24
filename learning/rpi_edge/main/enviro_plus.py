#!/usr/bin/env python3

import time
from bme280 import BME280
from smbus import SMBus
from ltr559 import LTR559
from gas_sensor import MIC6814

class EnviroPlus:
    def __init__(self):
        self.bus = SMBus(1)
        self.bme280 = BME280(i2c_dev=bus)
        self.ltr559 = LTR559()
        self.gas_sensor = MIC6814()
    
    def measure_temperature(self):
        return self.bme280.get_temperature()

    def measure_pressure(self):
        return self.bme280.get_pressure()

    def measure_humidity(self):
        return self.bme280.get_humidity()
    
    def measure_lux(self):
        return self.ltr559.get_lux()
        
    def get_gas_readings(self):
        return self.gas_sensor.get_readings()
    
    def get_oxidising(self):
        return self.gas_sensor.get_oxidising()
    
    def get_reducing(self):
        return self.gas_sensor.get_reducing()

    def get_nh3(self):
        return self.gas_sensor.get_nh3()        
    def measure_CO(self):
        return self.gas_sensor.measure_CO()
    
    def measure_NO2(self):
        return self.gas_sensor.measure_NO2()
    
    def measure_NH3(self):
        return self.gas_sensor.measure_NH3()

    def measure_C3H8(self):
        return self.gas_sensor.measure_C3H8()

    def measure_C4H10(self):
        return self.gas_sensor.measure_C4H10()

    def measure_CH4(self):
        return self.gas_sensor.measure_CH4()

    def measure_H2(self):
        return self.gas_sensor.measure_H2()

    def measure_C2H50H(self):
        return self.gas_sensor.measure_C2H50H()
