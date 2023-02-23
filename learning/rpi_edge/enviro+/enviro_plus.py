#!/usr/bin/env python3

import time
from bme280 import BME280
from smbus import SMBus
from ltr559 import LTR559
from gas_advance_tutorial import MIC6814

class enviroSensor:
    def __init__():
        self.bus = SMBus(1)
        self.bme280 = BME280(i2c_dev=bus)
        self.ltr559 = LTR559()
        self.gas_sensor = MIC6814()
    
    def measure_temperature():
        return self.bme280.get_temperature()

    def measure_pressure():
        return self.bme280.get_pressure()

    def measure_humidity():
        return self.bme280.get_humidity()
    
    def measure_lux():
        return self.ltr559.get_lux()
        
    def measure_CO():
        return self.gas_sensor.measure_CO()
    
    def measure_NO2():
        return self.gas_sensor.measure_NO2()
    
    def measure_NH3():
        return self.gas_sensor.measure_NH3()

    def measure_C3H8():
        return self.gas_sensor.measure_C3H8()

    def measure_C4H10():
        return self.gas_sensor.measure_C4H10()

    def measure_CH4():
        return self.gas_sensor.measure_CH4()

    def measure_H2():
        return self.gas_sensor.measure_H2()

    def measure_C2H50H():
        return self.gas_sensor.measure_C2H50H()