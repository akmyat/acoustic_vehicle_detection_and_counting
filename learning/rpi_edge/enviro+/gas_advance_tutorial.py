# !/usr/bin/env python3

import time
import math
from enviroplus import gas

class MIC6814:
    def read_r0(self):
        data = gas.read_all()
        r0 = [0] * 3
        r0[0] = data.oxidising
        r0[1] = data.reducing
        r0[2] = data.nh3
        return r0

    def read_rs(self):
        data = gas.read_all()
        rs = [0] * 3
        rs[0] = data.oxidising
        rs[1] = data.reducing
        rs[2] = data.nh3
        return rs

    def calc_gas(self, gas_type):
        r0 = self.read_r0()
        rs = self.read_rs()

        ratio0 = rs[0] / r0[0]
        if ratio0 < 0:
            ratio0 = 0.0001
        ratio1 = rs[1] / r0[1]
        if ratio1 < 0:
            ratio1 = 0.0001
        ratio2 = rs[2] / r0[2]
        if ratio2 < 0:
            ratio2 = 0.0001

        if gas_type == "CO":
            c = pow(ratio1, -1.179) * 4.385
        if gas_type == "NO2":
            c = pow(ratio2, 1.007) / 6.855
        if gas_type == "NH3":
            c = pow(ratio0, -1.67) / 1.47
        if gas_type == "C3H8":
            c = pow(ratio0, -2.518) * 570.164
        if gas_type == "C4H10":
            c = pow(ratio0, -2.138) * 398.107
        if gas_type == "CH4":
            c = pow(ratio1, -4.363) * 630.957
        if gas_type == "H2":
            c = pow(ratio1, -1.8) * 0.73
        if gas_type == "C2H50H":
            c = pow(ratio1, -1.552) * 1.622

        return -3 if math.isnan(c) else c

    def measure_CO(self):
        return self.calc_gas("CO")
    
    def measure_NO2(self):
        return self.calc_gas("NO2")
    
    def measure_NH3(self):
        return self.calc_gas("NH3")
    
    def measure_C3H8(self):
        return self.calc_gas("C3H8")
    
    def measure_C4H10(self):
        return self.calc_gas("C4H10")
    
    def measure_CH4(self):
        return self.calc_gas("CH4")
    
    def measure_H2(self):
        return self.calc_gas("H2")
    
    def measure_C2H50H(self):
        return self.calc_gas("C2H50H")


gas_sensor = MIC6814()
print("CO:", gas_sensor.measure_CO())
print("NO2:", gas_sensor.measure_NO2())
print("NH3:", gas_sensor.measure_NH3())
print("C3H8:", gas_sensor.measure_C3H8())
print("C4H10:", gas_sensor.measure_C4H10())
print("CH4:", gas_sensor.measure_CH4())
print("H2:", gas_sensor.measure_H2())
print("C2H50H:", gas_sensor.measure_C2H50H())