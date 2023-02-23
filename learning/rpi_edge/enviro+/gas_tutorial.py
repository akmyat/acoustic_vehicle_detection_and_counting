#!/usr/bin/env python3

import time
from enviroplus import gas

try:
    while True:
        readings = gas.read_all()
        print("\t-------- Gas --------\n")
        print("Oxidised: ", readings.oxidising / 1000)
        print("Reduced:", readings.reducing / 1000)
        print("NH3:", readings.nh3 / 1000)
        time.sleep(1.0)
except KeyboardInterrupt:
    pass