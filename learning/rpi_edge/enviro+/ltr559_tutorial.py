# !/usr/bin/env python3

import time
from ltr559 import LTR559

ltr559 = LTR559()

try:
    while True:
        lux = ltr559.get_lux()
        prox = ltr559.get_proximity()

        print("Lux:\t", lux)
        print("Proximity:\t", prox)
        print("\n")

        time.sleep(1)
except KeyboardInterrupt:
    pass