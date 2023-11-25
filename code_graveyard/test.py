# pylint: skip-file
import logging
import time
logging.basicConfig(level=logging.DEBUG)

import sys
from coordinates import CoordinateConversion, CoordinateConversionAstroPy
import datetime

astropy = CoordinateConversionAstroPy()
my = CoordinateConversion()
utcnow = datetime.datetime.utcnow()

for engine, converter in [('AstroPy', astropy), ('my', my)]:
    converter.setEarthLocation(37.37667, -122.03277, 30.0) # Sunnyvale, CA

for alt, az in [(0, 0), (15, 0), (15, 27), (45, 34), (52, 231), (60, 192), (87, 339), (89.999, 359.999)]:
    for engine, converter in [('AstroPy', astropy), ('my', my)]:
        print(
            'Engine {} maps horizontal ({}, {}) to equatorial {}'.format(
                engine, alt, az, converter.horizontalToEquatorial(
                    alt, az, utcnow, refraction=True
                )[:2]
            ), file=sys.stderr, flush=True
        )
