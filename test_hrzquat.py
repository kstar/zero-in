import sys
import datetime
from IPython import embed

import numpy as np
from pyquaternion import Quaternion

from coordinates import CoordinateConversion, GeoLocation, AltAz, TimestampedQuaternion

seed = 17
c = CoordinateConversion()

rng = np.random.RandomState(seed=seed) # pylint: disable=no-member

ut = datetime.datetime.utcnow()
print('Testing horizontal -> quaternion -> horizontal')
for test_i in range(300):
    lat, lon, el = map(
        float,
        rng.rand(3) * np.asarray([180, 360, 2000]) - np.asarray([90, 180, 300])
    )
    c.setEarthLocation(lat, lon, el) # Irrelevant
    frame = c.makeAltAzFrame(ut, refraction=True, temperature=25, humidity=0.3)
    altaz = AltAz(
        alt=float(rng.random(1) * 180 - 90), az=float(rng.random(1) * 360), frame=frame
    )
    altaz_ = c.quaternionToHorizontal(c.horizontalToQuaternion(altaz))
    if abs(altaz.alt - altaz_.alt) >= 1e-3 or abs(altaz.az - altaz_.az) % 360.0 >= 1e-3:
        embed(header=f'(horiz)->quat->(horiz) test {test_i} failed')
        sys.exit(1)

print('Testing quaternion -> horizontal -> quaternion')
for test_i in range(300):
    lat, lon, el = map(
        float,
        rng.rand(3) * np.asarray([180, 360, 2000]) - np.asarray([90, 180, 300])
    )
    c.setEarthLocation(lat, lon, el) # Irrelevant
    q = Quaternion(rng.rand(4)).normalised
    tq = TimestampedQuaternion(ut=ut, q=q)

    tq_ = c.horizontalToQuaternion(c.quaternionToHorizontal(tq))

    # Note: Because of the S^1 worth of roll degeneracy in these
    # quaternions, we can only check the altaz values

    altaz = c.quaternionToHorizontal(tq)
    altaz_ = c.quaternionToHorizontal(tq_)
    if abs(altaz.alt - altaz_.alt) >= 1e-3 or abs(altaz.az - altaz_.az) % 360.0 >= 1e-3:
        embed(header=f'quat->(horiz)->quat->(horiz) test {test_i} failed')
        sys.exit(1)

print('OK!')
sys.exit(0)
