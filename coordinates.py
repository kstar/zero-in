"""Conversion to Horizontal Coordinates"""

import logging
logger = logging.getLogger('CoordinateConversion')

import datetime
import math
import collections

import julian
from pyquaternion import Quaternion


AltAz = collections.namedtuple("AltAz", ["alt", "az", "frame"])

AltAz.__doc__ = """
An immutable tuple of altitude (in degrees), azimuth (in degrees) and an AltAzFrame object
Azimuth is referenced to have North = 0
"""

ICRS = collections.namedtuple("ICRS", ["ra", "dec"])

ICRS.__doc__ = """
An immutable tuple of ICRS (ra, dec) coordinates. They are referred to epoch J2000.0 and are considered "absolute" for the purposes of this program.
"""

EquatorialCoordinates = collections.namedtuple(
    "EquatorialCoordinates",
    ["ra", "dec", "epoch_jd"]
)
EquatorialCoordinates.__doc__ = """
An immutable tuple of equatorial (ra, dec) coordinates. They are referred to the Julian epoch given by the Julian Day epoch_jd.
"""

GeoLocation = collections.namedtuple("GeoLocation", ["latitude", "longitude", "elevation"])

GeoLocation.__doc__ = """
An immutable tuple describing a location on earth. latitude and longitude are in degrees, and the elevation is in meters.
Longitudes are negative west, latitudes are negative south.
"""

AltAzFrame = collections.namedtuple("AltAzFrame", [
    "UT",
    "LST",
    "JD",
    "refraction_enabled",
    "location",
    "pressure",
    "temperature",
    "relative_humidity",
])

AltAzFrame.__doc__ = """
An immutable tuple containing all the required elements (with redundancy for convenience) describing the state needed to go between ICRS and Alt/Az coordinates.
Use CoordinateConversion.makeAltAzFrame() to make one
"""

class LiteAltAzFrame:
    _LST = None
    _lat = None

    def __init__(self, frame: AltAzFrame):
        self._LST = frame.LST
        self._lat = frame.location.latitude

    @property
    def LST(self):
        return self._LST

    @property
    def lat(self):
        return self._lat

TimestampedICRS = collections.namedtuple(
    "TimestampedICRS",
    ["ut", "icrs"],
)

TimestampedQuaternion = collections.namedtuple(
    "TimestampedQuaternion",
    ["ut", "q"]
)
TimestampedQuaternion.__doc__ = """
An immutable tuple containing a quaternion along with a timestamp indicating time of measurement
"""

EulerAngles = collections.namedtuple(
    "EulerAngles",
    ["roll", "pitch", "yaw"]
)
EulerAngles.__doc__ = """
An immutable tuple containing an Euler angle triad (convention: in degrees)
"""

TimestampedEulerAngles = collections.namedtuple(
    "TimestampedEulerAngles",
    ["ut", "e"]
)
TimestampedEulerAngles.__doc__ = """
An immutable tuple containing an EulerAngles along with a (UT) timestamp indicating time of measurement
"""


class CoordinateConversion:
    """Convert ICRS RA/Dec to Alt/Az or vice versa.

    AstroPy is super slow even though it might be ultra-accurate. We
    just want something that finishes very quickly and is sufficiently
    accurate for our amateur purposes.
    """

    def __init__(self):
        self._earth_location = None

    def setEarthLocation(self, latitude, longitude, elevation):

        """
        latitude: in decimal degrees
        longitude: in decimal degrees
        elevation: in meters
        """

        assert type(latitude) is float
        assert type(longitude) is float
        assert type(elevation) in (int, float)

        if self._earth_location:
            logger.info('Overwriting old earth location with new one')

        self._earth_location = GeoLocation(latitude=latitude, longitude=longitude, elevation=elevation)

        # Standard atmospheric model to determine pressure
        # from https://www.grc.nasa.gov/WWW/K-12/airplane/atmosmet.html

        T = 15.04 - 0.00649 * elevation # temperature in °C
        p = 101.29 * (((T + 273.1)/288.08) ** 5.256) # pressure in kPa
        logger.info(
            'Atmospheric pressure at height = {}m is {:.2f}kPa'.format(
                elevation, p
            )
        )
        self._pressure = p

    def precessFromJ2000(self, ra0, dec0, jd):
        """
        Takes ra0, dec0 in degrees; precesses to JD = jd; returns ra, dec in degrees
        N.B. We do not implement nutation
        """
        j2000 = 2451545.0

        # Meeus Chapter 21
        t = (jd - j2000)/36525 # Julian centuries

        t2 = t * t
        t3 = t * t * t

        # Correction angles in arcseconds
        zeta = 2306.2181 * t + 0.30188 * t2 + 0.017998 * t3
        z = 2306.2181 * t + 1.09468 * t2 + 0.018203 * t3
        theta = 2004.3109 * t - 0.42665 * t2 - 0.041833 * t3

        # Convert all angles to radian
        ra0 = math.radians(ra0)
        dec0 = math.radians(dec0)
        zeta = math.radians(zeta/3600.0)
        theta = math.radians(theta/3600.0)
        z = math.radians(z/3600.0)

        cosdec0 = math.cos(dec0)
        sindec0 = math.sin(dec0)
        sintheta = math.sin(theta)
        costheta = math.cos(theta)
        P = math.cos(ra0 + zeta) * cosdec0

        A = cosdec0 * math.sin(ra0 + zeta)
        B = costheta * P - sintheta * sindec0
        C = sintheta * P + costheta * sindec0

        ra = z + math.atan2(A, B)
        dec = math.asin(C)
        if (abs(dec) > 89.0):
            # Meeus suggests numerically better alternative
            dec = math.acos(math.sqrt(A**2 + B**2))

        # DEBUG:
        # logger.info(
        #     'Precessing J2000 coordinates ({}, {}) to current epoch JD={} yields ({}, {})'.format(ra0, dec0, jd, math.degrees(ra), math.degrees(dec))
        # )

        return (math.degrees(ra), math.degrees(dec))

    def precessToJ2000(self, ra, dec, jd):
        """
        Inputs and outputs in degrees
        """
        j2000 = 2451545.0

        ra, dec = math.radians(ra), math.radians(dec)

        # Put t0 = -T in Meeus' (21.2)
        T = (jd - j2000)/36525 # Julian centuries
        t = -T

        T2 = T * T
        T3 = T2 * T
        t2 = T2
        t3 = -T3

        zeta = (2306.2181 + 1.39656 * T - 0.000139 * T2) * t + (0.30188 - 0.000344 * T) * t2 + 0.017998 * t3
        z = (2306.2181 + 1.39656 * T - 0.000139 * T2) * t + (1.09468 + 0.000066 * T) * t2 + 0.018203 * t3
        theta = (2004.3109 - 0.85330 * T - 0.000217 * T2) * t - (0.42665 + 0.000217 * T) * t2 - 0.041833 * t3

        zeta = math.radians(zeta/3600.0)
        z = math.radians(z/3600.0)
        theta = math.radians(theta/3600.0)

        cosdec = math.cos(dec)
        sindec = math.sin(dec)
        sintheta = math.sin(theta)
        costheta = math.cos(theta)
        P = math.cos(ra + zeta) * cosdec

        A = cosdec * math.sin(ra + zeta)
        B = costheta * P - sintheta * sindec
        C = sintheta * P + costheta * sindec

        ra0 = z + math.atan2(A, B)
        dec0 = math.asin(C)
        if (abs(dec0) > 89.0):
            # Meeus suggests numerically better alternative
            dec0 = math.acos(math.sqrt(A**2 + B**2))

        return (math.degrees(ra0), math.degrees(dec0))

    def ICRSToEquatorial(self, icrs: ICRS, jd: float):
        ra, dec = self.precessFromJ2000(icrs.ra, icrs.dec, jd)
        return EquatorialCoordinates(
            ra=ra,
            dec=dec,
            epoch_jd=jd
        )

    def EquatorialToICRS(self, eq: EquatorialCoordinates):
        ra, dec = self.precessToJ2000(eq.ra, eq.dec, eq.epoch_jd)
        return ICRS(ra=ra, dec=dec)

    def refract(self, altitude, altaz_frame):

        if (altitude < 0):
            return altitude

        # Formula due to G. G. Bennett / Saemundsson, Chapter 16 of Meeus
        # Note that this formula takes altitude in _degrees_ and returns the result in arcminutes
        R = 1.02 / math.tan(math.radians(altitude + 10.3/(altitude + 5.11)))

        # Pressure in kPa, temperature in °C
        pressure, temperature = altaz_frame.pressure, altaz_frame.temperature
        correction = (pressure/101.0) * (283 / (273 + temperature)) * R # arcminutes

        # DEBUG
        # logger.info(
        #     'Refraction correction at {:.2f} degrees altitude for p = {:.2f} kPa and T = {:.2f} °C is {:.2f} arcmin'.format(
        #         altitude, pressure, temperature, correction
        #     )
        # )

        alt_refracted = altitude + correction/60.0

        return alt_refracted # degrees

    def unrefract(self, alt_apparent, altaz_frame):

        if (alt_apparent < 0):
            return alt_apparent

        # Formula due to G. G. Bennett / Saemundsson, Chapter 16 of Meeus
        # Note that this formula takes altitude in _degrees_ and returns the result in arcminutes
        R = 1.0 / math.tan(math.radians(alt_apparent + 7.31/(alt_apparent + 4.4)))

        # Pressure in kPa, temperature in °C
        pressure, temperature = altaz_frame.pressure, altaz_frame.temperature
        correction = (pressure/101.0) * (283 / (273 + temperature)) * R # arcminutes

        # DEBUG
        # logger.info(
        #     'Refraction correction at {:.2f} degrees altitude for p = {:.2f} kPa and T = {:.2f} °C is {:.2f} arcmin'.format(
        #         altitude, pressure, temperature, correction
        #     )
        # )

        altitude = alt_apparent - correction/60.0

        return altitude # degrees

    def quaternionToEuler(self, tq: TimestampedQuaternion) -> TimestampedEulerAngles:
        """
        tq: TimestampedQuaternion
        Returns TimestampedEulerAngles
        """
        
        # From https://answers.unity.com/questions/416169/finding-pitchrollyaw-from-quaternions.html
        w, x, y, z = tq.q
        e = EulerAngles(
            roll=math.degrees(math.atan2(2 * y * w - 2 * x * z, 1.0 - 2 * y * y - 2 * z * z)),
            pitch=math.degrees(math.atan2(2 * x * w - 2 * y * z, 1 - 2 * x * x - 2 * z * z)),
            yaw=math.degrees(math.asin(2 * x * y + 2 * z * w))
        )
        return TimestampedEulerAngles(
            ut=tq.ut,
            e=e,
        )

    def horizontalToVector(self, altaz: AltAz) -> TimestampedQuaternion:
        assert isinstance(altaz, AltAz)
        h, A = math.radians(altaz.alt), math.radians(altaz.az)
        cos_h, sin_h = math.cos(h), math.sin(h)
        cos_A, sin_A = math.cos(A), math.sin(A)
        return TimestampedQuaternion(
            ut=altaz.frame.UT,
            q=Quaternion(
                0,
                cos_h * cos_A,
                -cos_h * sin_A,
                sin_h,
            )
        )

    def vectorToHorizontal(self, tvec: TimestampedQuaternion, refraction=True, temperature=25, humidity=0.3):
        """
        vector: pyquaternion.Quaternion with 0th component = 0, representing the scope vector
        frame: An AltAzFrame object for the correct timestamp of vector; use makeAltAzFrame to create one
        """
        assert isinstance(tvec, TimestampedQuaternion)
        vector = tvec.q
        A = math.atan2(-vector[2], vector[1]) # -v_y / v_x
        h = math.atan2(vector[3], math.sqrt(vector[1] ** 2 + vector[2] ** 2)) # v_z / sqrt(v_x² + v_y²)
        frame = self.makeAltAzFrame(
            tvec.ut, refraction=refraction, temperature=temperature, humidity=humidity
        )
        return AltAz(
            alt=math.degrees(h),
            az=math.degrees(A),
            frame=frame,
        )

    def quaternionToHorizontal(self, tq, refraction=True, temperature=25, humidity=0.3):
        """
        [DEPRECATED]
        tq: TimestampedQuaternion

        Returns an AltAz object
        """
        raise DeprecationWarning('This method is deprecated; you should avoid calling it')
        assert isinstance(tq, TimestampedQuaternion)
        frame = self.makeAltAzFrame(
            tq.ut, refraction=refraction, temperature=temperature, humidity=humidity
        )

        q = tq.q
        # Convert quaternion to alt-az
        v = q * Quaternion(0, 0, 0, 1) * q.inverse # vector

        ### MINIMUM ERROR ESTIMATE ###
        # FIXME: This may need to be fixed with the corrected math

        # By solving the minimization problem
        #
        # h = argmin [ v_x^2 + v_y^2 - cos^2 h ]^2 + [ v_z^2 - sin^2 h]^2
        #
        # one obtains
        #
        # h = 0.5 * arccos(v_x^2 + v_y^2 - v_z^2) * sgn(v_z)

        # alt = 0.5 * math.acos(
        #     v[1] * v[1] + v[2] * v[2] - v[3] * v[3]
        # )
        # if v[3] < 0:
        #     alt = -alt
        #

        ### NAÏVE METHOD ###

        # We work in the coordinate system where i points towards
        # north cardinal point, j towards west cardinal point, and k
        # towards the zenith

        # The quaternion is defined to be the one that takes k to the
        # "scope" vector:
        # v = 0 + i sin(z) cos(A) -j sin(z) sin(A) + k cos(z)
        #
        # where z is the zenithal distance z = pi/2 - h, i.e.
        # v = 0 + i cos(h) cos(A) -j cos(h) sin(A) + k sin(h)
        #
        #
        # Therefore, we can estimate h as asin(v[3]) except when v[3]
        # is close to 1, in which case
        #
        # acos(sqrt(v[1] ** 2 + v[2] ** 2))
        #
        # provides a better estimate

        # Assume norm of the vector is 1.0 (true up to numerical error)
        alt = math.asin(v[3])

        if abs(alt) >= 88.0:
            zdist = math.acos(math.sqrt(v[2] * v[2] + v[1] * v[1]))
            alt_aliter = math.pi / 2.0 - zdist
            if alt < 0:
                alt = -alt_aliter
            else:
                alt = alt_aliter

        # Similarly, we see we can get A from atan2(-v[2], v[1]),
        # noting that cos(h) > 0 for all sensible h

        az = math.atan2(
            -v[2],
            v[1],
        )

        alt = math.degrees(alt)
        az = math.degrees(az)


        return AltAz(
            alt=alt,
            az=az,
            frame=frame,
        )

    def horizontalToQuaternion(self, altaz):
        """
        [DEPRECATED] Take an AltAz object and return a TimestampedQuaternion
        """
        raise DeprecationWarning('This method is deprecated; you should avoid calling it')

        # The earth reference frame shall have the X-axis pointing to
        # the north cardinal point, and the Z-axis pointing to the
        # zenith; the Y-axis is then determined by the right hand rule
        # (i.e. points to the west cardinal point).

        # The given attitude quaternion is defined to be a quaternion
        # that will rotate the vector k into the vector pointing in
        # the direction of (alt [refracted], az)
        #
        # That is, the quaternion q satisfies:
        # q k q^-1 = sin(alt) k + cos(alt) * cos(-az) i + cos(alt) * sin(-az) j
        #
        # Writing down a general parametrization of a unit quaternion as
        #
        # q = cos(µ)cos(ν) + i sin(µ)cos(ρ) + j sin(µ)sin(ρ) + k cos(µ)sin(ν),
        #
        # we obtain the relations:
        #
        # µ = z / 2
        # ν + ρ = π/2 - A
        #
        # where `z` is the zenithal distance, i.e. z = π/2 - h. This
        # fixes µ, but does not fix the other two parameters.
        #
        # One can show that a quaternion with minimial rotation
        # (i.e. max real part) that satisfies this is given by:
        #
        # q = cos(mu) + sin(mu) * [ cos(az) i + sin(az) j ]
        # where mu := (alt - pi/2)/2
        #
        # but THIS IS NOT THE QUATERNION WE SEEK
        #
        # We want the quaternion with NO ROLL.
        #
        # Let us imagine a horrible telescope where the focuser points
        # downwards. Let us say that the "parked position" of the
        # telescope is achieved by first pointing it to the north
        # cardinal point (tube along i, and focuser along -k) and
        # lifting it along the altitude to go to the zenith. Thus the
        # focuser now (in the "parked" position) points along 'i' and
        # the tube of the telescope points along 'k'. This is the
        # canonical orientation that `q` shall transform.
        #
        # Thus, when the telescope is at a point P(h, A), we expect
        # the downward "focuser" to point along
        #
        # sin(h) cos(A) i - sin(h) sin(A) j - cos(h) k
        #
        # Using the result
        #
        # q i q¯¹ = i [cos²µ cos(2ν) + sin²µ cos(2ρ)]
        #           + j [cos²µ sin(2ν) + sin²µ sin(2ρ)]
        #           + k sin(2µ) sin(ν - ρ)
        #
        # and comparing the coefficients of `k`, one sees that
        #
        # sin(ν - ρ) = 1 => ν - ρ = π/2
        #
        # Thus, we have:
        #
        # µ = z/2,
        # ν = -A/2
        # ρ = (π - A)/2
        #
        # Thus, the desired quaternion is:
        #
        # q = cos(z/2)cos(A/2) + i sin(z/2)sin(A/2) + j sin(z/2)cos(A/2) - k cos(z/2)sin(A/2)

        alt, az, frame = altaz

        half_z = math.radians((90 - alt)/2.0)
        half_A = math.radians(az/2.0)

        chz, shz = math.cos(half_z), math.sin(half_z)
        chA, shA = math.cos(half_A), math.sin(half_A)

        q = Quaternion(
            chz * chA,
            shz * shA,
            shz * chA,
            -chz * shA
        )

        return TimestampedQuaternion(
            ut=frame.UT,
            q=q,
        )

    def makeAltAzFrame(self, datetime_utc, refraction=True, temperature=25, humidity=0.3):
        # Check input sanity
        assert isinstance(datetime_utc, datetime.datetime)
        assert type(refraction) is bool
        assert type(temperature) in (int, float)
        assert type(humidity) is float

        if self._earth_location is None:
            raise RuntimeError('Cannot convert coordinates: Geolocation not initialized')

        obs_jd = julian.to_jd(datetime_utc)

        # Implementing algorithm for ST at prime meridian from Meeus, Chap 12
        ref_jd = int(obs_jd) + 0.5
        T = (ref_jd - 2451545.0)/36525 # Julian centuries

        ref_theta_0 = (
            100.46061837 + 36000.770053608 * T
            + 0.000387933 * T * T - (T**3)/(38710000)
        ) # degrees
        theta_0 = (ref_theta_0 + 1.00273790935 * 360.0 * (obs_jd - ref_jd)) # Mean ST @ Greenwich
        while theta_0 < 0:
            theta_0 += 360
        while theta_0 >= 360:
            theta_0 -= 360

        # Compute LST
        LST = theta_0 - (-self._earth_location.longitude) # Westward longitudes are negative

        pressure = self._pressure if refraction else 0.0
        return AltAzFrame(
            UT=datetime_utc,
            LST=LST,
            JD=obs_jd, # JD
            refraction_enabled=refraction,
            location=self._earth_location,
            pressure=pressure, # kPa
            temperature=temperature, # °C
            relative_humidity=humidity, # RH
        )


    def ICRSToHorizontal(self, icrs, frame):
        """
        icrs: An ICRS coordinate object (ra, dec)
        frame: An AltAzFrame object, created using e.g. makeAltAzFrame
        """
        assert isinstance(icrs, ICRS)
        assert isinstance(frame, AltAzFrame)

        ra0, dec0 = icrs.ra, icrs.dec

        ra, dec = self.precessFromJ2000(ra0, dec0, frame.JD)
        HA = frame.LST - ra

        # Convert to radians
        HA = math.radians(HA)
        dec = math.radians(dec)
        lat = math.radians(frame.location.latitude)

        sinH = math.sin(HA)
        cosH = math.cos(HA)
        sindec = math.sin(dec)
        cosdec = math.cos(dec)
        tandec = sindec / cosdec
        sinlat = math.sin(lat)
        coslat = math.cos(lat)

        azimuth = math.atan2(sinH, cosH * sinlat - tandec * coslat) + math.pi
        altitude = math.asin(sinlat * sindec + coslat * cosdec * cosH)
        while azimuth >= 2 * math.pi:
            azimuth -= 2 * math.pi
        while azimuth <= - 2 * math.pi:
            azimuth += 2 * math.pi

        altitude = math.degrees(altitude)
        azimuth = math.degrees(azimuth)
        if frame.refraction_enabled:
            altitude = self.refract(altitude, frame)

        return AltAz(
            alt=altitude,
            az=azimuth,
            frame=frame,
        )


    def horizontalToICRS(self, altaz):
        """
        altaz: An AltAz object
        """

        altaz_frame = altaz.frame
        alt = altaz.alt
        az = altaz.az

        if altaz_frame.refraction_enabled:
            alt = self.unrefract(alt, altaz_frame)

        alt = math.radians(alt)
        az = math.radians(az) - math.pi
        lat = math.radians(altaz_frame.location.latitude)

        sinalt = math.sin(alt)
        cosalt = math.cos(alt)
        tanalt = sinalt / cosalt
        sinlat = math.sin(lat)
        coslat = math.cos(lat)
        sinaz = math.sin(az)
        cosaz = math.cos(az)

        HA = math.degrees(math.atan2(sinaz, cosaz * sinlat + tanalt * coslat))
        dec = math.degrees(math.asin(sinlat * sinalt - coslat * cosalt * cosaz))

        LST = altaz_frame.LST # degrees

        ra = LST - HA # degrees

        ra0, dec0 = self.precessToJ2000(ra, dec, altaz_frame.JD)

        return ICRS(ra=ra0, dec=dec0)


# Stateless methods
def angularDistance(p1: ICRS, p2: ICRS) -> float:
    """
    Given two points (ICRS coordinates) `p1` and `p2`, return the
    angular distance between the two points in degrees

    Note: Uses Haversine formula
    """

    hv_ddec = math.sin(math.radians((p2.dec - p1.dec)/2.0)) ** 2
    hv_dra = math.sin(math.radians((p2.ra - p1.ra)/2.0)) ** 2
    cos_dec1 = math.cos(math.radians(p1.dec))
    cos_dec2 = math.cos(math.radians(p2.dec))

    d = 2 * math.asin(
        math.sqrt(
            hv_ddec + cos_dec1 * cos_dec2 * hv_dra
        )
    )

    return math.degrees(d)
