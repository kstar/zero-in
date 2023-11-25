import logging
logger = logging.getLogger('IMUCalibration')
import math
import collections
import datetime
import json

from pyquaternion import Quaternion

from coordinates import TimestampedQuaternion, AltAz, AltAzFrame, CoordinateConversion
from dms import pretty_dec

def _format_hms(ut):
    return ut.strftime('%H:%M:%S') if isinstance(ut, datetime.datetime) else 'NULL'

class ImuCalibrationPoint:
    """
    Encapsulates an TimestampedQuaternion object and an AltAz object
    """

    def __init__(self, tq_imu, altaz_scope):
        self._tq_imu = tq_imu
        self._altaz_scope = altaz_scope

    @property
    def tq_imu(self):
        return self._tq_imu

    @property
    def q_imu(self):
        return self._tq_imu.q

    @property
    def ut(self):
        return self._tq_imu.ut

    @property
    def h(self):
        return self._altaz_scope.alt

    @property
    def A(self):
        return self._altaz_scope.az

    @property
    def altaz_scope(self):
        return self._altaz_scope

    def __str__(self):
        return (
            f'@UT={_format_hms(self.ut)}: IMU={self.q_imu}, '
            f'alt={pretty_dec(self.h)}, az={pretty_dec(self.A)} '
            f'(frameUT={_format_hms(self._altaz_scope.frame.UT)})'
        )

class _CalibrationInfo:
    point=None
    old_point=None
    az_offset=None
    az_offset_ut=None

class ImuCalibration:
    """ Manages IMU Calibrations in a stateful manner """

    def __init__(self, coordinate_converter: CoordinateConversion):
        assert isinstance(coordinate_converter, CoordinateConversion)
        self._c = coordinate_converter
        self._calibration = _CalibrationInfo()

    def reset(self):
        logger.warning('Resetting IMU Calibration!')
        self._calibration = _CalibrationInfo()

    def _calibrate_azimuth(self, point1: ImuCalibrationPoint, point2: ImuCalibrationPoint):
        """
        point1: Old point
        point2: New point
        """
        # See Page 42

        y = (point2.q_imu * point1.q_imu.inverse).normalised

        tv2 = self._c.horizontalToVector(point2.altaz_scope)
        tv1 = self._c.horizontalToVector(point1.altaz_scope)
        v2, v1 = tv2.q, tv1.q

        x2, x1 = v2[1], v1[1]
        y2, y1 = v2[2], v1[2]

        A0 = (
            math.atan2(y[2], y[1])
            - math.atan2(
                y[0] * (x2 - x1) + y[3] * (y2 + y1),
                y[3] * (x2 + x1) - y[0] * (y2 - y1),
            )
        )

        self._calibration.az_offset = math.degrees(A0)
        self._calibration.az_offset_ut = tv2.ut

    def calibrate(self, point: ImuCalibrationPoint, calib_alt_change=10.0, point_expiry_time=180):
        assert isinstance(point, ImuCalibrationPoint)
        usage=None

        logger.info(f'IMU Calibration Point: {point}')

        if self._calibration.old_point and float(point.ut - self._calibration.old_point.ut) > point_expiry_time:
            logger.info(f'Expunging antiquated: {self._calibration.old_point}')
            self._calibration.old_point = None

        if self._calibration.point is None:
            logger.info('First point: Storing.')
            self._calibration.point = point
            usage='first_point'

        elif abs(self._calibration.point.h - point.h) >= calib_alt_change:
            # Calibrate Azimuth
            logger.info('Sufficient alt change to calibrate azimuth.')
            old_az_offset, old_az_offset_ut = self._calibration.az_offset, self._calibration.az_offset_ut
            self._calibrate_azimuth(self._calibration.point, point)
            az_offset, az_offset_ut = self._calibration.az_offset, self._calibration.az_offset_ut
            logger.info(
                f'Overwrote {old_az_offset}° @{_format_hms(old_az_offset_ut)} '
                f'with {az_offset}° @{_format_hms(az_offset_ut)}'
            )
            self._calibration.point = point
            self._calibration.old_point = None
            usage='az_offset_using_current_point'

        elif (
                self._calibration.az_offset is None # FIXME: Maybe we want to relax this?
                and self._calibration.old_point is not None
                and abs(self._calibration.old_point.h - point.h) >= calib_alt_change
        ):
            logger.info('Sufficient alt change to calibrate azimuth with OLD point {self._calibration.old_point}')
            old_az_offset, old_az_offset_ut = self._calibration.az_offset, self._calibration.az_offset_ut
            self._calibrate_azimuth(self._calibration.old_point, point)
            az_offset, az_offset_ut = self._calibration.az_offset, self._calibration.az_offset_ut
            logger.info(
                f'Overwrote {old_az_offset}° @{_format_hms(old_az_offset_ut)} '
                f'with {az_offset}° @{_format_hms(az_offset_ut)}'
            )
            self._calibration.point = point
            self._calibration.old_point = None
            usage='az_offset_using_old_point'
        else:
            logger.info('Updating calibration point')
            self._calibration.point = point # Update only the current point and not the az offset.
            usage='update_current_point'

        now = datetime.datetime.now().strftime('%Y%m%d')
        with open(f'{now}_calibration.json', 'a') as f:
            json.dump({
                'UT': _format_hms(point.ut),
                'point': {
                    'q_imu': list(map(float, point.q_imu.q)),
                    'alt': point.altaz_scope.alt,
                    'az': point.altaz_scope.az,
                },
                'usage': usage,
                'current_calibration': {
                    'point': str(self._calibration.point),
                    'old_point': str(self._calibration.old_point),
                    'az_offset': self._calibration.az_offset,
                    'az_offset_ut': {
                        'date': self._calibration.az_offset_ut.strftime('%Y%m%d'),
                        'time': self._calibration.az_offset_ut.strftime('%H%M%S'),
                    } if self._calibration.az_offset_ut is not None else None,
                },
            }, f)

    @property
    def is_calibrated(self):
        return (self._calibration.az_offset is not None and self._calibration.point is not None)

    @property
    def calibration_timestamps(self):
        if not self.is_calibrated:
            return None
        return self._calibration.az_offset_ut, self._calibration.point.ut


    def predict(self, tq_imu: TimestampedQuaternion) -> AltAz:
        if not self.is_calibrated:
            raise RuntimeError('Trying to predict scope position from IMU reading without a valid calibration')

        # change in IMU quaternion in IMU-anchored reference frame
        y = (tq_imu.q * self._calibration.point.q_imu.inverse).normalised

        A0 = math.radians(self._calibration.az_offset)
        cos_A0, sin_A0 = math.cos(A0), math.sin(A0)

        # change in IMU quaternion is north-referenced reference frame
        ybar = Quaternion(
            y[0],
            y[1] * cos_A0 + y[2] * sin_A0,
            y[2] * cos_A0 - y[1] * sin_A0,
            y[3],
        ).normalised

        tv_calib = self._c.horizontalToVector(self._calibration.point.altaz_scope)
        v_calib = tv_calib.q

        frame = self._calibration.point.altaz_scope.frame
        refraction, temperature, humidity = frame.refraction_enabled, frame.temperature, frame.relative_humidity
        res = self._c.vectorToHorizontal(
            TimestampedQuaternion(ut=tq_imu.ut, q=(ybar * v_calib * ybar.inverse)),
            refraction=refraction, temperature=temperature, humidity=humidity
        )
        # logger.info(f'Quaternion {tq_imu.q} @{_format_hms(tq_imu.ut)} / AzOffset {self._calibration.az_offset}° => {pretty_dec(res.alt)}, {pretty_dec(res.az)}')
        return res
