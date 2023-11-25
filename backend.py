import threading
import logging
import sys
import concurrent.futures
import datetime
import collections
import os
import json
import math
import pickle
import time
import typing

from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal, QObject, QTimer
import numpy as np
from pyquaternion import Quaternion

import timing
from dms import *
from error_handler import error_handler
from ccd_client import CCDClient, Debayer
import platesolve
from platesolve import PlateSolution
from coordinates import CoordinateConversion, ICRS, AltAzFrame, AltAz, TimestampedQuaternion, LiteAltAzFrame, angularDistance
from eyepieceview import compute_eyepiece_rotation
from dssclient import DSSClient
from arduinoimu import SimpleArduinoMotionSensor
from usfsimu import USFSMotionSensor, TimestampedTemperature # FIXME: Move TimestampedTemperature elsewhere
from imucalib import ImuCalibration, ImuCalibrationPoint


logger = logging.getLogger("Backend")

SyncResult = collections.namedtuple(
    "SyncResult",
    ["sync_info", "frame", "cache"]
)
SyncResult.__doc__ = """
An immutable tuple containing information about a time-sync (`sync_info`) and the AltAzFrame (`frame`) corresponding to the time-sync. An optional `cache` dictionary is provided that can be filled with any computations pertaining to the specific sync.
"""

PLATESOLVE_DEBUG_DIR = '/mnt/AstroCaptures/platesolvedebug/' # When not None, any plate solve failures and successes are dated and saved in the PLATESOLVE_DEBUG_DIR directory

def debug_dump_image(fits_image, context, result, metadata):
    try:
        if PLATESOLVE_DEBUG_DIR:
            dump_path = os.path.join(
                PLATESOLVE_DEBUG_DIR,
                '{}/{}/{}/{}'.format(
                    result, # success or failure
                    datetime.datetime.now().strftime('%Y-%m-%d'),
                    context,
                    datetime.datetime.now().strftime('%H%M%S'),
                )
            )
            if not os.path.isdir(os.path.dirname(dump_path)):
                os.makedirs(os.path.dirname(dump_path))
            with open(dump_path + '.fits', 'wb') as fd:
                fd.write(fits_image)
            with open(dump_path + '.json', 'w') as fd:
                json.dump(metadata, fd)
    except Exception as e:
        logger.warning(f'Could not save plate-solve {result} debug image to {dump_path}.')


def make_qimage_from_image_data(image_data, gamma=0.5):
    if gamma and gamma != 1.0:
        scaled_data = np.ascontiguousarray(np.minimum(
            np.maximum(
                ((np.asarray(image_data).astype(np.float32)/255.0) ** gamma) * 255.0, 0
            ), 255
        ).astype(np.uint8))
    else:
        scaled_data = np.ascontiguousarray(image_data)

    if len(scaled_data.shape) == 2:
        qimg = QtGui.QImage(
            scaled_data,
            scaled_data.shape[1], scaled_data.shape[0], scaled_data.strides[0],
            QtGui.QImage.Format_Indexed8
        )
        qimg.setColorTable([QtGui.qRgb(i, i, i) for i in range(256)])

    elif len(scaled_data.shape) == 3 and scaled_data.shape[2] == 3:
        qimg = QtGui.QImage(
            scaled_data,
            scaled_data.shape[0], scaled_data.shape[1], scaled_data.strides[0],
            QtGui.QImage.Format_RGB888
        )

    else:
        raise NotImplementedError(
            'No idea how to turn numpy array of shape {} into QImage'.format(
                scaled_data.shape
            )
        )

    return qimg

class MainBackend(QObject):

    alignmentChanged = pyqtSignal(dict, name='alignmentChanged')
    temperatureUpdate = pyqtSignal(TimestampedTemperature, name='temperatureUpdate')
    temperaturePollingError = pyqtSignal(str, name='temperaturePollingError')

    def __init__(self):
        super().__init__()
        self._ccd_client = None
        self._alignment = None
        self._exposure_lock = threading.Lock()
        self._last_exposure_lt = None
        self._sync = None
        self._coordinate_conversion = CoordinateConversion()
        self._dss_client = DSSClient()
        self._read_temp_timer = QTimer(self)
        self._imu = None
        self._imu_calibration = ImuCalibration(self._coordinate_conversion)
        self._solver_settings = {
            'sep_threshold': None,
            'binning': 1,
            'auto_debayer': False,
            'top_k': 0,
        }
        self._solution = None
        self._read_temp_timer.timeout.connect(self._slotPollTemperature)
        self._plate_data = None

    def indiConnect(self, ccd_name, indi_host, indi_port):
        """
        Attempts to connect to the INDI server and connect to the CCD device
        """
        if self._ccd_client is not None:
            logger.warning('A connection already seems to exist! Will not do anything.')
            return

        self._ccd_name = ccd_name
        self._indi_host = indi_host
        self._indi_port = indi_port
        try:
            self._ccd_client = CCDClient(
                self._ccd_name,
                server_host=self._indi_host, server_port=self._indi_port
            )
        except Exception as e:
            self._ccd_client = None
            raise

    def imuConnect(self, device, **config):
        """
        Attempts to connect to the motion sensor over serial communication
        """
        if device == 'USFS':
            self._imu = USFSMotionSensor()
        elif device == 'Arduino':
            self._imu = SimpleArduinoMotionSensor()
        else:
            raise NotImplementedError(f'Unhandled IMU device type: {device}')

        self._imu_name = device
        self._imu.connect(**config)
        if self._imu.connected:
            self._read_imu_timer = QTimer(self)
            self._read_imu_timer.timeout.connect(self._imu.poll)
            self._read_imu_timer.start(int(100.0)) # Trigger an update every 100ms
        return self._imu.connected

    @property
    def imu(self):
        """ Expose access to the IMU for slot registration """
        return self._imu

    @property
    def coordinate_conversion(self):
        return self._coordinate_conversion

    def pollTemperature(self, poll, device='IMU', interval=30):
        """
        poll: True or False
        interval: in seconds
        """
        if poll and (device != 'IMU'):
            raise NotImplementedError(f'Unhandled temperature device: {device}')

        if poll:
            self._read_temp_device = device
            self._read_temp_timer.start(interval * 1000.0)
        else:
            self._read_temp_device = None
            self._read_temp_timer.stop()

    def _slotPollTemperature(self):
        if self._read_temp_device == 'IMU':
            if (not self._imu) or (not self._imu.connected):
                logger.error(f'Could not update temperature: Not connected to IMU')
                return
            try:
                tt = self._imu.temperature
                self._timestamped_temperature = tt
                self.temperatureUpdate.emit(tt)
            except Exception as e:
                self.temperaturePollingError.emit(f'Failed to poll temperature from IMU. Exception was {e}')
        else:
            raise NotImplementedError(f'Unhandled temperature device: {self._read_temp_device}')

    def expose(self, exposure, timeout=15):
        """
        Sychronously starts and finishes an exposure
        """
        if self._ccd_client is None:
            raise RuntimeError('INDI / CCD Connection not setup yet!')

        lock_acquired = self._exposure_lock.acquire(blocking=False) # pylint: disable=assignment-from-no-return
        if not lock_acquired:
            raise RuntimeError(
                'Failed to acquire exposure lock: '
                'Another exposure seems to be underway!'
            )

        result = self._ccd_client.expose(exposure, timeout=timeout)
        self._exposure_lock.release()
        self._last_exposure_lt = datetime.datetime.now()
        self._last_exposure_ut = datetime.datetime.utcnow()
        self._last_exposure_duration = exposure
        return result

    @property
    def last_exposure_lt(self):
        return self._last_exposure_lt

    @property
    def last_exposure_ut(self):
        return self._last_exposure_ut

    def loadSyncFromPickle(self, path='timesync.pkl'):
        try:
            with open(path, 'rb') as f:
                self._sync = pickle.load(f)
            self.setSiteParameters(
                self._sync.frame.location.latitude,
                self._sync.frame.location.longitude,
                self._sync.frame.location.elevation,
            )
            return self._sync
        except Exception as e:
            logger.error(f'Failed to load time sync pickle: {e}')
            self._sync = None
            return None

    def syncTime(self, platform_travel, temperature):
        tz_offset = round((datetime.datetime.now().timestamp() - datetime.datetime.utcnow().timestamp())/3600.0, 2)
        logger.info('Timezone offset is {:.2f} hours'.format(tz_offset))
        sync_ut = datetime.datetime.utcnow()
        sync_lt = sync_ut + datetime.timedelta(hours=tz_offset)
        scope_ut = sync_ut + datetime.timedelta(hours=(platform_travel/2.0))
        scope_lt = scope_ut + datetime.timedelta(hours=tz_offset)
        logger.info('Sync at time {} -- scope frame is referenced to {} UTC'.format(sync_lt.strftime('%H:%M:%S'), scope_ut.strftime('%H:%M:%S')))
        sync_info = {
            'scope_ut': scope_ut,
            'scope_lt': scope_lt,
            'sync_ut': sync_ut,
            'sync_lt': sync_lt,
            'temperature': temperature,
            'platform_travel': platform_travel,
            'tz_offset': tz_offset,
        }
        scope_frame = self._coordinate_conversion.makeAltAzFrame(
            scope_ut, temperature=temperature
        )

        self._sync = SyncResult(sync_info=sync_info, frame=scope_frame, cache={})
        try:
            with open('timesync.pkl', 'wb') as f:
                pickle.dump(self._sync, f)
        except Exception as e:
            logger.error(f'Encountered exception while trying to pickle the time sync: {e}')
        return self._sync

    def setSiteParameters(self, lat, lon, height):
        self._coordinate_conversion.setEarthLocation(lat, lon, height)

    def equatorialToHorizontal(self, ra, dec):
        """ Convenience wrapper """
        if not self._sync.frame:
            raise RuntimeError('Trying to convert ICRS to Horizontal Coordinates without a frame sync!')
        return self._coordinate_conversion.ICRSToHorizontal(
            ICRS(ra=ra, dec=dec), self._sync.frame
        )

    def setSolverSettings(
            self,
            sep_threshold: typing.Optional[float], # pylint:disable=unsubscriptable-object
            binning: int,
            top_k: int,
            auto_debayer: bool,
        ):

        """
        If sep_threshold is a float and not None, use SEP with this threshold
        If sep_threshold is None, do not use SEP -- use solve-field with the image directly
        """
        self._solver_settings = {
            'sep_threshold': sep_threshold,
            'binning': binning,
            'top_k': top_k,
            'auto_debayer': auto_debayer
        }
        return dict(self._solver_settings)

    def _updateAlignment(self, alignment):
        """ Updates the alignment while writing it into history """
        self._alignment = alignment
        if alignment is None:
            return

        with open('alignment_history.json', 'a') as ah:
            ah.write(json.dumps(self._alignment) + '\n')

        if alignment['source'] != 'loaded':
            with open('alignment.json', 'w') as af:
                json.dump(self._alignment, af)

        self.alignmentChanged.emit(self._alignment)

        return self._alignment


    def loadPreviousAlignmentIfExists(self, alignment_file="alignment.json"):
        try:
            with open(alignment_file, 'r') as jf:
                alignment = json.load(jf)
            assert isinstance(alignment, dict)
            assert {'x', 'y', 'arcsecperpix', 'timestamp',} - set(alignment) == set()
            alignment['source'] = 'loaded'
            alignment['loaded_from'] = alignment_file
            alignment['loaded_timestamp'] = datetime.datetime.timestamp(datetime.datetime.utcnow())
        except Exception as e:
            logger.error('Failed to load previous alignment from {}: {}'.format(
                alignment_file, str(e)
            ))
            alignment = None

        return self._updateAlignment(alignment)

    def localAlignmentSync(self, align_ra, align_dec) -> dict:
        if not self._solution:
            raise RuntimeError('Cannot do a local sync without a recent solution!')
        if not self._sync:
            raise RuntimeError('Will not do a local sync with a time sync!')

        align_datetime = datetime.datetime.utcnow()
        align_time = datetime.datetime.timestamp(align_datetime)

        try:
            # Find the RA/Dec as per current alignment ("before sync") -- this
            # is so we can log it and investigate and maybe build a pointing
            # model
            ra, dec = list(map(float, self._solution.to_radec(0, 0, relative_to='center')))

            # Find the alignment solution
            x, y = list(map(float, self._solution.to_pixels(align_ra, align_dec)))
            logger.info(
                f'Sync coordinates {pretty_ra(align_ra)} {pretty_dec(align_dec)} '
                f'map to (x, y) = ({x}, {y})'
            )

            arcsecperpix = self._solution.compute_scale()
            logger.info('Estimated arcsec/pixel: {:.3f}'.format(arcsecperpix))

            real_frame = self._coordinate_conversion.makeAltAzFrame(
                align_datetime, temperature=self._sync.sync_info['temperature'],
            )
            real_altaz = self._coordinate_conversion.ICRSToHorizontal(
                ICRS(ra=align_ra, dec=align_dec), real_frame
            )
            real_old_altaz = self._coordinate_conversion.ICRSToHorizontal(
                ICRS(ra=ra, dec=dec), real_frame
            )

            scope_altaz = self.equatorialToHorizontal(align_ra, align_dec)
            scope_old_altaz = self.equatorialToHorizontal(ra, dec)

            w, h = self._solution.image_size
            xc, yc = w / 2.0, h / 2.0

            alignment = {
                'x': x,
                'y': y,
                'x_c': (x - xc),
                'y_c': (yc - y),
                'dx': x - self._alignment['x'],
                'dy': y - self._alignment['y'],
                'arcsecperpix': arcsecperpix,
                'timestamp': align_time,
                'source': 'sync',
                'align_position': { # Position according to new alignment (after sync)
                    'icrs': (align_ra, align_dec), # This is the (RA, Dec) of the alignment target
                    'altaz': (real_altaz.alt, real_altaz.az), # This is the actual (Alt, Az) of the target in the sky at time of sync
                    'altaz_scopeframe': (scope_altaz.alt, scope_altaz.az), # This is the (Alt, Az) of the target in the scope LST frame
                },
                'old_alignment': { # Position according to previous alignment (before sync)
                    'icrs': (ra, dec), # This is the (RA, Dec) of the alignment that existed earlier
                    'altaz': (real_old_altaz.alt, real_old_altaz.az),
                    'altaz_scopeframe': (scope_old_altaz.alt, scope_old_altaz.az),
                },
                # TODO: Add temperature if the temperature can be determined from USFS board.
            }
        except Exception as e:
            logger.error('Failed to do local alignment sync. Exception was: {}'.format(e))
            alignment = self._alignment
            raise

        finally:
            return self._updateAlignment(alignment) # pylint:disable=E0601

    def alignCapturedImage(self, align_ra, align_dec, arcsecperpix_hint=None, timeout=120) -> dict:
        """
        Given that a CCD exposure has finished, plate-solve it to
        determine the pixel offset for the given (J2000.0) RA/Dec

        `sep_threshold`: set this to None to disable SEP. Otherwise, set it to a float to enable
        """
        if not self._sync:
            raise RuntimeError('Cannot align without a sync. This is because we need to log alt/az')
        align_datetime = datetime.datetime.utcnow()
        align_time = datetime.datetime.timestamp(align_datetime)
        if self._alignment is not None:
            logger.warning('An alignment already exists. Will overwrite it.')

        if arcsecperpix_hint is not None:
            scale_low_hint = arcsecperpix_hint * (0.800)
            scale_high_hint = arcsecperpix_hint * (1.25)
        else:
            scale_low_hint = None
            scale_high_hint = None

        meta = {
            'align_ra': align_ra,
            'align_dec': align_dec,
            'timeout': timeout,
            'arcsecperpix_hint': arcsecperpix_hint,
            'scale_low_hint': scale_low_hint,
            'scale_high_hint': scale_high_hint,
            'exposure_ut': self._last_exposure_ut.strftime('%H%M%S'),
            'exposure': self._last_exposure_duration,
        }
        meta.update(self._solver_settings)

        # FIXME: pylint does not like this popping of keys and therefore we have to disable important checks below
        solver_args = dict(self._solver_settings)
        auto_debayer = solver_args.pop('auto_debayer')
        sep_threshold = solver_args.pop('sep_threshold')
        solver_args.update({
            'scale_units': 'arcsecperpix',
            'scale_low': scale_low_hint,
            'scale_high': scale_high_hint,
            'ra': align_ra,
            'dec': align_dec,
            'radius': 8,
            'timeout': timeout,
        })

        try:
            # Solve around the Alignment coordinates
            if sep_threshold:
                if PLATESOLVE_DEBUG_DIR is not None:
                    fits_image = self._ccd_client.get_fits_image()
                else:
                    fits_image = None
                solution = platesolve.solve_field_sep( # pylint: disable=redundant-keyword-arg,unexpected-keyword-arg
                    self._ccd_client.get_image_data(auto_debayer=auto_debayer),
                    sep_threshold,
                    **solver_args
                )
            else:
                fits_image = self._ccd_client.get_fits_image()
                solution = platesolve.solve_field( # pylint: disable=redundant-keyword-arg,unexpected-keyword-arg
                    Debayer.debayer_fits_image(fits_image) if auto_debayer else fits_image,
                    **solver_args
                )

        except Exception as e:
            self._updateAlignment(None)
            logger.error('Alignment plate solve failed with exception: {}'.format(str(e)))
            debug_dump_image(fits_image, 'alignment', 'failure', meta)
            raise

        try:
            # Find the alignment solution
            x, y = list(map(float, solution.to_pixels(align_ra, align_dec)))
            logger.info('Found solution: (x, y) = ({}, {})'.format(x, y))

            arcsecperpix = solution.compute_scale()
            logger.info('Estimated arcsec/pixel: {:.3f}'.format(arcsecperpix))


            real_frame = self._coordinate_conversion.makeAltAzFrame(
                align_datetime, temperature=(
                    self._sync.sync_info['temperature'] if self._sync else 10
                )
            )
            real_altaz = self._coordinate_conversion.ICRSToHorizontal(
                ICRS(ra=align_ra, dec=align_dec), real_frame
            )

            scope_altaz = self.equatorialToHorizontal(align_ra, align_dec)

            w, h = solution.image_size
            xc, yc = w / 2.0, h / 2.0

            alignment = {
                'x': x,
                'y': y,
                'x_c': (x - xc),
                'y_c': (yc - y),
                'arcsecperpix': arcsecperpix,
                'timestamp': align_time,
                'source': 'align',
                'align_position': {
                    'icrs': (align_ra, align_dec),
                    'altaz': (real_altaz.alt, real_altaz.az),
                    'altaz_scopeframe': (scope_altaz.alt, scope_altaz.az),
                },
            }


            debug_dump_image(fits_image, 'alignment', 'success', dict(meta, alignment=alignment))
            self._solution = solution

            if (abs(x - xc) > w / 2.0) or (abs(y - yc) > h / 2.0):
                logger.warning(
                    'Alignment inferred very large pixel offsets ({:.2f}, {:.2f}) '
                    'which place the alignment target outside the CCD field of view. '
                    'Are you sure you have a valid alignment?'.format(
                        abs(x - xc), abs(y - yc)
                    )
                )

            return self._updateAlignment(alignment)

        except Exception as e:
            logger.error('Encountered exception while aligning: {}'.format(str(e)))
            self._updateAlignment(None)
            raise

        finally:
            return self._alignment

    def solveCapturedImage(self, timeout=20, tq_imu=None, sep_plot_detections=False) -> (float, float):
        """
        If the IMU TimestampedQuaternion tq_imu is supplied, we also updateImuOffset()
        """
        if self._alignment is None:
            raise RuntimeError(
                'It looks like alignment has not been performed! Please perform that first!'
            )

        # Plate solve
        arcsecperpix_hint = self._alignment['arcsecperpix']
        scale_lo, scale_hi = 0.95 * arcsecperpix_hint, 1.05 * arcsecperpix_hint
        logger.info(
            'Plate solving with scale range ({:.3f}, {:.3f}) arcsec/pixel'.format(
                scale_lo, scale_hi
            )
        )

        if tq_imu is not None and self._imu_calibration.is_calibrated:
            scope_est_icrs = self.getScopeICRS(tq_imu)
            ra, dec = scope_est_icrs.ra, scope_est_icrs.dec
        else:
            ra, dec = None, None

        meta = {
            'scope_est_ra': ra,
            'scope_est_dec': dec,
            'timeout': timeout,
            'arcsecperpix_hint': arcsecperpix_hint,
            'scale_low': scale_lo,
            'scale_high': scale_hi,
            'exposure_ut': self._last_exposure_ut.strftime('%H%M%S'),
            'exposure': self._last_exposure_duration,
        }
        meta.update(self._solver_settings)

        # FIXME: pylint does not like this popping of keys and therefore we have to disable important checks below
        solver_args = dict(self._solver_settings)
        auto_debayer = solver_args.pop('auto_debayer')
        sep_threshold = solver_args.pop('sep_threshold')
        solver_args.update({
            'scale_units': 'arcsecperpix',
            'scale_low': scale_lo,
            'scale_high': scale_hi,
            'timeout': timeout,
        })

        Timer = timing.makeOrGetTimingClass('PlateSolve')
        try:
            if sep_threshold:
                solver_args['plot_detections'] = sep_plot_detections
                if PLATESOLVE_DEBUG_DIR is not None:
                    fits_image = self._ccd_client.get_fits_image()
                else:
                    fits_image = None
                solution = platesolve.solve_field_sep(  # pylint: disable=redundant-keyword-arg,unexpected-keyword-arg
                    self._ccd_client.get_image_data(auto_debayer=auto_debayer),
                    sep_threshold,
                    **solver_args
                )
            else:
                with Timer('0:get_debayer_fits'):
                    fits_image = self._ccd_client.get_fits_image()
                    solve_image = Debayer.debayer_fits_image(fits_image) if auto_debayer else fits_image,
                solution = platesolve.solve_field( # pylint: disable=redundant-keyword-arg,unexpected-keyword-arg
                    fits_image,
                    **solver_args
                )

            Timer.timing.update(platesolve.timing)

        except Exception as e:
            logger.error('Solve failed with exception: {}'.format(str(e)))
            debug_dump_image(fits_image, 'solve', 'failure', meta)
            raise

        # Find the alignment solution, preview it
        self._solution = solution
        x, y = self._alignment['x'], self._alignment['y']
        ra, dec = list(
            map(
                float,
                solution.to_radec(x, y)
            )
        )
        logger.info('Coordinates at (x, y) = ({:.2f}, {:.2f}) are RA: {}, Dec: {}'.format(
            x, y, pretty_ra(ra), pretty_dec(dec)
        ))

        debug_dump_image(fits_image, 'solve', 'success', dict(meta, alignment=self._alignment, scope_pos=[ra, dec]))

        self._scope_pos = ICRS(ra=ra, dec=dec)

        if tq_imu is not None:
            self.updateImuOffset(tq_imu, self._scope_pos)

        return (ra, dec)

    def updateImuOffset(self, tq_imu, solved_pos):
        """
        Given the plate-solve result (solved_pos) and the IMU
        quaternion for the same time (as a TimestampedQuaternion),
        compute and store the offset between IMU frame and Scope frame

        tq_imu: a TimestampedQuaternion object indicating the IMU's
        quaternion output along with UT of the measurement

        solved_pos: an ICRS object indicating the scope's plate-solved
        (ra, dec)

        """

        earth_frame = self._coordinate_conversion.makeAltAzFrame(
            tq_imu.ut, temperature=self._sync.sync_info['temperature']
        )
        altaz = self._coordinate_conversion.ICRSToHorizontal(
            solved_pos, earth_frame
        )

        # For debugging only
        logger.debug(f'Ground-referenced scope altazimuth {pretty_dec(altaz.alt)} {pretty_dec(altaz.az)}')

        self._imu_calibration.calibrate(ImuCalibrationPoint(tq_imu, altaz))


    @property
    def lastImuCalibration(self):
        return self._imu_calibration.calibration_timestamps

    def getScopeICRS(self, tq_imu):
        """
        Returns Scope's ICRS estimate based off of the supplied IMU quaternion
        We must have called updateImuOffset() prior to this to store the offset
        """
        # FIXME: We may need to optimize this method as it has to take < 50ms
        # This may involve returning AltAz instead of ICRS

        altaz_scope = self._imu_calibration.predict(tq_imu)

        # Update temperature in case a frame-sync has occurred between
        # calibration and now
        # FIXME: Can't mutate a tuple
        #        altaz_scope.frame.temperature = self._sync.sync_info['temperature']

        logger.debug(
            'IMU-derived ground-referenced scope (alt, az): {}, {}'.format(
                pretty_dec(altaz_scope.alt), pretty_dec(altaz_scope.az)
            )
        )

        # Note: This happens in the earth-referenced frame, not the platform-referenced frame
        imu_icrs = self._coordinate_conversion.horizontalToICRS(altaz_scope)

        return imu_icrs

    def getDSSPlate(self, plate_size: float, max_offset=0.4, force=False) -> PlateSolution:
        """
        Returns a tuple containing a plate (PlateSolution object) and the current scope position

        NOTE: The plate need not be centered on the plate-solved scope
        position (self._scope_pos).

        If the plate_fov has not changed and the scope position's
        angular distance from the previous plate's center is within
        max_offset fraction of the plate FOV's radius (plate_size/2.0),
        we just return the previous plate with the scope
        position. Otherwise we fetch a fresh plate centered on the scope
        position.

        plate_size: arcminutes
        max_offset: The fraction of (plate_size/2.0) within which a change of scope_pos does not cause a new plate to be fetched
        force: Force fetching a new plate, ignoring the above-explained logic
        """

        if (
                not force
                and self._plate_data
                and self._plate_data['size'] == plate_size
                and angularDistance(
                    self._plate_data['center'],
                    self._scope_pos
                ) * 60.0 < max_offset * plate_size/2.0
        ):

            plate = self._plate_data['plate']

        else:
            plate = self._dss_client.getPlate(self._scope_pos, plate_size)
            self._plate_data = {
                'plate': plate,
                'size': plate_size,
                'center': ICRS(self._scope_pos.ra, self._scope_pos.dec),
            }

        return plate, self._scope_pos

    def _j2kncp(self):
        if 'j2kncp' not in self._sync.cache:
            self._sync.cache['j2kncp'] = self._coordinate_conversion.ICRSToEquatorial(
                ICRS(ra=0.0, dec=90.0), self._sync.frame.JD
            )
        return self._sync.cache['j2kncp']

    def getEyepieceRotation(self, focuser_angle: float, display_angle: float, north_offset: float, display_on_scope: bool):
        """Get the CW Eyepiece Rotation in Degrees

        focuser_angle: Focuser offset in degrees

        display_angle: (CCW) Offset angle of display viewport in degrees

        north_offset: (CCW) angle that north makes in the plate in
        degrees, obtained by calling
        PlateSolution.compute_north_angle. Usually close to zero.

        display_on_scope: Is the display mounted on the telescope? It
        must be mounted parallel to the image plane,
        with an orientation offset described by
        `display_angle`

        returns the *clockwise* rotation of the DSS plate (ICRS north
        up) in degrees
        """

        if not self._sync:
            raise RuntimeError('Cannot compute eyepiece rotation: time sync not performed')

        if math.isnan(north_offset):
            logger.error('north_offset came out to be NaN. Returning None! The orientation of the plate will be wrong!')
            return None

        scope_jnow = self._coordinate_conversion.ICRSToEquatorial(
            self._scope_pos, self._sync.frame.JD
        )
        return north_offset + compute_eyepiece_rotation(
            scope_jnow,
            self._j2kncp(),
            LiteAltAzFrame(self._sync.frame),
            focuser_angle,
            display_angle,
            display_on_scope,
        )
