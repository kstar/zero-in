"""
Manage connection to the Arduino's Serial interface for the IMU
"""

import logging
import sys
logger = logging.getLogger('MotionSensor')
logger.setLevel(logging.INFO)
import serial
import time
import binascii
import numpy as np
import threading
import datetime
from pyquaternion import Quaternion
from coordinates import TimestampedQuaternion
from PyQt5.QtCore import pyqtSignal, QObject
from multiprocessing import Process

# The serial protocol is defined below; must be in sync with the Arduino sketch!
SERIAL_CONNCHECK = b'\x01'
SERIAL_STATUS = b'\x02'
SERIAL_OK = b'\x03'
SERIAL_FAULT = b'\x04'
SERIAL_ENABLE_INTERRUPTS = b'\x05'
SERIAL_DISABLE_INTERRUPTS = b'\x06'
SERIAL_INTERRUPTS_ENABLED = b'\x07'
SERIAL_READ_QUATERNION = b'\x0e'
SERIAL_READ_EULER = b'\x0f'
SERIAL_RESET = b'\x10'
SERIAL_MOTION_INTERRUPT = b'\x11'
SERIAL_NO_MOTION_INTERRUPT = b'\x12'
SERIAL_TOGGLE_HUMAN_READABLE = b'\x13'
SERIAL_DATA_BEGIN = b'\xae'
SERIAL_DATA_END = b'\xaf'

def _interpret_data(bytesequence, dtype=None):
    if bytesequence[0] != SERIAL_DATA_BEGIN[0] or bytesequence[-1] != SERIAL_DATA_END[0]:
        logger.warning('Interpret data received incorrect data begin/end')
        return None
    buf = binascii.unhexlify(bytesequence[1:-1])
    if dtype is None:
        return buf
    return np.frombuffer(buf, dtype=dtype)

class MotionSensor(QObject):
    """ Base class """
    _dev = None
    _last_quaternion = None

    def connect(self, *args, **kwargs):
        raise NotImplementedError('Base class must implement the `connect` method and assign the device to `self._dev`')

    @property
    def connected(self):
        return (self._dev is not None)

    @property
    def quaternion(self):
        return self._quaternion()

    @property
    def most_recent_tq(self):
        return TimestampedQuaternion(
            ut=self._last_quaternion['ut'],
            q=self._last_quaternion['q'],
        ) if self._last_quaternion else None

    def _quaternion(self):
        raise NotImplementedError('Base classes must implement the _quaternion method')

    def poll(self):
        raise NotImplementedError('Base classes must implement the poll method')

class ArduinoMotionSensor(MotionSensor):

    _serialPort = None
    _baud = None
    _interrupts = set()

    def connect(self, port='/dev/ttyACM0', baudrate=9600, timeout=10):
        if self._dev is not None:
            raise RuntimeError('Already connected to motion sensor.')
        self._dev = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self._serialPort = port
        self._baudRate = baudrate

    @property
    def _isr(self):
        raise NotImplementedError('No interrupt handler (_isr method) implemented')

    def _read(self, N, echo=False):
        """
        Reads N control OR data sequences, echoing any non-control bytes seen.
        An entire buffer of data bytes is treated as if it is one control sequence
        If N == 0, then read and sort all available data until there is no more
        We wait for data sequences to be completed
        Interrupt bytes are not counted as control bytes
        """
        non_control = []
        control = []
        data = []
        data_mode = False
        while (N == 0 and self._dev.in_waiting) or len(control) < N or data_mode:
            byte = self._dev.read()
            if byte is None or len(byte) == 0:
                raise TimeoutError('Did not receive the expected response from the serial line in the expected time')
            if byte == SERIAL_DATA_BEGIN:
                if data_mode:
                    raise RuntimeError('Received SERIAL_DATA_BEGIN in data_mode!')
                data.append(byte)
                data_mode = True
            elif byte == SERIAL_DATA_END:
                if not data_mode:
                    logger.error('Received SERIAL_DATA_END without SERIAL_DATA_BEGIN. Will ignore it.')
                else:
                    data.append(byte)
                    control.append(b''.join(data))
                    data = []
                    data_mode = False
            elif byte in self._interrupts:
                self._isr(byte)
            elif data_mode:
                data.append(byte)
            elif byte in (b'\r', b'\t', b'\n') or (byte >= b' ' and byte < b'\x7f') and not data_mode:
                non_control.append(byte)
            else:
                control.append(byte)

        if echo:
            print((b''.join(non_control)).decode(), file=sys.stderr, flush=True)

        return control



class ComplexArduinoMotionSensor(ArduinoMotionSensor):
    _in_motion = False
    _comm_lock = threading.RLock()

    motionDetected = pyqtSignal(name='motionDetected')
    motionStopped = pyqtSignal(name='motionStopped')

    _interrupts = {
        SERIAL_MOTION_INTERRUPT,
        SERIAL_NO_MOTION_INTERRUPT,
    }

    def _poll_for_interrupt(self):
        self._read(0)

    def _isr(self, instruction):
        if instruction == SERIAL_MOTION_INTERRUPT:
            logger.info('Received "in motion" interrupt')
            self._in_motion = True
            self.motionDetected.emit()
        elif instruction == SERIAL_NO_MOTION_INTERRUPT:
            logger.info('Received "no motion" interrupt')
            self._in_motion = False
            self.motionStopped.emit()
        else:
            assert False, instruction

    @property
    def in_motion(self):
        return self._in_motion

    def flush(self):
        control = self._read(0)
        if len(control) > 0:
            logger.warning('Flushed {} control sequences'.format(len(control)))

    def checkConnection(self):
        with self._comm_lock:
            self._dev.write(SERIAL_CONNCHECK)
            try:
                result = self._read(1)
            except StopIteration:
                return False

            if not result[0] == SERIAL_OK:
                    raise ValueError('Unexpected response from motion sensor to connection check: {}'.format(result[0]))

        return True


    def _quaternion(self, timeout=-1):
#        logger.info('Quaternion read requested')
        lock_acquired = self._comm_lock.acquire(blocking=True, timeout=timeout)
        if not lock_acquired:
            logger.info('Failed to acquire lock within timeout {}'.format(timeout))
            return None
#        logger.info('Lock acquired')
        self.flush()
        self._dev.write(SERIAL_READ_QUATERNION)
        result = self._read(1)[0]
        quat_data = _interpret_data(result, dtype=np.int16).astype(np.float64)
        assert quat_data.shape[0] == 4, quat_data.shape
        q = Quaternion(quat_data).normalised
        self._comm_lock.release()
        self._last_quaternion = {
            'q': q,
            't': time.time(), # Unix time
            'ut': datetime.datetime.utcnow(),
        }
#        logger.info('Read quaternion: {}'.format(q))
        return q


    @property
    def euler(self):
        with self._comm_lock:
            self._dev.write(SERIAL_READ_EULER)
            result = self._read(1)[0]
            heading, roll, pitch = map(float, _interpret_data(result, dtype=np.float32))
        return (heading, roll, pitch)


    def poll(self):
        """Slot to be invoked from a QTimer to poll for information. Interrupts get polled in the process of updating the quaternion"""

        try:
            q = self._quaternion(timeout=0)
        except Exception as e:
            logger.error('Exception when trying to poll for quaternion: {}'.format(str(e)))
            return

        if q is not None:
            ut = self._last_quaternion['ut']
            self.motionUpdate.emit(TimestampedQuaternion(ut=ut,q=q))


class SimpleArduinoMotionSensor(ArduinoMotionSensor):
    _autopoll = None

    motionUpdate = pyqtSignal(TimestampedQuaternion, name='motionUpdate')

    def _quaternion(self):
        result = self._read(1)[0]
        quat_data = _interpret_data(result, dtype=np.int16).astype(np.float64)
        assert quat_data.shape[0] == 4, quat_data.shape
        q = Quaternion(quat_data).normalised
        self._last_quaternion = {
            'q': q,
            't': time.time(), # Unix time
            'ut': datetime.datetime.utcnow(),
        }
        logger.debug(f'Read quaternion: {q}') # Remove later
        return q

    def _autopoll_begin(self, poll_delay=100.0):
        """
        poll_delay: microseconds
        """

        raise NotImplementedError('FIXME: This does not seem to work, probably due to arcane issues with multiprocessing in Python')

        if self._dev is None:
            raise RuntimeError('Cannot begin polling without a live serial link. Please call hte connect method first.')

        poll_delay_s = poll_delay / 1000000.0

        if self._autopoll:
            logger.error(
                f'Error: autopoll called more than once on the same object: {id(self)}.'
                f'Ignoring request.'
            )
            return

        def poll_closure():
            """
            poll_delay: microseconds
            """
            try:
                self._read(0)
            except Exception as e:
                logger.error(f'Exception while flushing: {e}')

            while True:
                try:
                    self._read(0) # Might slow things down but make things more reliable
                except Exception as e:
                    logger.error(f'Exception while flushing: {e}')

                try:
                    print('Fetching quaternion', file=sys.stderr)
                    self._quaternion()
                    print('Fetched quaternion', file=sys.stderr)
                except Exception as e:
                    logger.error(f'Failed to poll IMU quaternion because of exception: {e}')

                time.sleep(poll_delay_s)

        self._autopoll = Process(target=poll_closure)
        self._autopoll.start()
        logger.debug(f'Spawned serial IMU polling process {self._autopoll.pid}')

    def poll(self):

        if self._dev is None:
            raise RuntimeError('Cannot poll IMU without serial link. Please connect first.')

        try:
            self._read(0) # Flush
            q = self._quaternion()
        except Exception as e:
            logger.error(f'Exception when trying to poll for quaternion: {e}')
            return

        if q is not None:
            ut = self._last_quaternion['ut']
            self.motionUpdate.emit(TimestampedQuaternion(ut=ut,q=q))

    def __del__(self):
        logger.debug(f'Joining serial IMU polling process {self._autopoll.pid}')
        self._autopoll.join()
