import logging
import functools

from PyQt5.QtCore import pyqtSignal, QObject, QTimer
import gpsd

from coordinates import GeoLocation

logger = logging.getLogger('GPS')

class GPSManager(QObject):
    """
    A class to manage connection with gpsd
    """

    gpsUpdate = pyqtSignal(GeoLocation, name='gpsUpdate')

    class NoDeviceError(Exception):
        pass

    def __init__(self):
        self._connected = False
        self._timer = None
        self._stop_polling = False

    def connect(self, gpsd_host='127.0.0.1', gpsd_port=2947):
        try:
            gpsd.connect(host=gpsd_host, port=gpsd_port)
        except ConnectionRefusedError:
            raise ConnectionRefusedError(f'Could not connect to gpsd on {gpsd_host}:{gpsd_port}. Make sure it is running.')

        try:
            gpsd.device()
        except IndexError:
            raise GPSManager.NoDeviceError(f'No GPS devices found')

        self._connected = True
        return True

    @property
    def connected(self):
        return self._connected

    def position(self):
        if not self._connected:
            logger.error(f'No connection to gpsd')
            return (None, None)
        try:
            packet = gpsd.get_current()
            return packet.position()
        except gpsd.NoFixError:
            logger.error(f'Do not have a fix to return GPS position yet')
            return (None, None)

    @property
    def fix(self):
        packet = gpsd.get_current()
        return bool(packet.mode >= 2)

    def altitude(self):
        if not self._connected:
            logger.error(f'No connection to gpsd')
            return None
        try:
            packet = gpsd.get_current()
            return packet.altitude()
        except gpsd.NoFixError:
            logger.error(f'Do not have a 3-D GPS fix to return altitude yet')
            return None

    def stop_polling(self):
        self._stop_polling = True

    def auto(self, interval_prefix=60, interval_postfix=None):
        """
        Automatically poll to try and get a fix.

        interval_prefix: Poll interval (seconds) before getting a full 3-D fix
        interval_postfix: Poll interval (seconds) after getting a full 3-D fix

        interval_prefix cannot be None
        interval_postfix can be None, indicates that we should stop polling once we have a 3D fix
        
        Emits the signal gpsUpdate every time there is an update
        """

        if not self._connected:
            raise RuntimeError(f'Trying to poll for GPS fix before connecting to gpsd')

        if self._stop_polling:
            logger.warning(f'Stopping polling for GPS position because I was asked to.')
            self._stop_polling = False
            return True

        packet = gpsd.get_current()
        pos = (None, None)
        elev = None
        if packet.mode >= 2:
            pos = packet.position()
        if packet.mode >= 3:
            elev = packet.altitude()

        if pos[0] or pos[1] or elev:
            self.gpsUpdate.emit(GeoLocation(latitude=pos[0], longitude=pos[1], elevation=elev))

        poll = False
        interval = None
        if packet.mode < 3:
            poll = True
            interval = interval_prefix
        elif interval_postfix:
            poll = True
            interval = interval_postfix

        if poll:
            if self._timer:
                del self._timer
            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setInterval(int(interval * 1000))
            def _poll():
                return self.auto(interval_prefix=interval_prefix, interval_postfix=interval_postfix)
            self._timer.timeout.connect(_poll)
            self._timer.start()

        return (not poll) # True if we're done
