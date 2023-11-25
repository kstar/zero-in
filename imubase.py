import logging
logger = logging.getLogger('IMUBase')
logger.setLevel(logging.INFO)

from PyQt5.QtCore import QObject, pyqtSignal

from coordinates import TimestampedQuaternion

class MotionSensor(QObject):
    """ Base class """
    _dev = None
    _last_quaternion = None
    _q_mem = None
    _filtering = 0.0

    motionUpdate = pyqtSignal(TimestampedQuaternion, name='motionUpdate')

    def connect(self, *args, **kwargs):
        raise NotImplementedError('Derived class must implement the `connect` method and assign the device to `self._dev`')

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

    def lowpass(self, enabled: bool, alpha: float):
        """
        alpha: amount of memory (parameter between 0 and 1, should be close to 0)

        Enable SLERP-based quaternion lowpass:

        If the estimate from the sensor is q̂ₖ, we report

        qₖ = q̂ₖ (q̂ₖ⁻¹ qₖ₋₁)^α      k ≥ 1
        q₀ = q̂₀

        The memory is reset every time this method is called
        """
        if not (alpha > 0.0 and alpha < 1.0):
            return ValueError(f'Invalid lowpass memory coefficient: {alpha}')
        self._q_mem = None
        if enabled:
            self._filtering = alpha
        else:
            self._filtering = 0.0


    def _raw_quaternion(self, timeout=-1, **kwargs) -> dict:
        """ Return a dict with {'q': Quaternion, 't': unix time, 'ut': datetime.datetime.utcnow()} """
        raise NotImplementedError('Derived classes must implement the _raw_quaternion method')

    def _quaternion(self, **kwargs):
        _qdict = self._raw_quaternion(**kwargs)
        q = _qdict['q']
        if self._filtering > 0.0:
            if self._q_mem is None:
                self._q_mem = (0, q)
                k = 0
            else:
                q_est = q
                k, q_old = self._q_mem
                # FIXME: Scale the memory coefficient based on the time interval between subsequent quaternions!
                q = (q_est * ((q_est.inverse * self._q_mem[1]) ** self._filtering)).normalised
                k += 1
                self._q_mem = (k, q)
            _qdict['memory'] = {'k': k, 'alpha': self._filtering}
        else:
            _qdict['memory'] = None

        _qdict['q'] = q
        self._last_quaternion = dict(_qdict)
        return q


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
