import logging
import sys
import struct
from enum import Enum, IntEnum
from typing import Union
import time
import datetime
import collections
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('USFSIMU')


import numpy as np
from pyquaternion import Quaternion
from coordinates import TimestampedQuaternion

# I²C alternatives:
from smbus2 import SMBus
from pyftdi import i2c as pyftdi_i2c

from imubase import MotionSensor

TimestampedTemperature = collections.namedtuple( # FIXME: Move somewhere more general
    "TimestampedTemperature",
    ["ut", "T"]
)
TimestampedTemperature.__doc__ = """
An immutable tuple containing a temperature reading in degrees C along with a UTC timestamp indicating time of measurement
"""

# TODO:
#
# 1. Clean up pressure with TimestampedPressure
# 2. Move defs like TimestampedTemperature and TimestampedPressure to a common module
# 3. Implement the ISR for error handling etc
# 4. Plumb the magnetometer / accelerometer merge-rate settings
# 5. Tune the default values as needed
# 6. Write a method to set individual parameters during runtime

# Note: The following definitions have been copied from:
# https://github.com/kriswiner/EM7180_SENtral_sensor_hub/blob/master/EM7180_LSM6DSM_LIS2MDL_LPS22HB/USFS.h
# and modified for Python
#
# Note that the license for parts of this code is either 3-clause BSD
# or an extremely permissive "use as you deem appropriate with
# attribution license."
#
# Attribution: Tlera Corp. and Kris Winer

# EM7180 SENtral register map
# see http://www.emdeveloper.com/downloads/7180/EMSentral_EM7180_Register_Map_v1_3.pdf

class EM7180(Enum):
    QX                 = 0x00  # this is a 32-bit normalized floating point number read from registers 0x00-03
    QY                 = 0x04  # this is a 32-bit normalized floating point number read from registers 0x04-07
    QZ                 = 0x08  # this is a 32-bit normalized floating point number read from registers 0x08-0B
    QW                 = 0x0C  # this is a 32-bit normalized floating point number read from registers 0x0C-0F
    QTIME              = 0x10  # this is a 16-bit unsigned integer read from registers 0x10-11
    MX                 = 0x12  # int16_t from registers 0x12-13
    MY                 = 0x14  # int16_t from registers 0x14-15
    MZ                 = 0x16  # int16_t from registers 0x16-17
    MTIME              = 0x18  # uint16_t from registers 0x18-19
    AX                 = 0x1A  # int16_t from registers 0x1A-1B
    AY                 = 0x1C  # int16_t from registers 0x1C-1D
    AZ                 = 0x1E  # int16_t from registers 0x1E-1F
    ATIME              = 0x20  # uint16_t from registers 0x20-21
    GX                 = 0x22  # int16_t from registers 0x22-23
    GY                 = 0x24  # int16_t from registers 0x24-25
    GZ                 = 0x26  # int16_t from registers 0x26-27
    GTIME              = 0x28  # uint16_t from registers 0x28-29
    Baro               = 0x2A  # start of two-byte MS5637 pressure data, 16-bit signed interger
    BaroTIME           = 0x2C  # start of two-byte MS5637 pressure timestamp, 16-bit unsigned
    Temp               = 0x2E  # start of two-byte MS5637 temperature data, 16-bit signed interger
    TempTIME           = 0x30  # start of two-byte MS5637 temperature timestamp, 16-bit unsigned
    QRateDivisor       = 0x32  # uint8_t
    EnableEvents       = 0x33
    HostControl        = 0x34
    EventStatus        = 0x35
    SensorStatus       = 0x36
    SentralStatus      = 0x37
    AlgorithmStatus    = 0x38
    FeatureFlags       = 0x39
    ParamAcknowledge   = 0x3A
    SavedParamByte0    = 0x3B
    SavedParamByte1    = 0x3C
    SavedParamByte2    = 0x3D
    SavedParamByte3    = 0x3E
    ActualMagRate      = 0x45
    ActualAccelRate    = 0x46
    ActualGyroRate     = 0x47
    ActualBaroRate     = 0x48
    ActualTempRate     = 0x49
    ErrorRegister      = 0x50
    AlgorithmControl   = 0x54
    MagRate            = 0x55
    AccelRate          = 0x56
    GyroRate           = 0x57
    BaroRate           = 0x58
    TempRate           = 0x59
    LoadParamByte0     = 0x60
    LoadParamByte1     = 0x61
    LoadParamByte2     = 0x62
    LoadParamByte3     = 0x63
    ParamRequest       = 0x64
    ROMVersion1        = 0x70
    ROMVersion2        = 0x71
    RAMVersion1        = 0x72
    RAMVersion2        = 0x73
    ProductID          = 0x90
    RevisionID         = 0x91
    RunStatus          = 0x92
    UploadAddress      = 0x94 # uint16_t registers 0x94 (MSB)-5(LSB)
    UploadData         = 0x96
    CRCHost            = 0x97  # uint32_t from registers 0x97-9A
    ResetRequest       = 0x9B
    PassThruStatus     = 0x9E
    PassThruControl    = 0xA0
    ACC_LPF_BW         = 0x5B  #Register GP36
    GYRO_LPF_BW        = 0x5C  #Register GP37
    BARO_LPF_BW        = 0x5D  #Register GP38
    ADDRESS            = 0x28   # Address of the EM7180 SENtral sensor hub

    # Parameter numbers (undocumented; c.f. https://github.com/gregtomasch/EM7180_SENtral_Calibration/issues/8)
    ParamGyroFS        = 75
    ParamMagAccFS      = 74
    ParamStillness     = 0x49
    ParamGBias         = 0x48
    ParamTransientProt = 59
    ParamMagMergeRate  = 52
    ParamAccMergeRate  = 53

class _Address(Enum):
    M24512DFM_DATA_ADDRESS    = 0x50   # Address of the 500 page M24512DFM EEPROM data buffer, 1024 bits (128 8-bit bytes) per page
    M24512DFM_IDPAGE_ADDRESS  = 0x58   # Address of the single M24512DFM lockable EEPROM ID page
    MPU9250_ADDRESS           = 0x68   # Device address when ADO = 0
    AK8963_ADDRESS            = 0x0C   #  Address of magnetometer
    MS5637_ADDRESS            = 0x76   # Address of altimeter

class SENtralParams(IntEnum):
    # Choose Interrupts to Enable (can be ORed)
    IntCPUReset    = 0x01
    IntError       = 0x02
    IntQuatResult  = 0x04
    IntMagResult   = 0x08
    IntAccelResult = 0x10
    IntGyroResult  = 0x20
    IntReserved1   = 0x40
    IntReserved2   = 0x80

    AlgoStandby              = 0x01
    AlgoRawData              = 0x02
    AlgoEulerData            = 0x04
    Algo6DOF                 = 0x08 # No magnetometer
    AlgoENUOutput            = 0x10 # Default is NED
    AlgoDisableGyroWhenStill = 0x20

    AlgoParamTransfer        = 0x80

    IntDefault  = 0x03 # Set the default behavior to interrupt on CPU Reset or Error. We will poll for Quaternions.
    AlgoDefault = AlgoENUOutput # Quaternion + Scaled data output in ENU frame is our default choice
    QRateDivDefault = 0x09 # By default, set the quaternion rate to 1/10 of the gyro rate

class MPU9250AccelParams(IntEnum):
    # Ref: https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/def.h (BSD 3-clause license)
    BW_250Hz = 0x00
    BW_184Hz = 0x01
    BW_92Hz  = 0x02
    BW_41Hz  = 0x03
    BW_20Hz  = 0x04
    BW_10Hz  = 0x05
    BW_5Hz   = 0x06
    BW_Max   = 0x07 # No filter, 3600 Hz

    FS_2g    = 0x02
    FS_4g    = 0x04
    FS_8g    = 0x08
    FS_16g   = 0x10

    BW_Default   = BW_41Hz
    FS_Default   = FS_8g
    Rate_Default = 1000 # samples/second

class MPU9250GyroParams(IntEnum):
    # Ref: https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/def.h (BSD 3-clause license)
    BW_250Hz   = 0x00
    BW_184Hz   = 0x01
    BW_92Hz    = 0x02
    BW_41Hz    = 0x03
    BW_20Hz    = 0x04
    BW_10Hz    = 0x05
    BW_5Hz     = 0x06
    BW_Max     = 0x07 # No filter, 3600 Hz

    FS_250dps  = 250
    FS_500dps  = 500
    FS_1000dps = 1000
    FS_2000dps = 2000

    BW_Default   = BW_41Hz
    FS_Default   = FS_1000dps
    Rate_Default = 1000 # samples/second


class MPU9250MagParams(IntEnum):
    # Ref: https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/def.h (BSD 3-clause license)
    Rate_8Hz   = 0x08
    Rate_100Hz = 0x64

    FS_1000uT  = 1000

    Rate_Default = Rate_100Hz
    FS_Default   = FS_1000uT

class BaroParams(IntEnum):
    # Ref: https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/def.h (BSD 3-clause license)
    Rate_Default = 25 # samples/second

# My board has MPU9250
AccelParams = MPU9250AccelParams
GyroParams  = MPU9250GyroParams
MagParams   = MPU9250MagParams

DEBUG = False

class I2CInterfaceWrapperBase:
    def __init__(self, dev, addr):
        """
        dev: The underlying interface that we are wrapping
        addr: I²C address of peripheral
        """
        self._dev = dev
        self._addr = addr

    def read(self, register, *args):
        if isinstance(register, Enum):
            register = register.value
        assert 0 <= register and register <= 255, f'Invalid register: {register}'
        if len(args) == 1 and type(args[0]) is int and args[0] > 1:
            if DEBUG:
                logger.info(f'Reading {args[0]} byte block from register {register} = 0x{register:02X}')
            return self._read_i2c_block_data(register, args[0])
        elif len(args) == 0 or (len(args) == 1 and type(args[0]) is int):
            if DEBUG:
                logger.info(f'Reading single byte from register {register} = 0x{register:02X}')
            return self._read_byte_data(register)
        else:
            raise AssertionError('Invalid arguments to _read')

    def write(self, register, data):
        if isinstance(register, Enum):
            register = register.value
        assert 0 <= register and register <= 255, f'Invalid register: {register}'
        if type(data) in (list, tuple, np.ndarray):
            for entry in data:
                assert 0 <= entry and entry <= 255, f'Not a list of bytes: {data}'
            if DEBUG:
                logger.info(f'Writing len(list(data)) bytes to register {register} = 0x{register:02X}. The bytes are {list(data)}')
            return self._write_i2c_block_data(register, list(data))
        else:
            assert 0 <= data and data <= 255, f'Not a byte: {data}'
            if DEBUG:
                logger.info(f'Writing single byte to register {register} = 0x{register:02X}. The byte is {data} = 0x{data:02X} = 0b{data:08b}')
            return self._write_byte_data(register, data)

    def _read_i2c_block_data(self, register, length: int, **kwargs):
        raise NotImplementedError('Derived class must implement `_read_i2c_block_data` method')

    def _write_i2c_block_data(self, register, data: list, **kwargs):
        raise NotImplementedError('Derived class must implement `_write_i2c_block_data` method')

    def _read_byte_data(self, register, **kwargs):
        raise NotImplementedError('Derived class must implement `_read_byte_data` method')

    def _write_byte_data(self, register, data: int, **kwargs):
        raise NotImplementedError('Derived class must implement `_write_byte_data` method')


class SMBusI2CInterfaceWrapper(I2CInterfaceWrapperBase):

    def _read_i2c_block_data(self, register, length: int):
        return self._dev.read_i2c_block_data(self._addr, register, length)

    def _write_i2c_block_data(self, register, data: list):
        return self._dev.write_i2c_block_data(self._addr, register, data)

    def _read_byte_data(self, register):
        return self._dev.read_byte_data(self._addr, register)

    def _write_byte_data(self, register, data: int):
        return self._dev.write_byte_data(self._addr, register, data)


class PyFTDII2CInterfaceWrapper(I2CInterfaceWrapperBase):

    def __init__(self, dev, addr):
        super().__init__(dev, addr)
        self._peripheral = self._dev.get_port(self._addr)

    def _read_i2c_block_data(self, register, length: int, **kwargs):
        time.sleep(0.005) # FIXME: This seems to be necessary because we have a fast processor?
        self._peripheral.flush()
        result = list(self._peripheral.read_from(register, length, **kwargs))
        self._peripheral.flush()
        return result

    def _write_i2c_block_data(self, register, data: list, **kwargs):
        time.sleep(0.005) # FIXME: This seems to be necessary because we have a fast processor?
        self._peripheral.flush()
        result = self._peripheral.write_to(register, bytes(data), **kwargs)
        self._peripheral.flush()
        return result

    def _read_byte_data(self, register, **kwargs):
        return int(self._read_i2c_block_data(register, 1, **kwargs)[0])

    def _write_byte_data(self, register, data: int, **kwargs):
        return self._write_i2c_block_data(register, bytes([data,]), **kwargs)

class USFSMotionSensor(MotionSensor):
    """ Motion Sensor based on the 'Ultimate Sensor Fusion Solution' from 'Pesky Products':
    https://www.tindie.com/products/onehorse/ultimate-sensor-fusion-solution-mpu9250/
    """

    _config = {}

    class NotConnectedError(Exception):
        def __init__(self):
            super().__init__("Not connected to motion sensor; call connect()")

    class NotRunningError(Exception):
        def __init__(self):
            super().__init__("SENtral MPU is not in 'running' state!")

    def initialize(self):
        """Perform the initialization rigmarole
        
        This code is largely based on
        https://github.com/kriswiner/EM7180_SENtral_sensor_hub/blob/master/EM7180_LSM6DSM_LIS2MDL_LPS22HB/USFS.cpp
        in particular, the method USFS::initEM7180,
        and
        https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/EM7180.cpp
        EM7180::initSensors() method.

        The license text on the former reads: "Library may be used freely and without limit with attribution."
        Attribution: Tlera Corporation, Kris Winer
        The license text on the latter is BSD 3-clause
        Attribute: Gregory Tomasch

        Recognized `params`:
        * acc_lpf_bw   : A value from AccelParams.BW_* setting low-pass filter bandwidth for the accelerometer
        * acc_fs       : Accelerometer full-scale range, a value from AccelParams.FS_*
        * acc_rate     : Accelerometer output data rate (ODR) in samples/second (rounded to the nearest multiple of 10)

        * gyro_lpf_bw  : A value from GyroParams.BW_* setting low-pass filter bandwidth for the gyroscope
        * gyro_fs      : Gyroscope full-scale range, a value from GyroParams.FS_*
        * gyro_rate    : Gyroscope output data rate (ODR) in samples/second (rounded to the nearest multiple of 10)

        * mag_fs       : A value from MagParams.FS_* setting the full-scale of the magnetometer
        * mag_rate     : A value from MagParams.Rate_* setting the rate of the magnetometer

        * baro_rate    : Sample rate for barometer

        * qratediv     : Quaternion ODR divisor: Ratio of Gyro ODR to Quaternion ODR minus 1 (See EM7180 datasheet)

        * algo_control : Algorithm control choice, from SENtralParams.Algo* (See EM7180 Datasheet: AlgorithmControl reg.)
        * int_control  : Interrupt choices, from SENtralParams.Int* (See EM7180 Datasheet: EnableEvents register)

        The default values for most parameters are placed in the Enums for readability, except for the ones below
        """

        self._initialized = False

        read = self._read
        write = self._write

        logger.info('Initializing SENtral MPU (EM7180)...')
        ctr = 0
        stat = read(EM7180.SentralStatus)
        while not bool(stat & 0x01):
            write(EM7180.ResetRequest, 0x01)
            time.sleep(0.5)
            ctr += 1
            stat = read(EM7180.SentralStatus)

            if ctr >= 100:
                raise TimeoutError(f'Failed to initialize SENtral MPU. The SentralStatus register reads: {stat:08b}')

        logger.info(f'SENtral Status: {stat:08b} binary (should be 0b00001011) = 0x{stat:02X} = {stat} decimal')

        write(EM7180.HostControl, 0x00) # set SENtral in initialized state to configure registers
        write(EM7180.PassThruControl, 0x00) # ensure pass through mode is off
        write(EM7180.HostControl, 0x01) # Force initialize
        time.sleep(0.02)

        logger.info(f'Initializing SENtral MPU with the following config:\n{self._config}')
        

        # Set LPF bandwidths (undocumented step, but necessary!)
        acc_lpf_bw = int(self._config.get('acc_lpf_bw', AccelParams.BW_Default))
        gyro_lpf_bw = int(self._config.get('gyro_lpf_bw', GyroParams.BW_Default))
        write(EM7180.ACC_LPF_BW, acc_lpf_bw)
        write(EM7180.GYRO_LPF_BW, gyro_lpf_bw)
        time.sleep(0.2)
        acc_lpf_bw_result = read(EM7180.ACC_LPF_BW)
        gyro_lpf_bw_result = read(EM7180.GYRO_LPF_BW)
        if acc_lpf_bw != acc_lpf_bw_result:
            raise ValueError(f'Accelerometer bandwidth setting {acc_lpf_bw} was not accepted. The register value was {acc_lpf_bw_result}')
        if gyro_lpf_bw != gyro_lpf_bw_result:
            raise ValueError(f'Gyroscope bandwidth setting {gyro_lpf_bw} was not accepted. The register value was {gyro_lpf_bw_result}')

        # Set ODRs
        acc_rate  = int(self._config.get('acc_rate', AccelParams.Rate_Default))/10.0
        gyro_rate = int(self._config.get('gyro_rate', GyroParams.Rate_Default))/10.0
        mag_rate  = int(self._config.get('mag_rate', MagParams.Rate_Default))
        baro_rate = int(self._config.get('baro_rate', BaroParams.Rate_Default) * 2)
        qratediv  = int(self._config.get('qratediv', SENtralParams.QRateDivDefault))

        if int(acc_rate) != acc_rate:
            logger.warning(f'Accelerometer output data rate will be truncated to nearest multiple of 10: {int(acc_rate)*10}')
        if int(gyro_rate) != gyro_rate:
            logger.warning(f'Gyroscope output data rate will be truncated to nearest multiple of 10: {int(gyro_rate)*10}')

        acc_rate = int(acc_rate)
        baro_rate = int(baro_rate)
        gyro_rate = int(gyro_rate)

        if qratediv < 0:
            logger.warning(f'Invalid value for Quaternion Rate Divisor. Resetting Quaternion ODR = Gyro ODR. See EM7180 datasheet for details.')
            qratediv = 0

        write(EM7180.QRateDivisor, qratediv)
        write(EM7180.MagRate, mag_rate)
        write(EM7180.AccelRate, acc_rate)
        write(EM7180.GyroRate, gyro_rate)

        # FIXME: Whereas Kris Winer's code says that the Baro ODR is
        # half the value written into the register, Gregory Tomasch's
        # code sets it to the exact value. Trusting the Kris Winer
        # code here.
        # TODO: Read the BMP280 datasheet and find the right behavior
        write(EM7180.BaroRate, 0x80 | baro_rate) # set enable bit and set baro rate

        # Configure operating mode
        algo_control = int(self._config.get('algo_control', SENtralParams.AlgoDefault))
        write(EM7180.AlgorithmControl, algo_control)
        write(EM7180.HostControl, 0x01)
        time.sleep(2.0)

        # Check for errors
        error = self._diagnose_errors()
        stat = read(EM7180.EventStatus)
        if bool(stat & 0x03):
            error = self._EM7180_EventStatus_error(stat) or error
        if error:
            raise RuntimeError(f'Motion coprocessor did not initialize. See console for error details.')

        # Read the full-scale ranges
        mag_fs, acc_fs = self._EM7180_get_mag_acc_fs()
        gyro_fs = self._EM7180_get_gyro_fs()

        logger.info(f'Default full-scale ranges are Mag: +/-{mag_fs} uT, Acc: +/-{acc_fs} g, Gyro: +/-{gyro_fs} dps')

        # Magic settings (undocumented)
        # c.f. https://github.com/gregtomasch/EM7180_SENtral_Calibration/issues/8
        # ref: https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/EM7180.cpp
        time.sleep(1.0)
        self._EM7180_set_param(EM7180.ParamStillness, 0x00) # Disable stillness mode
        self._EM7180_set_param(EM7180.ParamGBias, 0x01)     # Set GBias mode to 1 (I believe this is gyroscope zero error removal -- Akarsh)

        # Set the full-scale ranges
        acc_fs = int(self._config.get('acc_fs', AccelParams.FS_Default))
        gyro_fs = int(self._config.get('gyro_fs', GyroParams.FS_Default))
        mag_fs = int(self._config.get('mag_fs', MagParams.FS_Default))
        self._EM7180_set_mag_acc_fs(mag_fs, acc_fs)
        self._EM7180_set_gyro_fs(gyro_fs)
        
        # More magic settings (undocumented)
        # c.f. https://github.com/gregtomasch/EM7180_SENtral_Calibration/issues/8
        # ref: https://github.com/gregtomasch/EM7180_SENtral_Calibration/blob/master/ESP32_INV_USFS_Baseline_Calibration_Utility_Acc_WS/EM7180.cpp
        self._EM7180_set_param(EM7180.ParamTransientProt, 0.0) # Completely disable magnetic transient protection: "It's more work than is worth" according to above
        self._EM7180_set_param(EM7180.ParamGBias, 0x01)     # Set GBias mode to 1 (I believe this is gyroscope zero error removal -- Akarsh)
        
        int_control = int(self._config.get('int_control', SENtralParams.IntDefault))
        write(EM7180.EnableEvents, int_control)
        write(EM7180.HostControl, 0x01) # Set SENtral in normal run mode

        time.sleep(1.0) # Sleep 1s (may need to be adjusted)

        stat = read(EM7180.EventStatus)
        if bool(stat & 0x03):
            self._EM7180_EventStatus_error(stat)
            self._diagnose_errors()
            return False

        stat = read(EM7180.SentralStatus)
        ctr = 0
        while stat != 0x03:
            time.sleep(0.5)
            stat = read(EM7180.SentralStatus)
            ctr += 1
            if ctr > 10:
                logger.error(f'Failed to put the SENtral MPU in the right mode. SentralStatus register reads: {stat} = {stat:08b} binary. Expected decimal value 3')
                return False

        self._initialized = (not self._diagnose_errors())
        return self._initialized

    @property
    def config(self):
        return dict(self._config)

    def reconfigure(self, **new_config):
        """
        Reconfigure the USFS while running

        For the arguments, see initialize()
        Returns: True if the update was successful, False if not successful. See console for diagnostics.
        """

        self._initialized = False

        read = self._read
        write = self._write

        write(EM7180.HostControl, 0x01) # Ensure SENtral in normal run mode

        c = dict(new_config)
        for key, val in c.items():
            if key in self._config and self._config[key] == val:
                # Value hasn't changed, ignore
                del new_config[key]

        # Set LPF bandwidths (undocumented step, but necessary!)
        acc_lpf_bw = new_config.get('acc_lpf_bw', None)
        gyro_lpf_bw = new_config.get('gyro_lpf_bw', None)
        if acc_lpf_bw:
            write(EM7180.ACC_LPF_BW, acc_lpf_bw)
        if gyro_lpf_bw:
            write(EM7180.GYRO_LPF_BW, gyro_lpf_bw)

        # Set ODRs
        acc_rate = int(new_config.get('acc_rate', -10))/10.0
        gyro_rate = int(new_config.get('gyro_rate', -10))/10.0
        mag_rate = new_config.get('mag_rate', None)
        baro_rate = int(new_config.get('baro_rate', -1) * 2)
        qratediv = new_config.get('qratediv', None)

        if int(acc_rate) != acc_rate:
            logger.warning(f'Accelerometer output data rate will be truncated to nearest multiple of 10: {int(acc_rate)*10}')
        if int(gyro_rate) != gyro_rate:
            logger.warning(f'Gyroscope output data rate will be truncated to nearest multiple of 10: {int(gyro_rate)*10}')

        acc_rate = int(acc_rate)
        baro_rate = int(baro_rate)
        gyro_rate = int(gyro_rate)

        if qratediv and (qratediv < 0):
            logger.warning(f'Invalid value for Quaternion Rate Divisor. Resetting Quaternion ODR = Gyro ODR. See EM7180 datasheet for details.')
            qratediv = 0

        if qratediv:
            write(EM7180.QRateDivisor, qratediv)
        if mag_rate:
            mag_rate = int(mag_rate)
            write(EM7180.MagRate, mag_rate)
        if acc_rate >= 0:
            write(EM7180.AccelRate, acc_rate)
        if gyro_rate >= 0:
            write(EM7180.GyroRate, gyro_rate)

        # FIXME: Whereas Kris Winer's code says that the Baro ODR is
        # half the value written into the register, Gregory Tomasch's
        # code sets it to the exact value. Trusting the Kris Winer
        # code here.
        # TODO: Read the BMP280 datasheet and find the right behavior
        if baro_rate >= 0:
            write(EM7180.BaroRate, 0x80 | baro_rate) # set enable bit and set baro rate

        # Configure operating mode
        algo_control = new_config.get('algo_control', None)
        if algo_control:
            write(EM7180.AlgorithmControl, algo_control)
        write(EM7180.HostControl, 0x01)
        time.sleep(0.02)

        # Magic settings (undocumented)
        # Even though nothing has changed, we re-do it for good measure
        self._EM7180_set_param(EM7180.ParamStillness, 0x00) # Disable stillness mode
        self._EM7180_set_param(EM7180.ParamGBias, 0x01)     # Set GBias mode to 1

        # Set the full-scale ranges
        acc_fs = new_config.get(
            'acc_fs', self._config.get(
                'acc_fs', AccelParams.FS_Default
            )
        )
        mag_fs = new_config.get(
            'mag_fs', self._config.get(
                'mag_fs', MagParams.FS_Default
            )
        )
        self._EM7180_set_mag_acc_fs(mag_fs, acc_fs)
        gyro_fs = new_config.get('gyro_fs', None)
        if gyro_fs:
            self._EM7180_set_gyro_fs(gyro_fs)

        # More magic settings (undocumented)
        # Once again, we do them again for good measure
        self._EM7180_set_param(EM7180.ParamTransientProt, 0.0)

        int_control = new_config.get('int_control', None)
        if int_control:
            write(EM7180.EnableEvents, int_control)
        write(EM7180.HostControl, 0x01) # Set SENtral in normal run mode

        error = False
        stat = read(EM7180.EventStatus)
        if bool(stat & 0x03):
            self._EM7180_EventStatus_error(stat)
            error = True

        if not error:
            stat = read(EM7180.SentralStatus)
            ctr = 0
            while stat != 0x03:
                time.sleep(0.5)
                stat = read(EM7180.SentralStatus)
                ctr += 1
                if ctr > 10:
                    logger.error(f'Failed to put the SENtral MPU in the right mode. SentralStatus register reads: {stat} = {stat:08b} binary. Expected decimal value 3')
                    error = True
                    break

        error = (self._diagnose_errors() or error)
        if not error:
            self._config = dict(self._config, **new_config)
            logger.info(f'Successfully updated config!')

        self._initialized = (not error)
        return (not error)

    @property
    def running(self):
        if not self._initialized:
            return False
        if self._read(EM7180.SentralStatus) != 0x03:
            return False
        if not (self._read(EM7180.RunStatus) & 0x01):
            return False
        return True

    @property
    def connected(self):
        """ Override the base class method to also check for initialization """
        return self._initialized

    @staticmethod
    def _EM7180_SentralStatus_error(stat):
        """ Log error details and report True if error """
        if (stat == 0x06) or bool(stat & 0x04):
            logger.error(f'SENtral MPU failed to load data from EEPROM.')
            return True
        elif bool(stat & 0x10):
            logger.error(f'SENtral MPU failed to detect EEPROM!')
            return True
        else:
            logger.info(f'SENtral MPU reports status {stat:08b}:')
            if bool(stat & 0x01):
                logger.info('\tEEPROM detected')
            else:
                logger.info('\tEEPROM NOT detected (yet?)')
            if bool(stat & 0x02):
                logger.info('\tEEPROM upload done')
            else:
                logger.info('\tEEPROM upload NOT done')
            if bool(stat & 0x08):
                logger.info('\tMPU in unprogrammed or initialized (not running) state')
            else:
                logger.info('\tMPU not in initialized state, or is running normally')
            return False

    @staticmethod
    def _EM7180_EventStatus_error(stat):
        """ Log error details and report True if error """
        if not bool(stat & 0x03):
            logger.info(f'SENtral MPU reports no error')
            return False
        if stat & 0x02:
            logger.error(f'SENtral MPU reports error!')
            return True
        if stat & 0x01:
            logger.error(f'SENtral MPU reports CPUReset. This means that the config file failed to load!')
            return True

    @staticmethod
    def _EM7180_SensorStatus_error(stat):
        """ Log error details and report True if error """
        if stat == 0x00:
            logger.info(f'All sensors are okay!')
            return False
        if stat & 0x01:
            logger.error(f'NACK from Magnetometer')
        if stat & 0x02:
            logger.error(f'NACK from Accelerometer')
        if stat & 0x04:
            logger.error(f'NACK from Gyroscope')
        if stat & 0x08:
            logger.error(f'Unexpected Device ID from Magentometer')
        if stat & 0x10:
            logger.error(f'Unexpected Device ID from Accelerometer')
        if stat & 0x20:
            logger.error(f'Unexpected Device ID from Gyroscope')
        return True

    @staticmethod
    def _EM7180_ErrorRegister_error(stat):
        if stat == 0x00:
            logger.info(f'No error logged in the ErrorRegister')
            return False
        if stat & 0x80:
            logger.error(f'Invalid sample rate detected')
        if stat & 0x30:
            logger.error(f'Mathematical error')
        if stat & 0x21:
            logger.error(f'Magnetometer initialization failed')
        if stat & 0x22:
            logger.error(f'Accelerometer initialization failed')
        if stat & 0x24:
            logger.error(f'Gyroscope initialization failed')
        if stat & 0x11:
            logger.error(f'Magnetometer rate failure')
        if stat & 0x12:
            logger.error(f'Accelerometer rate failure')
        if stat & 0x14:
            logger.error(f'Gyroscope rate failure')
        return True

    def _diagnose_errors(self):
        """
        Return True if there was an error, False if there was no error
        Log the details of the diagnostics to the console
        """
        error = False
        error = error or self._EM7180_SentralStatus_error(self._read(EM7180.SentralStatus))
        error = error or self._EM7180_EventStatus_error(self._read(EM7180.EventStatus))
        for register in (EM7180.MagRate, EM7180.AccelRate, EM7180.GyroRate):
            if self._read(register) == 0x00:
                logger.error(f'ODR register {register} has lost its value!')
                error = True
        error = error or self._EM7180_SensorStatus_error(self._read(EM7180.SensorStatus))
        error = error or self._EM7180_ErrorRegister_error(self._read(EM7180.ErrorRegister))
        if error:
            logger.error('See above: Errors were detected =============')
        else:
            logger.info('No errors were detected =====================')
        return error
        
    def _EM7180_get_param(self, param: np.uint8, output_type=np.uint64):
        read = self._read
        write = self._write
        if isinstance(param, Enum):
            param = param.value
        if param <= 0 or param >= 128:
            raise ValueError(f'Parameter selection out of range: 0x{param:02X}')

        time.sleep(0.1)
        write(EM7180.ParamRequest, param)
        time.sleep(0.1)
        write(EM7180.AlgorithmControl, SENtralParams.AlgoParamTransfer) # Put the SENtral in parameter transfer mode
        time.sleep(0.1)

        stat = read(EM7180.ParamAcknowledge)
        ctr = 1
        while not (stat == param):
            time.sleep(0.1)
            stat = read(EM7180.ParamAcknowledge)
            ctr += 1
            if ctr % 4 == 0:
                write(EM7180.ParamRequest, param)
                time.sleep(0.1)
                write(EM7180.AlgorithmControl, SENtralParams.AlgoParamTransfer) # Put the SENtral in parameter transfer mode
            if ctr > 15:
                raise TimeoutError(
                    f'Did not get parameter acknowledge from SENtral MPU despite checking {ctr} times! '
                    f'Param (read) was 0x{param:02X} = {param}. Status is {stat}' 
                )

        raw = read(EM7180.SavedParamByte0, 4) # list of int bytes

        algo_control = int(self._config.get('algo_control', SENtralParams.AlgoDefault)) # Get the default algo control mode
        write(EM7180.ParamRequest, 0x00) # Write 0 to end parameter transfer process
        write(EM7180.AlgorithmControl, algo_control) # Resume business as usual

        if output_type in (list, tuple, bytes, bytearray):
            return output_type(raw)
        elif output_type is int:
            return struct.unpack('<i', bytearray(raw))[0]
        elif output_type is np.uint64:
            return np.uint64(struct.unpack('<I', bytearray(raw))[0])
        elif output_type is float:
            return struct.unpack('<f', bytearray(raw))[0]
        else:
            raise TypeError(f'Unhandled output type {output_type} requested in read for parameter {param}')
        

    def _EM7180_set_param(self, param: np.uint8, param_val: Union[np.int32, np.float32, list]): # pylint: disable=unsubscriptable-object
        # Ref: https://github.com/kriswiner/EM7180_SENtral_sensor_hub/blob/master/EM7180_LSM6DSM_LIS2MDL_LPS22HB/USFS.cpp
        # Attribution: Tlera Corporation, Kris Winer
        # Based on the method USFS::EM7180_set_integer_param
        if isinstance(param, Enum):
            param = param.value
        if param <= 0 or param >= 128:
            raise ValueError(f'Parameter selection out of range: 0x{param:02X} = {param}')
        if np.issubdtype(type(param_val), np.integer):
            value = list(struct.pack('<i', param_val))
        elif np.issubdtype(type(param_val), np.floating):
            param_val = np.float32(param_val) # WARNING: Truncation
            value = list(struct.pack('<f', param_val))
        elif type(param_val) in (list, tuple):
            for entry in param_val:
                if entry < 0 or entry > 255:
                    raise TypeError(f'Invalid entry {entry} in parameter value {param_val} to set param 0x{param:02X} = {param}')
            value = list(param_val)

        param = param | 0x80 # We must set the MSB to high to indicate a write to the param

        if len(value) > 4:
            raise ValueError(
                f'Invalid number of bytes to write for a single parameter 0x{param:02X}. Tried to write the value {value}'
            )

        while len(value) < 4:
            value.append(0x00) # 0-pad (c.f. e.g. USFS::EM7180_set_gyro_FS)

        time.sleep(0.1)
        read = self._read
        write = self._write
        LoadRegs = [
            EM7180.LoadParamByte0,
            EM7180.LoadParamByte1,
            EM7180.LoadParamByte2,
            EM7180.LoadParamByte3,
        ]
        for reg, val in zip(LoadRegs, value):
            write(reg, val)
        time.sleep(0.1)
        write(EM7180.ParamRequest, param)
        time.sleep(0.1)
        write(EM7180.AlgorithmControl, SENtralParams.AlgoParamTransfer) # Put the SENtral in parameter transfer mode

        stat = read(EM7180.ParamAcknowledge)
        ctr = 1
        while not (stat == param):
            logger.error(f'Did not get parameter acknowledge trying to write {value} to register 0x{param - 0x80:02X}. Checking again.')
            time.sleep(0.1)
            stat = read(EM7180.ParamAcknowledge)
            ctr += 1
            if ctr > 30:
                raise TimeoutError(
                    f'Did not get parameter acknowledge from SENtral MPU despite checking {ctr} times! '
                    f'Param (write) was 0x{param:02X} (normally 0x{param - 0x80:02X}) and the value was {value}; Status is {stat}'
                )

        algo_control = int(self._config.get('algo_control', SENtralParams.AlgoDefault)) # Get the default algo control mode
        write(EM7180.ParamRequest, 0x00) # Write 0 to end parameter transfer process
        write(EM7180.AlgorithmControl, algo_control) # Resume business as usual
        return True

    def _EM7180_get_gyro_fs(self) -> int:
        return self._EM7180_get_param(EM7180.ParamGyroFS, output_type=int)

    def _EM7180_get_mag_acc_fs(self) -> (int, int):
        res = self._EM7180_get_param(EM7180.ParamMagAccFS, output_type=list)
        mag_fs = struct.unpack('<H', bytearray(res[0:2]))[0]
        acc_fs = struct.unpack('<H', bytearray(res[2:4]))[0]
        return (mag_fs, acc_fs)

    def _EM7180_set_gyro_fs(self, gyro_fs: np.uint16):
        return self._EM7180_set_param(EM7180.ParamGyroFS, np.uint16(gyro_fs))

    def _EM7180_set_mag_acc_fs(self, mag_fs: np.uint16, acc_fs: np.uint16):
        param_val = list(struct.pack('<I', mag_fs))[:2]
        param_val.extend(list(struct.pack('<I', acc_fs))[:2])
        return self._EM7180_set_param(EM7180.ParamMagAccFS, param_val=param_val)

    @property
    def pressure(self):
        """
        Read pressure data from the Barometer
        """
        if not self._initialized:
            raise self.NotRunningError
        data = self._read(EM7180.Baro, 2)
        self._last_pressure = {
            'p': float(struct.unpack('<h', bytearray(data))[0])*0.01 + 1013.25, # mBar
            't': time.time(),
            'ut': datetime.datetime.utcnow(),
        }
        return self._last_pressure['p'] # mBar

    @property
    def temperature(self):
        """
        Read temperature data from the Barometer
        """
        if not self._initialized:
            raise self.NotRunningError
        data = self._read(EM7180.Temp, 2)
        self._last_temperature = {
            'T': float(struct.unpack('<h', bytearray(data))[0])*0.01, # degrees C
            't': time.time(),
            'ut': datetime.datetime.utcnow(),
        }
        return TimestampedTemperature(
            T=self._last_temperature['T'], # deg C
            ut=self._last_temperature['ut'],
        )

    def connect(self, port, addr=EM7180.ADDRESS, **params):
        """Set up and initialize the USFS

        port: Either i2c://<number> for dedicated I²C device at
        /dev/i2c-<number> (like in Raspberry Pi), or something starting
        with ftdi://, like 'ftdi://ftdi:232h:1/1' for using an FTDI USB
        GPIO chipset supported by pyftdi

        addr: The address of the I²C address of the EM7180 SENtral
        processor (typically 0x28).

        See _initialize() for recognized params.

        """
        self._addr = addr.value if isinstance(addr, Enum) else int(addr)
        if port.startswith('i2c://'):
            try:
                i2c_num = int(port[len('i2c://'):])
            except:
                raise ValueError(f'Invalid I²C device number after i2c:// in `{port}`')
            self._dev = SMBusI2CInterfaceWrapper(SMBus(i2c_num), self._addr)
        elif port.startswith('ftdi://'):
            i2c = pyftdi_i2c.I2cController()
            i2c.configure(port)
            self._dev = PyFTDII2CInterfaceWrapper(i2c, self._addr)
        else:
            raise ValueError(f'Invalid/unsupported port {port} for USFS motion sensor. Must be an i2c:// or ftdi:// port')

        if self._addr < 0 or self._addr > 127:
            raise ValueError(f'Invalid I2C address: 0x{self._addr:02X}')
        self._config = dict(params)
        if not self.initialize():
            raise RuntimeError('Failed to initialize USFS. Please see the console for errors.')

    def _read(self, *args, **kwargs):
        return self._dev.read(*args, **kwargs)

    def _write(self, *args, **kwargs):
        return self._dev.write(*args, **kwargs)

    def _raw_quaternion(self, **kwargs):
        if self._dev is None:
            raise self.NotConnectedError
        if not self._initialized:
            raise self.NotRunningError

        raw_quat = self._read(0x00, 16)
        t = time.time()
        ut = datetime.datetime.utcnow()

        quat_data = np.asarray([
            struct.unpack('<f', bytearray(raw_quat[i*4:(i+1)*4]))[0]
            for i in range(4)
        ], dtype=np.float64)

        if (quat_data**2).sum() < 0.1:
            raise ValueError(f'Obtained invalid quaternion data: {quat_data} with norm {(quat_data**2).sum()} < 0.1')

        q = Quaternion(quat_data).normalised

        return {
            'q': q,
            't': t,
            'ut': ut,
        }

    def _isr(self):
        raise NotImplementedError # FIXME: Implement
