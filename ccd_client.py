import io
import threading
import typing

import numpy as np
from astropy.io import fits
import PyIndi

from simpleindiclient import IndiClient, indi_connect, indi_wait
import timing

class CCDClient:
    def __init__(self, ccd_name, server_host="localhost", server_port=7624):

        self._blobEvent = threading.Event()
        self._exposure_lock = threading.RLock()

        self._indiclient = IndiClient(self._blobEvent)
        indi_connect(self._indiclient, server_host=server_host, server_port=server_port)

        self._ccd = indi_wait(lambda: self._indiclient.getDevice(ccd_name))

        ccd_connect_switch = indi_wait(lambda: self._ccd.getSwitch("CONNECTION"))

        if not self._ccd.isConnected():
            ccd_connect_switch[0].s = PyIndi.ISS_ON
            ccd_connect_switch[1].s = PyIndi.ISS_OFF
            self._indiclient.sendNewSwitch(ccd_connect_switch)

        self._ccd_exposure = indi_wait(lambda: self._ccd.getNumber("CCD_EXPOSURE"))

        self._indiclient.setBLOBMode(PyIndi.B_ALSO, ccd_name, "CCD1")

        self._ccd_ccd1 = indi_wait(lambda: self._ccd.getBLOB("CCD1"))

        self._blobEvent.clear()

    def expose(self, exp, timeout=15):
        """Synchronous capture that waits on PyINDI to return a CCD blob"""
        Timer = timing.makeOrGetTimingClass('Exposure')
        lock_timer = Timer('0:acquire_exposure_lock')
        lock_timer.__enter__()
        with self._exposure_lock:
            lock_timer.__exit__()
            with Timer('1:expose'):
                self._blobEvent.clear() # FIXME: Is this correct?
                self._ccd_exposure[0].value = exp
                self._indiclient.sendNewNumber(self._ccd_exposure)
            with Timer('2:read_blob'):
                result = self._blobEvent.wait(timeout=timeout)
                if not result:
                    raise TimeoutError('Timed out waiting for CCD to return data!')
                self._blobEvent.clear()
            if result:
                with Timer('3:get_image_data'):
                    return self.get_image_data()

            return None # Should not be reached because of raise

    def _get_results(self):
        return [
            {
                'info': {
                    'name': blob.name, 'size': blob.size, 'format': blob.format
                },
                'data': blob.getblobdata(),
            }
            for blob in self._ccd_ccd1
        ]

    @property
    def last_read_fits_image(self):
        return self._last_fits_image

    # FIXME: The following several methods are in serious need of refactoring
    def get_fits_image(self, auto_debayer=False):
        """Get the FITS image as bytes

        N.B. For efficiency, do not use auto_debayer if you will access
        the raw image data as a numpy array later. Instead directly call
        `get_image_data` with auto_debayer=True. Only supply
        `auto_debayer=True` if you are going to write the output back to
        FITS anyway.
        """

        results = self._get_results()
        assert len(results) == 1, results
        info = results[0]['info']
        assert info['name'].lower() == 'CCD1'.lower(), info['name']
        #assert info['format'] == '.fits', info['format']
        self._last_fits_image = bytes(results[0]['data'])

        fitsdata = results[0]['data']

        if not auto_debayer:
            return fitsdata
        else:
            return Debayer.debayer_fits_image(fitsdata)


    def write_fits_image(self, path, auto_debayer=False):
        with open(path, 'wb') as fd:
            fd.write(
                self.get_fits_image(auto_debayer=auto_debayer)
            )

    def get_image_data(self, auto_debayer=False):
        """Read FITS with astropy.io.fits to get numpy array"""
        return Debayer.fits_image_to_data(
            self.get_fits_image(),
            auto_debayer=auto_debayer
        )

class Debayer: # Namespace

    @staticmethod
    def fits_image_to_data(fits_image: bytes, auto_debayer: bool, return_pattern=False):
        """Read FITS with astropy.io.fits to get numpy array"""
        hdu = fits.open(io.BytesIO(fits_image))[0]
        image = hdu.data
        if auto_debayer:
            pattern = hdu.header.get('BAYERPAT', None) # pylint: disable=no-member
            if pattern is not None:
                image = Debayer.debayer(image, pattern)
        else:
            pattern = None

        if return_pattern:
            return image, pattern
        else:
            return image


    @staticmethod
    def debayer_fits_image(fits_image: bytes, return_pattern=False) -> bytes:
        """
        fits_image: bytes
        Returns a debayered version of the fits image as bytes

        Uses astropy for FITS processing
        """
        # Debayer
        Timer = timing.makeOrGetTimingClass('Debayer')
        with Timer('debayer_fits'):
            hdul = fits.open(io.BytesIO(fits_image))
            image = hdul[0].data
            pattern = hdul[0].header.pop('BAYERPAT', None)  # pylint: disable=no-member
            if pattern is not None:
                hdul[0].data = Debayer.debayer(image, pattern)
            out = io.BytesIO()
            hdul[0].writeto(out)
            if not return_pattern:
                return out.getvalue()
            else:
                return out.getvalue(), pattern

    # The actual debayering method!
    @staticmethod
    def debayer(image: np.ndarray, pattern: str) -> np.ndarray:
        """
        Debayer the given image
        image: floating point array of 2 dimensions (both lengths should be even)
        pattern: Bayer mask pattern as a string (eg: RGGB)

        Return: array of 3 dimensions, with channel order R G B in the last dimension.
        """
        pattern = pattern.upper()
        if sorted(pattern) != ['B', 'G', 'G', 'R']:
            raise ValueError(f'Debayer: Invalid pattern: {pattern}')
        if image.ndim != 2:
            raise ValueError(f'Debayer: Invalid image shape {image.shape}')
        H0, W0 = image.shape
        if H0 % 2 != 0 or W0 % 2 != 0:
            raise ValueError(f'Debayer: Invalid image shape {image.shape}')

        r = pattern.index('R')
        g1 = pattern.index('G')
        g2 = pattern[g1+1:].index('G') + g1 + 1
        b = pattern.index('B')

        H, W = H0//2, W0//2

        image = image.reshape(H, 2, W, 2).transpose([0, 2, 1, 3]).reshape(H, W, 4)
        R = image[..., r]
        G = (image[..., g1] + image[..., g2])/2.0
        B = image[..., b]

        return np.stack([R, G, B], axis=-1) # Debayered image in RGB order
