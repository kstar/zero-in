import math
import logging
logger = logging.getLogger('EyepieceView')
from typing import Union

import urllib3
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import QPoint, QLineF

from coordinates import ICRS
from platesolve import ImageWithGnomonicSolution, PlateSolution
from dms import pretty_ra, pretty_dec

coordtype = Union[float, str] # pylint: disable=unsubscriptable-object

class DSSClient:

    def __init__(self, dss_url='http://localhost:8888/'):
        """
        dss_url: Is the URL for DSS that supports GET queries, usually: https://archive.stsci.edu/cgi-bin/dss_search
        """
        self._dss_url=dss_url

    def _get_cutout_gif(self, ra, dec, w, h) -> QPixmap:
        """ Get a DSS cutout in GIF format from the server
        w: width in arcmin
        h: height in arcmin
        returns a QPixmap
        """
        http = urllib3.PoolManager(maxsize=1, block=True)
        response = http.request('GET', '{}?r={}&d={}&f=gif&w={}&h={}'.format(self._dss_url, ra, dec, w, h))
        img_data = response.data
        response.release_conn()
        logger.info('Got {} bytes in response from DSS server'.format(len(img_data)))

        pixmap = QPixmap()
        pixmap.loadFromData(img_data)

        return pixmap

    def _get_cutout_fits(self, ra: coordtype, dec: coordtype, w: float, h: float) -> PlateSolution:
        """ Get a DSS cutout in FITS format from the server
        w: width in arcmin
        h: height in arcmin
        returns a PlateSolution object
        """
        http = urllib3.PoolManager(maxsize=1, block=True)
        response = http.request('GET', '{}?r={}&d={}&f=fits&w={}&h={}'.format(self._dss_url, ra, dec, w, h))
        img_data = response.data
        response.release_conn()
        logger.info('Got {} bytes in response from DSS server'.format(len(img_data)))

        return PlateSolution(img_data)


    def _arcminperpix(self, w_img, h_img, w_sky, h_sky):
        """ Assumes rectilinear lens (gnomonic projection) """
        # w_img = 2 * tan(w_sky/2.0) / radianperpix

        w_sky = math.radians(w_sky/60.0)
        h_sky = math.radians(h_sky/60.0)

        radianperpix_estimate_1 = 2 * math.tan(w_sky/2.0) / w_img
        radianperpix_estimate_2 = 2 * math.tan(h_sky/2.0) / h_img

        # FIXME: Weight below by image size
        radperpix = (radianperpix_estimate_1 + radianperpix_estimate_2)/2.0

        arcminperpix = math.degrees(radperpix) * 60.0

        return arcminperpix


    def _getPlateGIF(self, center: ICRS, plate_size: float) -> ImageWithGnomonicSolution:
        """
        center: an ICRS object indicating the coordinates of the center
        plate_size: plate width = height in arcminutes
        """

        pixmap = self._get_cutout_gif(center.ra, center.dec, plate_size, plate_size)

        W, H = pixmap.width(), pixmap.height()

        arcsecperpix = 60.0 * self._arcminperpix(
            W, H, plate_size, plate_size
        )

        logger.info(
            f'Fetched DSS plate of pixel size {W} x {H} (arcsec/pixel {arcsecperpix})'
        )

        return ImageWithGnomonicSolution(
            image=pixmap,
            center=center,
            scale=arcsecperpix,
            north_angle=math.nan # Unknown
        )

    def getPlate(self, center: ICRS, plate_size: float) -> PlateSolution:
        """
        center: an ICRS object indicating the coordinates of the center
        plate_size: plate width = height in arcminutes
        Returns: A FITS image in a PlateSolution object
        """

        result = self._get_cutout_fits(center.ra, center.dec, plate_size, plate_size)

        W, H = result.image_size

        logger.info(
            f'Fetched DSS plate at {pretty_ra(center.ra)} {pretty_dec(center.dec)} of pixel size {W} x {H} in FITS format'
        )

        return result
