import math
import logging
logger = logging.getLogger('EyepieceView')

import urllib3
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import QPoint, QLineF

class EyepieceView:

    def __init__(self, dss_url='http://localhost:8888/'):
        """
        dss_url: Is the URL for DSS that supports GET queries, usually: https://archive.stsci.edu/cgi-bin/dss_search
        """
        self._dss_url=dss_url

    def _get_cutout(self, ra, dec, w, h):
        """ Get a DSS cutout in GIF format from the server
        w: width in arcmin
        h: height in arcmin
        returns a QPixmap
        """
        http = urllib3.PoolManager()
        response = http.request('GET', '{}?r={}&d={}&f=gif&w={}&h={}'.format(self._dss_url, ra, dec, w, h))
        img_data = response.data
        logger.info('Got {} bytes in response from DSS server'.format(len(img_data)))

        pixmap = QPixmap()
        pixmap.loadFromData(img_data)

        return pixmap

    def _estimate_arcminperpix(self, w_img, h_img, w_sky, h_sky):
        assert w_img > 0 and h_img > 0, (w_img, h_img)
        return ((w_sky/w_img) + (h_sky/h_img))/2.0


    def get_eyepiece_view(self, ra, dec, lat, alt, az, fov, focuser_angle, plate_fov=1.3, target_ra=None, target_dec=None, draw_fov=True):
        """
        All angles are in degrees unless specified
        ra: Center RA (ICRS)
        dec: Center Dec (ICRS)
        lat: Latitude of the location
        alt: Center Altitude
        az: Center Azmiuth
        fov: Field of view of eyepiece in arcmin
        focuser_angle: Is the angle by which the focuser is offset from perpendicular.
        plate_fov: Plate FOV as a multiple of eyepiece FOV -- this is to fetch a little extra around the eyepiece FOV
        target_ra: RA of the target (if you need this plotted) (ICRS)
        target_dec: Dec of the target (if you need this plotted) (ICRS)
        draw_fov: Draw the FOV circle if set to True

        Returns a tuple containing:
        * The rendered eyepiece image
        * A eyepiece view metadata dictionary that must be supplied for other tasks
        """

        plate_size = plate_fov * fov
        pixmap = self._get_cutout(ra, dec, plate_size, plate_size)

        arcminperpix = self._estimate_arcminperpix(
            pixmap.width(), pixmap.height(), plate_size, plate_size
        )

        W, H = pixmap.width(), pixmap.height()
        def _coords_to_pix(ra_, dec_):
            """N.B. Coordinates returned are referenced to center of the pixmap!"""
            # Difference in RA and Dec in arcmin
            dRA = (ra_ - ra) * 60.0
            dDec = (dec_ - dec) * 60.0

            # Eastern RA > Western RA, hence - sign on dx
            # Increasing dec upwards, whereas computer convention of y axis is downwards, hence - sign on dy
            dx = -(dRA / arcminperpix) * math.cos(math.radians(dec))
            dy = -(dDec / arcminperpix)

            x = dx
            y = dy
            return x, y

        # Convert target ra, dec to pixels
        if target_ra is not None and target_dec is not None:
            plot_target = True
            tx, ty = _coords_to_pix(target_ra, target_dec)
            if abs(tx) > W/2 or abs(ty) > H/2:
                target_as_angle = True
            else:
                target_as_angle = False
        else:
            plot_target = False


        # Convert fov circle size to pixels
        logger.info('Plate size: {} x {}'.format(pixmap.width(), pixmap.height()))
        rx, ry = (pixmap.width()/plate_fov)/2.0, (pixmap.height()/plate_fov)/2.0
        r = int((rx + ry)/2.0) # FIXME: int
        logger.info('DEBUG: r = {}, rx = {}, ry = {}'.format(r, rx, ry))

        # Paint annotations on the image
        # N.B. Can't paint on a QPixmap in a non-GUI thread. Need to paint on QImage.
        image = pixmap.toImage()
        painter = QPainter(image)
        painter.translate(W/2, H/2)
        painter.scale(1, -1) # The coordinate system is now in usual math conventions
        pen = QPen()
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)

        # FOV circle
        if draw_fov:
            painter.drawEllipse(QPoint(0, 0), int(rx), int(ry)) # FIXME: int

        # Target
        if plot_target:
            if target_as_angle:
                painter.save()
                painter.rotate(-math.degrees(math.atan2(ty, tx))) # Qt: CCW -ve
                painter.drawLine(QLineF(0.90 * r, 0.000 * r, 1.0 * r, 0))
                painter.drawLine(QLineF(0.97 * r, +0.01 * r, 1.0 * r, 0))
                painter.drawLine(QLineF(0.97 * r, -0.01 * r, 1.0 * r, 0))
                painter.restore()
            else:
                painter.save()
                painter.translate(tx, ty)
                painter.drawLine(QLineF(0.03 * r, 0.0, 0.05 * r, 0.0))
                painter.drawLine(QLineF(0.0, 0.03 * r, 0.0, 0.05 * r))
                painter.restore()


        pixmap = QPixmap.fromImage(image)


        # Now we compute the key bit of information: the angle!
        sinlat = math.sin(math.radians(lat))
        sindec = math.sin(math.radians(dec))
        cosdec = math.cos(math.radians(dec))
        sinalt = math.sin(math.radians(alt))
        cosalt = math.cos(math.radians(alt))
        sinaz = math.sin(math.radians(az))
        cosNorthAngle = (sinlat - sinalt * sindec) / (cosalt * cosdec)
        northAngle = math.degrees(math.acos(cosNorthAngle))
        if sinaz > 0:
            northAngle = -northAngle

        # Clockwise rotation
        rotation = northAngle - alt + 180 - focuser_angle

        # sin(theta) + cos(theta)
        S = max(W, H) # Really W = H, so this is just overkill
        S_ext = int(
                S * (abs(math.sin(math.radians(rotation))) + abs(math.cos(math.radians(rotation))))
        )

        logger.info('DEBUG: Rendering pixmap has side {}'.format(S_ext))
        # We paint the pixmap with the required rotation on a bigger pixmap
        render_image = QImage(S_ext, S_ext, QImage.Format_RGB32)
        render_image.fill(QColor(0, 0, 0))
        painter = QPainter(render_image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.translate(S_ext / 2, S_ext / 2)
        painter.rotate(rotation)
        painter.drawPixmap(-W/2, -H/2, W, H, pixmap, 0, 0, W, H)

        metadata = {
            'center': (ra, dec),
            'arcminperpix': arcminperpix,
            'rotation': rotation
        }

        return QPixmap.fromImage(render_image), metadata

    def plate_pixel_to_icrs(self, metadata, x, y):
        raise NotImplementedError('TODO')

    def icrs_to_plate_pixel(self, metadata, ra, dec):
        raise NotImplementedError('TODO')
