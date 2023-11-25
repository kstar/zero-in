import math
import numpy as np
import logging
logger = logging.getLogger('EyepieceImageViewer')
from collections import OrderedDict
import time

from PyQt5.QtGui import QTransform, QPainterPath, QPen, QColor, QPixmap, QBrush, QFontMetricsF, QFont, QIcon
from PyQt5.QtCore import Qt, QPointF, QPoint, QRectF, pyqtSignal
from PyQt5.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QLabel

from imageviewer import ImageViewer
from coordinates import ICRS, angularDistance
from platesolve import PlateSolution
from error_handler import error_handler
from dms import pretty_ra, pretty_dec, pretty_icrs, pretty_short

class EyepieceImageViewer(ImageViewer):
    """N.B. There are four coordinate systems:

    1) ICRS coordinates (astronomical coordinates)

    2) Plate coordinates -- the plate center is (0, 0) and north is
    along the -ve y-axis, so that y = -plate_height/2.0 is the north
    edge of the plate

    3) Scene coordinates -- the plate is rotated by the eyepiece
    rotation amount to render onto the scene. This is the coordinate
    system used by the QGraphicsScene (self._scene). Here (0, 0) is at
    the top left of the scene, and the plate center is
    (plate_width/2.0, plate_height/2.0). The positive y-direction is
    downwards. The plate is rotated so that north is at an angle of
    self._rotation (clockwise positive).

    4) View coordinates -- these are the coordinates used for the
    rendering in QGraphicsView. We will hardly deal in view
    coordinates directly, and use mapFromScene and mapToScene to map
    back and forth between these coordinate systems as the first /
    last step in any conversion where they are needed.
    """

    clickedPointChanged = pyqtSignal(ICRS, name='clickedPointChanged')

    def __init__(self, parent):
        super().__init__(parent)
        self._target = None # ICRS
        self._transform = None # Transform that puts (0, 0) at the image center on the scene
        self._annotations = {}
        self._plate = None
        self._red_pen = QPen(QColor(255, 0, 0))
        self._size = None
        self._clicked_point = None # clicked point in scene coordinates
        self._fov = None # FOV circle size in arcmin (None for no circle)
        self._first_pixmap = True
        self._target_on_plate = None
        self._show_annotations = True
        self._labelWidgets = []
        self.addDefaultButtons()
        self.addButton('toggle_annotations', QIcon('annotations.png'), self.toggleAnnotations)
        self.addButton('popup_menu', QIcon('menu.png'), self._popupMenu)

    def setTarget(self, target_pos=None):
        assert isinstance(target_pos, (ICRS, type(None))), (target_pos, type(target_pos))
        self._target = target_pos
        self._updateTarget()

    def setFovCircle(self, fov=None):
        """
        fov: arcminutes, or False / None to unset
        """
        if fov == False:
            fov = None
        self._fov = fov
        self._updateFovCircle()
        self._updateTarget()

    def updateEyepieceView(self, plate, scope_pos: ICRS, rotation):
        """
        plate: A PlateSolution object containing the plate and solution metadata
        scope_pos: Scope position on the plate
        rotation: The amount to rotate the plate (to simulate the eyepiece view) in degrees
        """
        if not isinstance(plate, PlateSolution) and (plate != None):
            raise TypeError('Expected a PlateSolution object, got {}'.format(type(plate)))
        if plate is None and self._plate is None:
            raise ValueError('Called updateEyepieceView with None for the plate argument, but there is no prior plate to reuse')
        if plate is not None:
            self._plate = plate
            self._pixmap = QPixmap.fromImage(self._plate.as_qimage())

        if rotation is None:
            error_handler('Could not determine eyepiece rotation for this plate. The orientation is arbitrary.')
            rotation = 0

        self._rotation = rotation
        self._scope_pos = scope_pos # ICRS
        self._scope_pos_plate = self.mapFromICRSToPlate(self._scope_pos) # QPointF

        # N.B. To avoid any bugs from trying to mix the panning /
        # zooming transformations with the field rotation, we will
        # keep th transforms separate. So the "scene" will already
        # contain the rotated image. To map clicks and so on, we will
        # provide analogs to mapToScene() and mapFromScene() that take
        # the rotation and pixel scale of the plate into account
        w, h = self._pixmap.width(), self._pixmap.height()

        # This is the transform without the translation, which can be
        # used to map raw plate coordinates, but centered at the plate
        # center, easily to scene coordinates
        self._transform = QTransform.fromTranslate(+w/2.0, +h/2.0).rotate(
            self._rotation # N.B. rotation is +ve CW (Qt convention, not math convention)
        )

        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self._scene.clear()
        self._annotations = {} # therefore we must also clear the annotation dict

        self._pixmap_item = QGraphicsPixmapItem(self._pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._pixmap_item.setTransformOriginPoint(w/2.0, h/2.0)
        self._pixmap_item.setRotation(self._rotation)
        self._scene.addItem(self._pixmap_item)

        # ### DEBUG ###
        # axes = QPainterPath()
        # axes.moveTo(0, 0)
        # axes.lineTo(0, -300)
        # axes.moveTo(0, 0)
        # axes.lineTo(100, 0)
        # blue_pen = QPen()
        # green_pen = QPen()
        # blue_pen.setColor(QColor(0, 0, 255))
        # green_pen.setColor(QColor(0, 255, 0))
        # self._scene.addPath(self._transform.map(axes), self._red_pen)

        self._size = min(self._pixmap.width(), self._pixmap.height())
        self._red_pen.setWidth(int(max(0.001 * self._size, 1)))

        logger.info('Pixmap added to QGraphicsScene. Now adding annotations')
        if self._first_pixmap:
            self.resetTransform()
            self.fitInView()
        self._updateAnnotations()
        self._first_pixmap = False

    def fitInView(self, *args, **kwargs):
        if (len(args) == 0 or (len(args) == 1 and type(args[0]) is bool)) and len(kwargs) == 0:
            return super().fitInView(
                self._transform.mapRect(QRectF(0, 0, self._pixmap.width(), self._pixmap.height())),
                Qt.KeepAspectRatio
            )
        else:
            super().fitInView(*args, **kwargs)


    def debug_embed(self):
        from IPython import embed
        embed(header='Debug embed in EyepieceImageViewer')

    def mapFromPlateToICRS_LEGACY(self, p_plate):
        assert False, 'Legacy code'
        if self._plate is None:
            raise RuntimeError('Tried to map coordinates when no plate has been set')
        if isinstance(p_plate, QPoint):
            p_plate = QPointF(p_plate.x(), p_plate.y())
        else:
            assert isinstance(p_plate, QPointF)

        # N.B. The - sign on the y-axis is because of computer
        # graphics convention vs math convention. dy, dy_arcsec
        # etc. are in math convention where Y-axis is +ve upwards

        # Rough algorithm: planar approximation of the sphere
        dx_arcsec, dy_arcsec = p_plate.x() * self._plate.scale, -p_plate.y() * self._plate.scale
        dx, dy = math.radians(dx_arcsec/3600.0), math.radians(dy_arcsec/3600.0)
        D = math.sqrt(dx**2 + dy**2) # D is in radians -- an approximation
        zeta = math.atan2(dx, dy)
        logger.debug(f'Approximate computation (radians): D = {D}, zeta = {zeta}')

        # More accurate: Use gnomonic projection
        radperpix = math.radians(self._plate.scale / 3600.0)
        D = math.atan(radperpix * math.sqrt(p_plate.x() ** 2 + p_plate.y() ** 2)) # D in radians
        zeta = math.atan2(p_plate.x(), -p_plate.y())
        logger.debug(f'Accurate computation (radians): D = {D}, zeta = {zeta}')

        # TO FIND THE DECLINATION OF THE POINT
        #
        # Let (α₀, δ₀) denote the center coordinates of the plate, and
        # let (α, δ) denote the ICRS corresponding to p_plate.
        # Let Δ denote the angular distance between the center and p_plate.
        #
        # We observe that the vertical line through the center of the
        # plate (i.e. the Y-axis) corresponds exactly to a meridian of
        # RA (or two, if the NCP lies in the plate)
        #
        # From applying the spherical cosine rule to the triangle
        # containing plate center, the point p_plate and NCP, one
        # obtains
        #
        # [1] sin(δ) = sin(δ₀)cos(Δ) + cos(δ₀)sin(Δ)cos(ζ)
        #
        # where ζ is the angle made by the arc Δ with the Y
        # (declination) axis.
        #
        # By approximating the small arc Δ by a straight line segment,
        # and applying the small angle approximations, we have
        #
        # [2] sin(δ) ≈ sin(δ₀)cos(Δ) + cos(δ₀) Δy
        #
        # where Δy is the plate coordinate difference along the
        # y-axis, converted into an angle
        #
        # When δ₀ and δ are close to ±90°, the arcsine is going to be
        # very insensitive, so as usual, it is preferable to use a
        # Haversine version. We go back to [1] and re-write sin(δ) as
        # cos(90° - δ) etc. and convert to Haversine to obtain:
        #
        # [3] hv(90°-δ) = hv(90° - δ - Δ) + hv(ζ) cos(δ₀) sin(Δ)
        #
        # which we can use when the plate is close to either pole.

        delta0 = math.radians(self._plate.center.dec)
        alpha0 = math.radians(self._plate.center.ra)

        logger.debug("DSS plate center is ICRS({}, {})".format(
            pretty_ra(self._plate.center.ra), pretty_dec(self._plate.center.dec))
        )

        use_approximate = False

        if abs(self._plate.center.dec) < 85: # Use simple version
            sin_delta0, cos_delta0 = math.sin(delta0), math.cos(delta0)
            cos_D = math.cos(D)
            sin_D = math.sin(D)
            if use_approximate:
                dx_arcsec, dy_arcsec = p_plate.x() * self._plate.scale, -p_plate.y() * self._plate.scale
                dx, dy = math.radians(dx_arcsec/3600.0), math.radians(dy_arcsec/3600.0)
                sin_delta = sin_delta0 * cos_D + cos_delta0 * dy
            else:
                cos_zeta = math.cos(zeta)
                sin_delta = sin_delta0 * cos_D + cos_delta0 * sin_D * cos_zeta

            delta = math.degrees(math.asin(sin_delta))

        else: # Use haversine version
            sin_D = math.sin(D)
            cos_delta0 = math.cos(delta0)
            hv_arg1 = math.sin(((math.pi/2.0) - delta0 - D)/2.0)**2
            hv_zeta = math.sin(zeta/2.0)**2

            delta = 90 - math.degrees(
                2 * math.asin(math.sqrt(hv_arg1 + hv_zeta * cos_delta0 * sin_D))
            )

        # TO FIND THE RIGHT ASCENSION OF THE POINT
        #
        # We apply the spherical law of sines to the same triangle and
        # apply the same small-angle approximation on Δ and use
        #
        # sin(Δ) sin(ζ) ≈ Δx
        #
        # and obtain (noting that left side of the plate is West and
        # RA increases going East)
        #
        # sin(α - α₀) = -Δx / cos(δ)
        #
        # Note that when δ = 90°, RA is undefined, so this is okay.
        # Also note that in the case where δ₀ = 90°, D ≈ (90° - δ), whereby
        #
        # cos(δ) ≈ cos(90° - D) = sin(D) ≈ D = sqrt(Δx² + Δy²) >= Δx
        #
        # therefore, if (-Δx / cos(δ)) exceeds 1, it is only because
        # of numerical error, and we can declare the RA undefined.
        cos_delta = math.cos(math.radians(delta))

        if cos_delta < 1e-10:
            dalpha = math.nan
        else:

            if use_approximate:
                sin_dalpha = -dx / cos_delta
            else:
                sin_zeta = math.sin(zeta)
                sin_dalpha = -sin_D * sin_zeta / cos_delta

            if sin_dalpha >= 1.0:
                dalpha = math.nan
            else:
                dalpha = math.asin(sin_dalpha)
        alpha = math.degrees(alpha0 + dalpha) % 360.0

        logger.debug(f'RA = {alpha}')

        assert alpha is not None
        assert delta is not None

        return ICRS(ra=alpha, dec=delta)

    def mapFromPlateToICRS(self, p_plate):
        if self._plate is None:
            raise RuntimeError('Tried to map coordinates when no plate has been set')
        if isinstance(p_plate, QPoint):
            p_plate = QPointF(p_plate.x(), p_plate.y())
        else:
            assert isinstance(p_plate, QPointF)

        ra, dec = self._plate.to_radec(p_plate.x(), p_plate.y(), relative_to='center_ydown')
        return ICRS(ra=ra, dec=dec)

    def mapFromSceneToICRS(self, p_scene):
        assert self._transform is not None
        if isinstance(p_scene, QPoint):
            p_scene = QPointF(p_scene.x(), p_scene.y())
        assert isinstance(p_scene, QPointF), type(p_scene)

        assert self._transform.isInvertible()
        return self.mapFromPlateToICRS(
            self._transform.inverted()[0].map(p_scene)
        )


    def mapFromICRSToPlate_LEGACY(self, p):
        """
        p: ICRS

        Note: The math for this method is explained in mapFromPlateToICRS
        """
        assert False, 'Legacy code'

        if self._plate is None:
            raise RuntimeError('Tried to map coordinates when no plate has been set')
        assert isinstance(p, ICRS)

        # Difference in RA and Dec in radian
        dRA = math.radians(p.ra - self._plate.center.ra)
        dDec = math.radians(p.dec - self._plate.center.dec)

        hvsin_dRA = (math.sin(dRA / 2.0) ** 2)
        hvsin_dDec = (math.sin(dDec / 2.0) ** 2)
        cos_delta0 = math.cos(math.radians(self._plate.center.dec))
        cos_delta = math.cos(math.radians(p.dec))
        sin_delta0 = math.sin(math.radians(self._plate.center.dec))
        sin_delta = math.sin(math.radians(p.dec))

        # Haversine formula to find angular distance Δ in radians
        D = 2 * math.asin(math.sqrt(
            hvsin_dDec + cos_delta0 * cos_delta * hvsin_dRA
        ))

        # See mapFromSceneToICRS for how the formula was arrived at
        dX = -math.sin(dRA) * math.cos(math.radians(p.dec)) # dX in radians
        logger.debug(f'D = {D}, dX = {dX}. Computing dY')
        dY_nosign = math.sqrt(D * D - dX * dX) # dY in radians
        logger.debug(f'dY without sign = {dY_nosign}')

        # dY has the same sign as sin(delta) - sin(delta_0) cos(D) except when delta_0 == 0
        if abs(self._plate.center.dec) < 89.99:
            if sin_delta < sin_delta0 * math.cos(D):
                dY = -dY_nosign
            else:
                dY = dY_nosign
        else:
            # Pretend that the pole is in the center of the
            # plate. Empirically verified that 0h lies to the left,
            # and 6h lies to the top.
            if (p.ra % 360.0) <= 180.0:
                dY = dY_nosign
            else:
                dY = -dY_nosign

        # dX and dY are in the "math" coordinate system, in units of radians
        # dx and dy are in computer coordinate system, in units of pixels
        dx = math.degrees(dX) * 3600.0 / self._plate.scale
        dy = -math.degrees(dY) * 3600.0 / self._plate.scale

        p_plate = QPointF(dx, dy)

        return p_plate

    def mapFromICRSToPlate(self, p: ICRS) -> QPointF:
        if self._plate is None:
            raise RuntimeError('Tried to map coordinates when no plate has been set')
        assert isinstance(p, ICRS)

        return QPointF(*self._plate.to_pixels(p.ra, p.dec, relative_to='center_ydown'))


    def mapFromICRSToScene(self, p):
        p_plate = self.mapFromICRSToPlate(p)

        assert self._transform is not None
        p_scene = self._transform.map(p_plate)

        return p_scene

    def toggleAnnotations(self):
        self._show_annotations = not self._show_annotations
        try:
            self._guideLabel.setVisible(self._show_annotations)
        except AttributeError:
            pass
        self._updateAnnotations()

    def _clearAnnotations(self):
        for key in list(self._annotations.keys()):
            self._scene.removeItem(self._annotations[key])
            del self._annotations[key]

    def _updateAnnotations(self):
        if not self._show_annotations:
            self._clearAnnotations()
            return

        self._updateFovCircle()
        self._updateTarget()
        self._updateNorthArrow()

        # N.B. updateClickedPoint does not recompute the ICRS values and
        # displays stale coords
        if self._clicked_point is not None:
            self._setClickedPoint(self._clicked_point)

    def _updateFovCircle(self):
        if not self._show_annotations:
            self._clearAnnotations()
            return

        if self._plate is None:
            logger.error('Trying to update FOV circle when plate is not available')
            return

        if 'fov' in self._annotations:
            self._scene.removeItem(self._annotations['fov'])
            del self._annotations['fov']

        if self._fov is None:
            return

        fov_pen = QPen(self._red_pen)
        fov_pen.setWidth(int(2.0 * self._red_pen.width()))

        diaPixels = (self._fov * 60) / self._plate.compute_scale()
        fovcirc = QPainterPath()
        fovcirc.addEllipse(self._transform.map(self._scope_pos_plate), diaPixels/2.0, diaPixels/2.0)
        self._annotations['fov'] = self._scene.addPath(fovcirc, fov_pen)
        logger.debug(f'diaPixels {diaPixels}')


    def _cross_hairs(self, gap, size, path=None):
        ch = QPainterPath() if path is None else path
        ch.moveTo(0, -gap)
        ch.lineTo(0, -size - gap)
        ch.moveTo(-gap, 0)
        ch.lineTo(-size - gap, 0)
        return ch

    def _full_cross_hairs(self, gap, size):
        fch = self._cross_hairs(gap, size)
        fch = QTransform().rotate(180).map(fch)
        return self._cross_hairs(gap, size, path=fch)

    def _circle_cross_hairs(self, radius, size):
        cch = self._full_cross_hairs(radius, size)
        cch.addEllipse(QPoint(0, 0), radius, radius)
        return cch

    def _vertical_arrow_head(self, height, width_ratio=0.5):
        ah = QPainterPath()
        ah.moveTo(0, 0)
        ah.lineTo(-width_ratio * height / 2.0, height)
        ah.lineTo(+width_ratio * height / 2.0, height)
        ah.lineTo(0, 0)
        return ah

    def _updateTarget(self):
        if not self._show_annotations:
            self._clearAnnotations()
            return

        if not self._plate:
            logger.error('Trying to update target point when plate is not available')
            return

        if 'target' in self._annotations:
            self._scene.removeItem(self._annotations['target'])
            del self._annotations['target']

        if self._target is None:
            self._target_on_plate = None
            return

        # Check if the target is on the plate
        tp = self.mapFromICRSToPlate(self._target)
        w, h = self._pixmap.width(), self._pixmap.height()
        if math.isnan(tp.x()) or math.isnan(tp.y()):
            logger.error(f'Target point {pretty_icrs(self._target)} cannot be mapped to plate (is it too far?)')
            self._target_on_plate = None
            return

        logger.debug('Target point coordinates {} {}'.format(tp.x(), tp.y()))
        if abs(tp.x()) > w/2.0 or abs(tp.y()) > h/2.0:
            target_on_plate = False
        else:
            target_on_plate = True

        logger.info('Target point {} is {}on the plate'.format(
            pretty_icrs(self._target),
            '' if target_on_plate else 'not '
        ))

        target_pen = QPen(self._red_pen)

        self._target_on_plate = target_on_plate

        if target_on_plate:
            # We want to draw cross-hairs, but we want them to be
            # "vertical" and not rotated
            target_pen.setStyle(Qt.DashLine)
            target_pen.setWidth(int(2 * target_pen.width()))
            ch = self._circle_cross_hairs(radius=0.03 * self._size, size=0.03 * self._size)
            tp_map = self._transform.map(tp)
            ch.translate(tp_map.x(), tp_map.y())
            self._annotations['target'] = self._scene.addPath(
                ch, target_pen
            )

        else:

            # We want to draw an arrow pointing the direction, butting
            # against the FOV circle
            target_pen.setStyle(Qt.SolidLine)
            delta = tp - self._scope_pos_plate
            angle = math.degrees(math.atan2(-delta.y(), delta.x()))
            if self._fov is not None:
                diaPixels = (self._fov * 60.0) / self._plate.compute_scale()
                logger.debug(f'diaPixels {diaPixels}')
            else:
                diaPixels = 0.8 * self._size
            ah = self._vertical_arrow_head(height=(0.03 * self._size), width_ratio=0.5)

            ah = ah.translated(
                0.0,
                -diaPixels/2.0,
            )
            ah = QTransform().translate(self._scope_pos_plate.x(), self._scope_pos_plate.y()).rotate(
                -(angle - 90) # Qt: CCW -ve, atan2 gave CCW angle from X-axis
            ).map(ah)
            ah = self._transform.map(ah)

            self._annotations['target'] = self._scene.addPath(
                ah, target_pen, QBrush(QColor(255, 0, 0, 128)),
            )

            self._updateDistanceText(angle=angle) # Save some computation by passing the angle

            # # Problematic Code:
            # self._annotations['target_distance'] = target_distance_item = self._scene.addText(
            #     f'{pretty_short(dist)} ({(60.0 * dist/self._fov):.1f}x)'
            # )
            # cy = QFontMetricsF(target_distance_item.font()).height()/2.0
            # target_distance_item.setTransform(
            #     QTransform().rotate(-angle).translate(diaPixels/2.0, 0).translate(0, -cy)
            #     * self._transform
            # )
            # target_distance_item.setDefaultTextColor(QColor(255, 0, 0))


    def _updateDistanceText(self, angle=None):
        if not self._show_annotations:
            self._clearAnnotations()
            return

        if 'target_distance' in self._annotations:
            self._scene.removeItem(self._annotations['target_distance'])
            del self._annotations['target_distance']

        if self._target_on_plate in (True, None):
            return

        # Determine the distance to the target
        tp = self.mapFromICRSToPlate(self._target)
        dist = angularDistance(self._target, self._scope_pos)
        if angle is None:
            delta = tp - self._scope_pos_plate
            angle = math.degrees(math.atan2(-delta.y(), delta.x()))

        if self._fov is not None:
            diaPixels = (self._fov * 60.0) / self._plate.compute_scale()
        else:
            diaPixels = 0.8 * self._size

        distance_path = QPainterPath()
        default_font = QFont()
        default_font_metrics = QFontMetricsF(default_font)
        distance_path.addText(
            QPointF(0, default_font_metrics.height()/2.0 - default_font_metrics.descent()),
            default_font,
            f'{pretty_short(dist)}' +
            (f' ({(60.0 * dist/self._fov):.1f}x)' if self._fov else '')
        )
        text_scale = 2.5 * ((0.015 * self._size) / default_font_metrics.height())
        distance_text_margin = 3
        distance_path = (
            QTransform().translate(
                self._scope_pos_plate.x(), self._scope_pos_plate.y()
            ).rotate(-angle).translate( # pylint: disable=invalid-unary-operand-type
                diaPixels/2.0 + distance_text_margin, 0
            ) * self._transform
        ).scale(text_scale, text_scale).map(distance_path)
        self._annotations['target_distance'] = self._scene.addPath(
            distance_path, QPen(Qt.NoPen), QBrush(QColor(255, 0, 0))
        )

    def _updateNorthArrow(self):
        if not self._show_annotations:
            self._clearAnnotations()
            return

        if 'north_arrow' in self._annotations:
            self._scene.removeItem(self._annotations['north_arrow'])
            del self._annotations['north_arrow']

        ncp = self.mapFromICRSToPlate(ICRS(ra=0, dec=90))
        w, h = self._pixmap.width(), self._pixmap.height()

        S = min(w, h)

        ncp_on_plate = False
        if abs(ncp.x()) < w/2.0 and abs(ncp.y()) < h/2.0:
            ncp_on_plate = True

        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(int(max(0.0001 * S, 1.1)))
        default_font = QFont()
        default_font_metrics = QFontMetricsF(default_font)
        if ncp_on_plate:
            ch = self._full_cross_hairs(gap=(0.005 * S), size=(0.005 * S))
            ch.addText(QPointF(0.01 * S, 0), default_font, 'NCP')
            ch.translate(ncp)
            self._annotations['north_arrow'] = self._scene.addPath(ch, pen)
        else:
            if self._fov is not None:
                diaPixels = (self._fov * 60.0) / self._plate.compute_scale()
            else:
                diaPixels = 0.8 * self._size

            northAngle = self._plate.compute_north_angle()
            ah = self._vertical_arrow_head(height=(0.01 * self._size), width_ratio=1.0)
            N_pos = default_font_metrics.width('N')/2.0
            ah.addText(QPointF(-N_pos, -3), default_font, 'N')
            cx, cy = self._scope_pos_plate.x(), self._scope_pos_plate.y()
            ah = (QTransform().translate(cx, cy).rotate(-northAngle).translate(0, -diaPixels/2.0)).map(ah)
            ah = self._transform.map(ah)
            # pen.setStyle(Qt.DashLine)
            self._annotations['north_arrow'] = self._scene.addPath(ah, pen)


    def _updateClickedPoint(self):
        if not self._show_annotations:
            self._clearAnnotations()
            return

        if not self._plate:
            logger.error('Trying to update clicked point annotation when plate is not available')
            return

        if 'clicked_point' in self._annotations:
            self._scene.removeItem(self._annotations['clicked_point'])
            del self._annotations['clicked_point']

        if 'clicked_point_coords' in self._annotations:
            self._scene.removeItem(self._annotations['clicked_point_coords'])
            del self._annotations['clicked_point_coords']

        if self._clicked_point is None:
            return

        S = min(self._pixmap.width(), self._pixmap.height())
        pen = QPen(self._red_pen)
        pen.setWidth(int(max(0.0025 * S, 2)))
        ch = self._full_cross_hairs(gap=(0.01 * S), size=(0.015 * S))
        ch.translate(self._clicked_point) # Already in Scene coordinates

        text = QPainterPath()
        default_font = QFont()
        default_font_metrics = QFontMetricsF(default_font)
        text.addText(
            QPointF(
                0.15 + default_font_metrics.width('+' if self._clicked_icrs.dec >= 0 else '-'),
                -default_font_metrics.descent()
            ),
            default_font,
            f'{pretty_ra(self._clicked_icrs.ra)}'
        )
        text.addText(
            QPointF(0.15, +default_font_metrics.ascent()),
            default_font,
            f'{pretty_dec(self._clicked_icrs.dec)}'
        )
        text_scale = 2.0 * (0.008 * S) / default_font_metrics.height()
        text = QTransform().translate(self._clicked_point.x(), self._clicked_point.y()).scale(text_scale, text_scale).map(text)

        self._annotations['clicked_point'] = self._scene.addPath(
            ch, self._red_pen, QBrush(QColor(255, 0, 0)),
        )
        self._annotations['clicked_point_coords'] = self._scene.addPath(
            text, QPen(Qt.NoPen), QBrush(QColor(255, 0, 0))
        )


    def _setClickedPoint(self, pos):
        if isinstance(pos, QPoint):
            pos = QPointF(pos)

        self._clicked_point = pos
        self._clicked_icrs = self.mapFromSceneToICRS(self._clicked_point)
        logger.debug(f'Clicked point resolved to ICRS({pretty_ra(self._clicked_icrs.ra)}, {pretty_dec(self._clicked_icrs.dec)})')

        self._updateClickedPoint()

        self.clickedPointChanged.emit(self._clicked_icrs)

    def _popupMenu(self):
        if self._clicked_point is None:
            logger.error('Cannot show popup menu -- no clicked point.')
            return

        self._last_context_menu_event = None

        view_pos = self.mapToGlobal(self.mapFromScene(self._clicked_point))
        self._last_context_menu_pos = x, y = int(view_pos.x()), int(view_pos.y())

        if self._menu:
            self._menu.exec(QPoint(x, y))
        else:
            logger.debug('No context menu to show!')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            try:
                self._setClickedPoint(self.mapToScene(event.pos()))
            except Exception as e:
                error_handler(e, 'Trying to set clicked point in EyepieceImageViewer')
            self._last_mouse_press_time = time.time()
            self._last_mouse_press_was_a_drag = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            self._last_mouse_press_time
        except AttributeError:
            self._last_mouse_press_time = None

        if self._last_mouse_press_time is not None:
            self._last_mouse_press_was_a_drag = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            self._last_mouse_press_was_a_drag
        except AttributeError:
            self._last_mouse_press_was_a_drag = False
        if not self._last_mouse_press_time:
            return

        if (time.time() - self._last_mouse_press_time) > 1.0 and (not self._last_mouse_press_was_a_drag):
            self.contextMenuEvent(event)
        self._last_mouse_press_time = None
        self._last_mouse_press_was_a_drag = False
        super().mouseReleaseEvent(event)

    def _addLabel(self, x, y, w, h):
        """Add a label that will respond to resizing of this widget.

        Specify (x, y) relative to the widget. Negative values bind it
        to the right or bottom corners.
        """
        newLabel = QLabel(parent=self)
        self._labelWidgets.append((newLabel, (x, y, w, h)))
        if x < 0:
            x = self.width() - w - int(x)
        if y < 0:
            y = self.height() - h - int(y)
        x = int(x)
        y = int(y)
        newLabel.move(x, y)
        newLabel.resize(w, h)
        return newLabel

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for label, (x, y, w, h) in self._labelWidgets:
            if x >= 0 and y >= 0:
                continue
            if x < 0:
                x = self.width() - label.width() - int(x)
                y = self.height() - label.height() - int(y)
            label.move(x, y)

    def updateGuideLabel(self, contents, w=450, h=35):
        try:
            self._guideLabel
        except AttributeError:
            self._guideLabel = self._addLabel(-10, 0, w, h)
            self._guideLabel.setPointSize(32) # default font size
            self._guideLabel.setVisible(self._show_annotations)

        self._guideLabel.setText(contents)
        self._guideLabel.adjustSize()
        self._guideLabel.move(self.width() - self._guideLabel.width(), 0)

    def setContextMenuItems(self, items, overwriteExisting=True):
        """
        items: An OrderedDict mapping the menu item name to a callback
        that accepts an ICRS argument
        """
        if not type(items) is OrderedDict:
            raise TypeError('EyepieceImageViewer.setContextMenuItems expects an OrderedDict. Got {}'.format(type(items)))

        new_items = OrderedDict([
            (name, lambda x, y: callback(self.mapFromSceneToICRS(self.mapToScene(QPoint(x, y)))))
            for name, callback in items.items()
        ] + [
            # Default actions from EyepieceView Dialog
            ('Fit In View', lambda x, y: self.fitInView()),
        ])
        super().setContextMenuItems(new_items, overwriteExisting=overwriteExisting)
