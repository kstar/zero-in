import math
import logging

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QGraphicsView, QMenu, QAction, QGraphicsScene, QPushButton
from collections import OrderedDict

logger = logging.getLogger('ImageViewer')

class ImageViewer(QGraphicsView):

    def __init__(self, parent):
        super(ImageViewer, self).__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap = None
        self.setScene(self._scene)
        self._no_pixmap = True
        self._zoom_level = 0
        self._zoom_factor = 1.25

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        self._menu = None
        self._last_context_menu_event = None
        self._buttons = OrderedDict()


    def _zoom_in(self):
        self._zoom_level += 1
        self.scale(self._zoom_factor, self._zoom_factor)

    def _zoom_out(self):
        self._zoom_level -= 1
        self.scale(1/self._zoom_factor, 1/self._zoom_factor)

    def wheelEvent(self, event):
        factor = self._zoom_factor
        if self.hasPixmap():
            delta = event.pixelDelta()
            if delta.isNull():
                delta = event.angleDelta()
            if delta.isNull():
                return
            angle = math.degrees(math.atan2(delta.y(), delta.x())) % 360.0
            if  angle >= 0 and angle <= 180:
                self._zoom_in()
            else:
                self._zoom_out()

    def hasPixmap(self):
        return (self._pixmap is not None)

    def reload(self):
        self._scene.clear()
        self._scene.addPixmap(self._pixmap)
        self._scene.update()

    def setPixmap(self, pixmap=None, keep_zoom=True):
        if pixmap and pixmap.isNull():
            pixmap = None

        firstPixmap = (self._no_pixmap and pixmap is not None)
        self._no_pixmap = (pixmap is None)
        self.setDragMode(QGraphicsView.ScrollHandDrag if pixmap else QGraphicsView.NoDrag)
        self._pixmap = QtGui.QPixmap(pixmap) if pixmap else QtGui.QPixmap()
        if not keep_zoom:
            self._zoom_level = 0

        self.reload()

        if firstPixmap or not keep_zoom:
            self.resetTransform()
            self.fitInView()

    def fitInView(self, *args, **kwargs):
        """
        Wrapper that provides the 0-argument overload fitInView() so it
        can be invoked as a slot
        """
        if self._pixmap is None:
            logger.error('Cannot fit in view: no pixmap')
            return

        if (len(args) == 0 or (len(args) == 1 and type(args[0]) is bool)) and len(kwargs) == 0:
            return self.fitInView(
                QtCore.QRectF(0, 0, self._pixmap.width(), self._pixmap.height()),
                QtCore.Qt.KeepAspectRatio
            )
        elif len(args) == 1 and isinstance(args[0], (QtCore.QRect, QtCore.QRectF)):
            return self.fitInView(
                args[0],
                QtCore.Qt.KeepAspectRatio
            )
        else:
            logger.warning(
                f'Calling fitInView with args {args} and keyword args {kwargs}'
            )
            return super().fitInView(*args, **kwargs)

    def _get_last_context_menu_location(self):
        assert self._last_context_menu_pos is not None
        return self._last_context_menu_pos

    def _create_context_menu_callback(self, callback):
        return lambda: callback(*self._get_last_context_menu_location())

    def setContextMenuItems(self, items, overwriteExisting=True):
        """
        items: An OrderedDict mapping the menu item name to a callback
        that accepts two args: (x, y) integer coordinates of context
        menu click relative to ImageViewer widget

        """
        if not type(items) is OrderedDict:
            raise TypeError('ImageViewer.setContextMenuItems expects an OrderedDict. Got {}'.format(type(items)))

        if self._menu is not None and overwriteExisting is False:
            return

        self._menu = QMenu(self)

        for name, callback in items.items():
            _action = QAction(name, parent=self)
            _action.triggered.connect(self._create_context_menu_callback(callback))
            self._menu.addAction(_action)

        logger.info('Added items to context menu!')


    def addButton(self, name, icon, callback):
        """
        name: Dictionary key to use for the button
        icon: Icon to use for the button
        callback: A function to call when the button is clicked
        """
        newButton = QPushButton(icon, "", parent=self)
        # newButton.setIconSize(icon.rect().size())
        newButton.clicked.connect(callback)
        if self._buttons is None:
            self._buttons = OrderedDict()
        self._buttons[name] = newButton
        newButton.move(5, 5 + 32 * (len(self._buttons) - 1))

    def addDefaultButtons(self):
        if 'zoom_in' not in self._buttons:
            self.addButton('zoom_in', QtGui.QIcon('zoom_in.png'), lambda: self._zoom_in())
        if 'zoom_out' not in self._buttons:
            self.addButton('zoom_out', QtGui.QIcon('zoom_out.png'), lambda: self._zoom_out())
        if 'fit' not in self._buttons:
            self.addButton('fit', QtGui.QIcon('fit.png'), lambda: self.fitInView())

    def clearButtons(self):
        for key in self._buttons:
            del self._buttons[key]
        self._buttons = OrderedDict()

    @property
    def getButton(self, name):
        return self._buttons.get(name, None)

    def unsetContextMenu(self):
        self._menu = None

    def contextMenuEvent(self, event):
        self._last_context_menu_event = event
        self._last_context_menu_pos = event.x(), event.y()
        if self._menu:
            self._menu.exec(event.globalPos())
        else:
            logger.debug('No context menu to show!')
