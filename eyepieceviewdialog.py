from PyQt5 import QtWidgets
from collections import OrderedDict

from ui_eyepieceview import Ui_eyepieceViewDialog as Ui_eyepieceViewDialog_Basic
from dms import pretty_icrs

class EyepieceViewDialog(QtWidgets.QDialog):

    def __init__(self, ui_eyepieceview):
        super().__init__()
        self._ui = ui_eyepieceview

class Ui_eyepieceViewDialog(Ui_eyepieceViewDialog_Basic):

    def __init__(self):
        super().__init__()

    def setupUi(self, eyepieceviewdialog):
        super().setupUi(eyepieceviewdialog)
