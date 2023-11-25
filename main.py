#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO)

import argparse
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from mainwindow import MainWindow, Ui_MainWindow
from eyepieceviewdialog import EyepieceViewDialog, Ui_eyepieceViewDialog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Telescope positioning UI')
    restore_group = parser.add_mutually_exclusive_group(required=False)
    restore_group.add_argument(
        '--restore-from',
        default='uistate.json',
        type=str,
        help='Restore UI settings from the supplied file',
    )
    restore_group.add_argument(
        '--no-restore',
        action='store_true',
        help='Do not restore settings. Instead, default to the "factory" values.',
    )
    parser.add_argument(
        '--connect-indi', '-i',
        action='store_true',
        help='Automatically connect to INDI Server with the automatically loaded settings.',
    )
    parser.add_argument(
        '--connect-imu', '-m',
        action='store_true',
        help='Automatically connect to IMU with the automatically loaded settings.',
    )
    parser.add_argument(
        '--sync', '-s',
        action='store_true',
        help=(
            'Automatically do a platform sync when the program starts. The'
            ' automatically loaded observing site and platform travel time settings'
            ' will be used.'
        )
    )
    args = parser.parse_args(sys.argv[1:])
    restore_file = args.restore_from if not args.no_restore else None

    app = QtWidgets.QApplication(sys.argv)

    ui_eyepieceview = Ui_eyepieceViewDialog()
    eyepieceViewDialog = EyepieceViewDialog(ui_eyepieceview)
    ui_eyepieceview.setupUi(eyepieceViewDialog)

    ui = Ui_MainWindow(eyepieceViewDialog)
    mainWindow = MainWindow(ui, eyepieceViewDialog)
    ui.setupUi(mainWindow, restore_file=restore_file)

    mainWindow.closed.connect(eyepieceViewDialog.accept)

    mainWindow.initialize(
        connect_indi=args.connect_indi, sync=args.sync, connect_imu=args.connect_imu
    )
    app.aboutToQuit.connect(ui.shutDown)
    mainWindow.close
    mainWindow.show()
    eyepieceViewDialog.show()

    sys.exit(app.exec_())
