import logging
import sys
import threading
import concurrent.futures
import datetime
import math
import time
from collections import OrderedDict


from PyQt5 import QtWidgets, QtGui, QtCore, QtDBus
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from dms import *
from error_handler import error_handler, silent_error_handler
import align
from backend import MainBackend, make_qimage_from_image_data, SyncResult
from ui import Ui_MainWindow as Ui_MainWindow_BasicSetup
from coordinates import ICRS
from uistate import save_state, restore_state
import timing

logger = logging.getLogger('UI')

statusBar = None

class BusyStatus:
    global statusBar
    def __init__(self, statusDuring=None, statusAfter=None, waitCursor=True):
        self._statusDuring = statusDuring
        self._statusAfter = statusAfter
        self._waitCursor = waitCursor

    def __enter__(self):
       if self._waitCursor:
           QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
       if self._statusDuring and statusBar:
           statusBar.showMessage(self._statusDuring)

    def __exit__(self, exc_type, exc_value, tb):
       if self._statusAfter and statusBar:
           statusBar.showMessage(self._statusAfter)
       if self._waitCursor:
           QtWidgets.QApplication.restoreOverrideCursor()
       if exc_type:
           raise


class MainWindow(QtWidgets.QMainWindow):

    exposureDone = pyqtSignal(concurrent.futures.Future, name='exposureDone')
    plateSolveDone = pyqtSignal(concurrent.futures.Future, name='plateSolveDone')
    successfulSolve = pyqtSignal(float, float, datetime.datetime, name='successfulSolve') # RA, Dec, ExposureLT
    synced = pyqtSignal(SyncResult, name='synced')
    closed = pyqtSignal(name='closed')
    scopePositionChanged = pyqtSignal(ICRS, str, name='scopePositionChanged')

    def __init__(self, ui, eyepieceViewDialog):
        super().__init__()

        self._ui = ui
        self._ui_eyepieceview = eyepieceViewDialog._ui
        self._eyepieceview = eyepieceViewDialog
        self._backend = MainBackend()
        self._auto_solve_timer = QTimer(self)
        self._altaz_sync_timer = QTimer(self)
        self._dbus = None
        self._capture_scene = QtWidgets.QGraphicsScene()
        self._sync_result = None
        self.alignment = None
        self._solved_pos = None
        self._plate_data = None
        self._solved_pos_lt = None
        self._imu_scope_pos = None
        self._target_pos = None
        self._start_exposure_executor = None
        self._indi_connected = False

        self.exposureDone.connect(self.restoreExposureUiElements)
        self.exposureDone.connect(self.updateCaptureImage)
        self.successfulSolve.connect(self.updateEyepieceView)

        self._auto_solve_timer.timeout.connect(self.solve_silent)
        self._altaz_sync_timer.timeout.connect(self.timeSync)
        self._backend.temperaturePollingError.connect(self.temperaturePollingErrorHandler)
        self._backend.temperatureUpdate.connect(self.slotTemperatureUpdate)
        self.scopePositionChanged.connect(self.guideUpdate)


    def _initGuideLabel(self):
        class GuideDialog(QtWidgets.QDialog):
            def __init__(self, parent):
                logger.info('Initializing guide dialog!')
                super().__init__(parent)
                self.setWindowTitle('Guide')
                self.setWindowFlags(
                    QtCore.Qt.WindowStaysOnTopHint
                    | QtCore.Qt.Tool
                    | QtCore.Qt.FramelessWindowHint
                )
                self.setModal(False)
                self._layout = QtWidgets.QGridLayout(self)
                self.targetGuideLabel = QtWidgets.QLabel(self)
                font = QtGui.QFont()
                font.setPointSize(48)
                self.targetGuideLabel.setFont(font)
                self.targetGuideLabel.setObjectName("targetGuideLabel")
                self.targetGuideLabel.setText("Guide")
                self._layout.addWidget(self.targetGuideLabel)

        self._guide_dialog = GuideDialog(self)
        self._guide_dialog.show()

    def initialize(self, connect_indi=False, sync=False, connect_imu=False):
        global statusBar
        statusBar = self._ui.statusbar
        # Any UI -> Backend initializations
        self.updateSolverBackendSettings()
        self.alignment = self._backend.loadPreviousAlignmentIfExists()
        self._ui.targetNameLineEdit.textChanged.connect(
            lambda text: self._ui.resolveTargetButton.setEnabled(len(text) > 0)
        )
        if self.alignment is not None:
            logger.info('Loaded previous alignment!')
            self._update_ui_with_alignment(
                self.alignment,
                "<font color='red'><b>WARNING:</b></font> Alignment was loaded from file and may be outdated." + (
                    ('Aligned at: ' + datetime.datetime.fromtimestamp(
                        self.alignment['timestamp']
                    ).strftime('%Y-%m-%d %H:%M:%S [UTC]'))
                )
            )
        if connect_indi:
            logger.info('Connecting to INDI server automatically')
            self.indiConnect()

        if sync:
            logger.info('Performing time sync automatically')
            self.timeSync()

        if connect_imu:
            logger.info('Performing IMU connection automatically')
            self.imuConnect()

        self._initGuideLabel()

        self.imuDeviceChanged(self._ui.imuDeviceComboBox.currentText())

        if not self._sync_result:
            self.timeSync(fromPickle=True)
            if self._sync_result:
                logger.warning('Loaded previous sync! Might be outdated')



    def updateSyncTimeInfo(self):
        raise NotImplementedError

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


    #### Slots ####
    def indiConnect(self):

        try:
            with BusyStatus(statusDuring='Connecting to INDI server...', statusAfter=''):
                self._backend.indiConnect(
                    ccd_name=self._ui.indiDeviceLineEdit.text(),
                    indi_host=self._ui.indiHostLineEdit.text(),
                    indi_port=self._ui.indiPortSpinBox.value()
                )
            self._ui.alignButton.setEnabled(self._sync_result is not None)
            self._ui.indiConnectButton.setEnabled(False)
            self._ui.indiDeviceLineEdit.setEnabled(False)
            self._ui.indiHostLineEdit.setEnabled(False)
            self._ui.indiPortSpinBox.setEnabled(False)
            self._indi_connected = True
        except Exception as e:
            self._indi_connected = False
            error_handler(e, 'connect to INDI / CCD')

    def imuDeviceChanged(self, device):
        if device == 'USFS':
            usfs = True
        elif device == 'Arduino':
            usfs = False
        else:
            raise NotImplementedError(f'Unhandled IMU device type: {device}')

        self._ui.imuSerialPortLineEdit.setVisible(not usfs)
        self._ui.imuBaudRateComboBox.setVisible(not usfs)
        self._ui.imuSerialPortLabel.setVisible(not usfs)
        self._ui.imuBaudRateLabel.setVisible(not usfs)
        self._ui.imuI2cPortLabel.setVisible(usfs)
        self._ui.imuI2cPortComboBox.setVisible(usfs)
        self._ui.imuAccelBWComboBox.setVisible(usfs)
        self._ui.imuAccelRangeComboBox.setVisible(usfs)
        self._ui.imuAccelMergeRateSpinBox.setVisible(usfs)
        self._ui.imuGyroBWComboBox.setVisible(usfs)
        self._ui.imuGyroRangeComboBox.setVisible(usfs)
        self._ui.imuMagMergeRateSpinBox.setVisible(usfs)
        self._ui.imuAccelBWLabel.setVisible(usfs)
        self._ui.imuAccelRangeLabel.setVisible(usfs)
        self._ui.imuAccelMergeRateLabel.setVisible(usfs)
        self._ui.imuGyroBWLabel.setVisible(usfs)
        self._ui.imuGyroRangeLabel.setVisible(usfs)
        self._ui.imuMagMergeRateLabel.setVisible(usfs)

    def imuConnect(self):
        device = self._ui.imuDeviceComboBox.currentText()

        try:
            config = {}
            if device == 'USFS':
                config['port'] = self._ui.imuI2cPortComboBox.currentText()
                for key, value in self._ui.usfs_property_map.items():
                    config[key] = value['map'][value['ui'].currentText()]
                # Non combo-boxed items like merge rate go here (not implemented yet)
            elif device == 'Arduino':
                config['port'] = self._ui.imuSerialPortLineEdit.text()
                config['baudrate'] = int(self._ui.imuBaudRateComboBox.currentText())
            else:
                raise NotImplementedError(f'Unhandled IMU device type: {device}')

            with BusyStatus(statusDuring='Connecting to IMU...', statusAfter=''):
                self._backend.imuConnect(device, **config)
            self._backend.imu.motionUpdate.connect(self.imuUpdate)

            self._ui.imuDeviceComboBox.setEnabled(False)
            self._ui.imuSerialPortLineEdit.setEnabled(False)
            self._ui.imuBaudRateComboBox.setEnabled(False)
            self._ui.imuLowpassCheckBox.setEnabled(True)
            self._ui.imuLowpassMemorySpinBox.setEnabled(True)
            self._ui.imuConnectButton.setEnabled(False)
            # self._ui.imuAccelBWComboBox.setEnabled(False)
            # self._ui.imuAccelRangeComboBox.setEnabled(False)
            # self._ui.imuAccelMergeRateSpinBox.setEnabled(False)
            # self._ui.imuGyroBWComboBox.setEnabled(False)
            # self._ui.imuGyroRangeComboBox.setEnabled(False)
            # self._ui.imuMagMergeRateSpinBox.setEnabled(False)

            self.updateImuLowpassConfig()

        except Exception as e:
            error_handler(e, 'connect to IMU')

    def updateImuLowpassConfig(self):
        if (not self._backend.imu) or (not self._backend.imu.connected):
            logger.warning(f'Not connected to IMU: ignoring lowpass config')
            return
        self._backend.imu.lowpass( # Enable or disable low-pass
            self._ui.imuLowpassCheckBox.isChecked(),
            self._ui.imuLowpassMemorySpinBox.value()
        )

    def setTrackingToAltAz(self, isAltAz):
        refresh_interval = 60.0
        if isAltAz:
            self._altaz_sync_timer.start(int(refresh_interval * 1000.0))
        else:
            self._altaz_sync_timer.stop()

    def updateUsfsConfig(self):
        if (not self._backend.imu) or (not self._backend.imu.connected):
            return # Called before initialization
        device = self._ui.imuDeviceComboBox.currentText()
        if device != 'USFS':
            error_handler(AssertionError('Trying to update USFS Config with IMU device {device}. Ignoring request.'))
        config = {}
        for key, value in self._ui.usfs_property_map.items():
            config[key] = value['map'][value['ui'].currentText()]
        # Non combo-boxed items like merge rate go here (not implemented yet)
        try:
            with BusyStatus(statusDuring='Reconfiguring IMU...', statusAfter=''):
                assert self._backend.imu.reconfigure(**config)
            logger.info(f'Successful reconfigure of IMU')
        except Exception as e:
            logger.error(f'Encountered exception while reconfiguring IMU: {e}. Trying to go back to old config.')
            try:
                assert self._backend.imu.initialize()
                config = self._backend.imu.config
                for key, value in self._ui.usfs_property_map.items():
                    rev_map = {v: k for k, v in value['map'].items()}
                    value['ui'].blockSignals(True)
                    value['ui'].setCurrentText(rev_map[config[key]]) # FIXME: Not great programming...
                    value['ui'].blockSignals(False)
                error_handler(RuntimeWarning('Failed to reconfigure USFS IMU. Exception: {e}. Went back to old settings!'))
            except Exception as e2:
                error_handler(Exception('Not only did we fail to reconfigure the USFS, we even failed to restore the old settings. See the terminal for diagnostics. Exception was {e2}. Your best bet is to restart the application and hard reset.'))

    def imuUpdate(self, tq):
        # TODO: Complete this method's implementation
        try:
            te = self._backend.coordinate_conversion.quaternionToEuler(tq)
            self._ui.imuHeadingLabel.setText(pretty_dec(te.e.yaw))
            self._ui.imuRollLabel.setText(pretty_dec(te.e.roll))
            self._ui.imuPitchLabel.setText(pretty_dec(te.e.pitch))
        except Exception as e:
            silent_error_handler(e, 'Updating IMU position (heading)')
            self._ui.imuHeadingLabel.setText('???')
            self._ui.imuRollLabel.setText('???')
            self._ui.imuPitchLabel.setText('???')

        try:
            # FIXME: Move to the backend!
            if self._backend.lastImuCalibration is not None:
                scope_estimated_icrs = self._backend.getScopeICRS(tq) # ICRS
                self._imu_scope_pos = scope_estimated_icrs
                scope_estimated_altaz = self._backend.equatorialToHorizontal(
                    scope_estimated_icrs.ra, scope_estimated_icrs.dec
                ) # Alt-Az referenced to platform frame
                self._ui.scopeRALabel.setText(pretty_ra(scope_estimated_icrs.ra))
                self._ui.scopeDecLabel.setText(pretty_dec(scope_estimated_icrs.dec))
                self._ui.scopeAltLabel.setText(pretty_dec(scope_estimated_altaz.alt))
                self._ui.scopeAzLabel.setText(pretty_dec(scope_estimated_altaz.az))
                self.scopePositionChanged.emit(scope_estimated_icrs, 'imu')
        except Exception as e:
            silent_error_handler(e, 'Updating IMU position (scope ICRS)')
            self._ui.scopeRALabel.setText('???')
            self._ui.scopeDecLabel.setText('???')


    def updateSolverBackendSettings(self):
        if self._ui.sepCheckBox.isChecked():
            sep_threshold = self._ui.sepThreshold.value()
        else:
            sep_threshold = None

        self._backend.setSolverSettings(
            sep_threshold=sep_threshold,
            binning=self._ui.binningSpinBox.value(),
            top_k=self._ui.starCountCutoffSpinBox.value(),
            auto_debayer=self._ui.debayerCheckBox.isChecked(),
        )

    def disableExposureUiElements(self):
        exposure_buttons = [
            self._ui.alignButton,
            self._ui.solveButton,
        ]

        self._exposure_ui_elements_state = [
            (button, button.isEnabled())
            for button in exposure_buttons
        ]

        for button in exposure_buttons:
            button.setEnabled(False)

    def restoreExposureUiElements(self):
        for button, state in self._exposure_ui_elements_state:
            button.setEnabled(state)

    def startExposure(self, timeout=15):
        """UI-friendly wrapper around backend's startExposure"""
        print('Entered startExposure', file=sys.stderr, flush=True)
        exposure = self._ui.exposureSpinBox.value()
        self.disableExposureUiElements()
        if self._start_exposure_executor == None:
            self._start_exposure_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = self._start_exposure_executor.submit(
            self._backend.expose, exposure, timeout=timeout
        )
        print('Submitted exposure task', file=sys.stderr, flush=True)
        future.add_done_callback(self.exposureDone.emit)

        print('Exiting startExposure', file=sys.stderr, flush=True)
        return future

    def waitForSignal(self, signal,
                      status_pre='Waiting for task to complete...',
                      status_post='Task complete.',
                      change_cursor=True, update_statusbar=True):
        """Runs the Qt Event Loop so we can process non-exposure events while
        we wait for the exposure to complete
        """
        status_pre = status_pre if update_statusbar else None
        status_post = status_post if update_statusbar else None
        with BusyStatus(statusDuring=status_pre, statusAfter=status_post, waitCursor=change_cursor):
            loop = QtCore.QEventLoop()
            signal.connect(loop.quit)
            loop.exec()

    def waitForExposure(self):
        return self.waitForSignal(
            self.exposureDone,
            status_pre='Waiting for exposure to complete...',
            status_post='Exposure complete.',
        )

    def waitForPlateSolve(self):
        return self.waitForSignal(
            self.plateSolveDone,
            status_pre='Waiting for plate-solve to complete...',
            status_post='Plate-solve complete.',
        )

    def updateCaptureImage(self, capture_future):
        assert capture_future.done()

        exception = capture_future.exception() # Should not block because of above

        if exception is not None:
            error_handler(exception, 'capture an image')
            return

        image_data = capture_future.result() # Should not block, again
        if image_data is None:
            error_handler(RuntimeError('Capture returned null result. Did it timeout?'), 'update the captured image preview')
            return

        try:
            pixmap = QtGui.QPixmap.fromImage(make_qimage_from_image_data(image_data))
            self._capture_pixmap = pixmap
            self._ui.currentCaptureView.setScene(self._capture_scene)
            self._update_capture_scene_pixmap()
        except Exception as e:
            error_handler(e, 'update the captured image preview')

    def indiExposureChanged(self, exposure):
        pass

    def _update_capture_scene_pixmap(self):
        assert self._capture_pixmap is not None
        self._capture_scene.clear()
        self._capture_scene.addPixmap(self._capture_pixmap.scaled(
            self._ui.currentCaptureView.width(),
            self._ui.currentCaptureView.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))
        self._capture_scene.update()


    def annotateCapture(self, x, y):
        """
        Place cross-hairs on capture scene at (x, y)
        """
        assert self._capture_pixmap is not None
        painter = QtGui.QPainter(self._capture_pixmap)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(3)
        painter.setPen(pen)
        L = int(
            0.005 * max(self._capture_pixmap.width(), self._capture_pixmap.height())
        )
        # painter.drawLine(x - 2 * L, y, x - L, y)
        # painter.drawLine(x + L, y, x + 2 * L, y)
        # painter.drawLine(x, y + L, x, y + 2 * L)
        # painter.drawLine(x, y - L, x, y - 2 * L)
        self._update_capture_scene_pixmap()


    def alignChooseTarget(self, index):
        align_target = self._ui.alignComboBox.currentText()
        assert align_target in align.ALIGN_TARGETS or align_target in ("Custom", "ask KStars"), align_target
        if align_target in align.ALIGN_TARGETS:
            ra, dec = align.ALIGN_TARGETS[align_target]
            self._ui.alignRALineEdit.setText(pretty_ra(ra))
            self._ui.alignDecLineEdit.setText(pretty_dec(dec))
            self._ui.alignRALineEdit.setEnabled(False)
            self._ui.alignDecLineEdit.setEnabled(False)
        elif align_target == "Custom":
            self._ui.alignRALineEdit.setEnabled(True)
            self._ui.alignDecLineEdit.setEnabled(True)
        elif align_target == "ask KStars":
            target_name, ok = QtWidgets.QInputDialog().getText(
                self,
                "Resolve alignment target using KStars",
                "Alignment target name:"
            )
            self._ui.alignComboBox.setCurrentIndex(
                self._ui.alignComboBox.findText("Custom")
            )
            self._ui.alignRALineEdit.setEnabled(True)
            self._ui.alignDecLineEdit.setEnabled(True)
            if ok and target_name:
                try:
                    target = self._resolveUsingKStars(target_name)
                    self._ui.alignRALineEdit.setText(pretty_ra(target.ra))
                    self._ui.alignDecLineEdit.setText(pretty_dec(target.dec))
                except Exception as e:
                    error_handler(e, context='Resolving alignment target using KStars')
                    self._ui.alignRALineEdit.setText('')
                    self._ui.alignDecLineEdit.setText('')

    def _update_ui_with_alignment(self, alignment, info=''):
        self._ui.currentAlignmentInfoLabel.setText(info)

        if alignment is None:
            return

        self._ui.alignArcsecPerPixelLabel.setText(
            '{:.3f}'.format(alignment['arcsecperpix'])
        )
        self._ui.solveButton.setEnabled(True)
        self._ui.solveAutoCheckBox.setEnabled(True)
        self._ui.solveAutoCheckBox.setChecked(False)

        x, y = alignment['x'], alignment['y']

        self._ui.alignOffsetXLabel.setText('{:.2f}'.format(x))
        self._ui.alignOffsetYLabel.setText('{:.2f}'.format(y))

    def align(self):
        if self.alignment is not None:
            confirm_box = QtWidgets.QMessageBox.question(
                self,
                'Confirmation of re-alignment',
                'It looks like you already have an alignment! '
                'Redoing it will lose any previous alignment. '
                'Are you sure you want to redo alignment?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if confirm_box != QtWidgets.QMessageBox.Yes:
                logger.info('Alignment canceled by user!')
                return
            logger.info('Re-doing alignment per user wishes')

        try:
            align_ra = convert_ra(self._ui.alignRALineEdit.text())
            align_dec = convert_dec(self._ui.alignDecLineEdit.text())

            # Exposure
            self._wait(self.startExposure(), self.waitForExposure)

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            arcsecperpix_hint = self._ui.alignArcsecPerPixelHintSpinBox.value()
            if arcsecperpix_hint == 0.0:
                arcsecperpix_hint = None
            future = executor.submit(
                self._backend.alignCapturedImage, align_ra, align_dec, arcsecperpix_hint
            )
            future.add_done_callback(self.plateSolveDone.emit)

            self.alignment = self._wait(future, self.waitForPlateSolve)
            assert self.alignment is not None, 'This should have been an exception...'

            self._update_ui_with_alignment(
                self.alignment,
                'Aligned at: {}'.format(datetime.datetime.fromtimestamp(
                    self.alignment['timestamp']
                ).strftime('%Y-%m-%d %H:%M:%S [UTC]'))
            )
            self.annotateCapture(self.alignment['x'], self.alignment['y'])

        except Exception as e:
            error_handler(e, 'alignment')
            self._ui.statusbar.showMessage(f'Error: {e}')
            self._update_ui_with_alignment(None, f'Alignment error: {e}')

    def timeSync(self, fromPickle=False):
        if not fromPickle:
            try:
                lat = convert_dms(self._ui.siteLatitudeLineEdit.text())
                lon = convert_dms(self._ui.siteLongitudeLineEdit.text())
                height = self._ui.siteElevationSpinBox.value()
            except Exception as e:
                error_handler(e, 'interpret observing site information')
                return

        try:
            if not fromPickle:
                self._backend.setSiteParameters(
                    lat=lat,
                    lon=lon,
                    height=height,
                )

                altaz = self._ui.altAzTrackingCheckBox.isChecked()

                self._sync_result = self._backend.syncTime(
                    1.0/60.0 if altaz else self._ui.platformTimeSpinBox.value(),
                    self._ui.siteTemperatureSpinBox.value(),
                )
            else:
                self._sync_result = self._backend.loadSyncFromPickle() # This fails quietly, logging only to the console

            if self._sync_result:
                self._ui.syncLTLabel.setText('{} UT{:+.2f}'.format(self._sync_result.sync_info['sync_lt'].strftime('%H:%M:%S'), self._sync_result.sync_info['tz_offset']))
                self._ui.scopeUTLabel.setText(self._sync_result.sync_info['scope_ut'].strftime('%H:%M:%S'))
                self._ui.scopeLSTLabel.setText(pretty_ra(self._sync_result.frame.LST))
                self._ui.alignButton.setEnabled(self._indi_connected)
                self.synced.emit(self._sync_result)

        except Exception as e:
            error_handler(e, 'time sync')

    def tempSourceChanged(self, source):
        if source == 'Manual':
            self._backend.pollTemperature(False)
            return
        self._backend.pollTemperature(True, device=source)

    def temperaturePollingErrorHandler(self, error):
        error_handler(RuntimeError(f'Failed to poll temperature: {error}. Changing temperature source to Manual.'))
        self._ui.tempSourceComboBox.setCurrentText('Manual')

    def slotTemperatureUpdate(self, tt):
        self._ui.siteTemperatureSpinBox.setValue(float(tt.T))
        self._ui.tempUpdateTimeLabel.setText(tt.ut.strftime('%H:%M:%S [UTC]'))

    def solveAutoToggled(self, state: bool) -> None:
        refresh_interval = 1
        if state is True:
            self._auto_solve_timer.start(int(refresh_interval * 1000.0))
        else:
            self._auto_solve_timer.stop()

    def _wait(self, future, waitMethod):
        if future is None:
            raise RuntimeError('Future is None!')
        if future.done():
            logger.warning(
                'In MainWindow._wait, future was already done when _wait was called.'
            )
            return future.result()
        waitMethod()
        while not future.done():
            logger.error('The future object called its call-back, but future.done() returns False... trying again in 0.25s')
            time.sleep(0.25)
        exception = future.exception()
        if exception is not None:
            raise(exception)
        return future.result()

    def _resolveUsingKStars(self, name) -> ICRS:
        import xml.etree.ElementTree as ElemTree
        kstars_dbus = self._create_or_get_dbus_connection()
        with BusyStatus(statusDuring='Resolving with KStars...', statusAfter=''):
            result = kstars_dbus.call(QtDBus.QDBus.AutoDetect, "getObjectPositionInfo", name)
        if result.errorMessage() != '':
            raise RuntimeError(
                f'DBus call for getObjectPositionInfo resulted in the following error: {result.errorMessage()}'
            )
        xmldata = result.arguments()[0]
        data = ElemTree.fromstring(xmldata)
        return ICRS(
            ra=float(data.find('RA_J2000_Degrees').text),
            dec=float(data.find('Dec_J2000_Degrees').text),
        )

    def resolveTarget(self) -> None:
        try:
            target = self._resolveUsingKStars(self._ui.targetNameLineEdit.text())
            self._ui.targetRALineEdit.setText(pretty_ra(target.ra))
            self._ui.targetDecLineEdit.setText(pretty_dec(target.dec))
        except Exception as e:
            error_handler(e, context='Resolving target using KStars')


    def setTarget(self) -> None:
        try:
            target_ra = convert_ra(self._ui.targetRALineEdit.text())
            target_dec = convert_dec(self._ui.targetDecLineEdit.text())
        except Exception as e:
            error_handler(e, context='Interpretation of Target RA/Dec')
            return

        if self._sync_result and 'target_altaz' in self._sync_result.cache:
            del self._sync_result.cache['target_altaz']

        try:
            self._target_pos = ICRS(target_ra, target_dec)
            self._ui_eyepieceview.eyepieceView.setTarget(self._target_pos)
        except Exception as e:
            error_handler(e, context='Setting the target in the eyepiece view')

        # TODO: Implement guide labels


    def solve(self) -> None:
        return self._solve(sep_plot_detections=True)

    def solve_silent(self) -> None:
        return self._solve(error_reporting='statusbar', sep_plot_detections=False)

    def _sync_alignment_to_point(self, icrs):
        logger.info('Local alignment sync {pretty_icrs(icrs)}')
        self._backend.localAlignmentSync(icrs.ra, icrs.dec)

    def applyNamedFOV(self, name):
        error_handler(NotImplementedError('This feature is not implemented yet'))

    def updateEyepieceView(self) -> None:
        """ Render the eyepiece view """
        Timer = timing.makeOrGetTimingClass('SolveLoop')
        try:
            with Timer('2.0:getDSSPlate'):
                fov = self._ui.fovSpinBox.value()
                focuser_angle = self._ui.focuserAngleSpinBox.value()
                plate_fov = self._ui.imageSizeSpinBox.value() * fov
                plate, scope_pos = self._backend.getDSSPlate(plate_fov)
            with Timer('2.1:eyepieceRotation'):
                display_angle = self._ui.eyepieceViewportAngleSpinBox.value()
                display_on_scope = self._ui.eyepieceViewportOnScopeCheckBox.isChecked()
                rotation = self._backend.getEyepieceRotation(
                    focuser_angle,
                    display_angle,
                    plate.compute_north_angle(),
                    display_on_scope,
                )
                logger.info(f'Got eyepiece rotation = {rotation}° (CW)')
            with Timer('2.2:updateEyepieceView'):
                epv = self._ui_eyepieceview.eyepieceView
                epv.updateEyepieceView(plate, scope_pos, rotation)
                epv.setFovCircle(fov)
                epv.setTarget(self._target_pos)
                epv.setContextMenuItems(
                    OrderedDict([
                        ('Sync to center', self._sync_alignment_to_point)
                    ])
                )

            # DEBUG code for optimization
            from matplotlib import pyplot as plt
            try:
                self._timings
            except AttributeError:
                self._timings = {}
            try:
                self._timing_plot
            except AttributeError:
                self._timing_plot = plt.figure()
                self._timing_plot.show()

            logger.info('============= TIMING =============')
            for key, value in Timer.get().items():
                self._timings.setdefault(key, []).append(float(value))

                if len(self._timings[key]) > 15:
                    self._timings[key] = self._timings[key][1:]
                print(f'\t{key}: {value:0.2f}\n', file=sys.stderr)

            self._timing_plot.clf()
            ax = self._timing_plot.add_subplot(111)
            ax.stackplot(list(map(float, range(len(self._timings[key])))), self._timings.values(), labels=self._timings.keys())
            ax.legend(loc='upper left')
            ax.set_title('Timing chart')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('seconds')
            self._timing_plot.canvas.draw()

        except Exception as e:
            silent_error_handler(e, context='rendering eyepiece view')

    def _create_or_get_dbus_connection(self):
        try:
            self._dbus
        except AttributeError:
            self._dbus = None

        try:
            self._kstars_dbus
        except AttributeError:
            self._kstars_dbus = None

        if self._dbus is None:
            self._dbus = QtDBus.QDBusConnection.sessionBus()
            if self._dbus is None:
                raise RuntimeError('Could not connect to DBus Session bus!')

        if self._kstars_dbus is None:
            self._kstars_dbus = QtDBus.QDBusInterface(
                "org.kde.kstars",
                "/KStars",
                "org.kde.kstars",
                self._dbus
            )
            if self._kstars_dbus is None or (not self._kstars_dbus.isValid()):
                self._kstars_dbus = None
                raise RuntimeError('Could not connect to KStars DBus interface!')

        return self._kstars_dbus


    def _solve(self,
               error_reporting='messagebox',
               sep_plot_detections=False,
               imu_update=True,
               smudge_threshold=15, # Smudge threshold in arcmin
    ) -> None:
        assert error_reporting in ('messagebox', 'statusbar')
        try:
            Timer = timing.makeOrGetTimingClass('SolveLoop')
            Timer.reset()

            # Exposure
            with Timer('0.0:exposure'):
                logger.info('=== Exposure ===')
                tq_imu_preexposure = None
                if self._backend.imu and self._backend.imu.connected:
                    self._backend.imu.poll() # Force a poll
                    tq_imu_preexposure = self._backend.imu.most_recent_tq
                future = self.startExposure()
                self._wait(future, self.waitForExposure)
                exposure_lt = self._backend.last_exposure_lt
                logger.info('Exposure Done. Exposure timestamp: {}'.format(exposure_lt))

            with Timer('0.1:imu_read'):
                logger.info('=== IMU ===')
                tq_imu = None
                if self._backend.imu and self._backend.imu.connected:
                    self._backend.imu.poll() # Force a poll
                    tq_imu = self._backend.imu.most_recent_tq
                    exposure_ut = self._backend.last_exposure_ut
                    time_delta = abs((tq_imu.ut - exposure_ut).total_seconds())
                    if time_delta > 10.0:
                        # FIXME: Make this warning more obvious than an obscure console log message!
                        logger.warning('{:.1f}s time difference between IMU read and exposure. Calibration may not be okay!')
                    if tq_imu is None:
                        logger.error('Could not read IMU! Will ignore IMU re-calibration.')
                else:
                    logger.warning('IMU not connected!')

            # FIXME: The IMU we are using right now is insensitive to
            # the tracking platform motion. This is not good. So we
            # actually use the raw IMU rotation values to determine
            # smudging. In reality, we should be using ICRS, and the
            # quaternion rotation should match the 1° in 4 minutes
            # sidereal rate. But because our sensor is crap, we read
            # the quaternion reading directly.
            with Timer('0.2:smudge_check'):
                if tq_imu and tq_imu_preexposure:
                    smudge_arcmin = (tq_imu_preexposure.q.inverse * tq_imu.q).degrees * 60.0
                    if smudge_arcmin > smudge_threshold:
                        raise RuntimeError(
                            f'Smudge of {smudge_arcmin:.1f}\' detected (threshold={smudge_threshold:.1f}\'). Bailing out without solving!'
                        )

            logger.info('=== Solve ===')
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                self._backend.solveCapturedImage, sep_plot_detections=sep_plot_detections, tq_imu=tq_imu
            )
            future.add_done_callback(self.plateSolveDone.emit)

            logger.info('Submitted plate solve. Waiting for it to finish.')
            ra, dec = self._wait(future, self.waitForPlateSolve)
            logger.info('Plate solve complete.')
            Timer.timing.update({
                f'0.3.{key}': value for key, value in timing.makeOrGetTimingClass('PlateSolve').get().items()
            })

            with Timer('0.4:ui_update'):
                self.annotateCapture(self.alignment['x'], self.alignment['y'])

                self._ui.solveRALabel.setText(pretty_ra(ra))
                self._ui.solveDecLabel.setText(pretty_dec(dec))
                self._solved_pos = ICRS(ra=ra, dec=dec)
                self.scopePositionChanged.emit(self._solved_pos, 'solve')

                self._ui.solveStatusLabel.setText("OK")
                self._ui.solveLTLabel.setText(exposure_lt.strftime('%H:%M:%S'))

                self.successfulSolve.emit(ra, dec, exposure_lt)
                self._solved_pos_lt = exposure_lt

                # Compute Alt/Az
                if self._sync_result is not None:
                    try:
                        altaz = self._backend.equatorialToHorizontal(ra, dec)
                        self._ui.solveAltLabel.setText(pretty_dec(altaz.alt))
                        self._ui.solveAzLabel.setText(pretty_dec(altaz.az))
                    except Exception as e:
                        error_handler(e, 'convert plate-solved scope coordinates to altazimuth')

                # FIXME: Move to a different class
                if self._ui.solveToKStarsCheckBox.isChecked():
                    try:
                        kstars_dbus = self._create_or_get_dbus_connection()
                        logger.warning('FIXME: Precession correction for KStars')
                        logger.info(
                            'Calling org.kde.kstars.setRaDecJ2000 with arguments {}, {}'.format(
                                ra/15.0, dec
                            )
                        )
                        kstars_dbus.call(
                            QtDBus.QDBus.AutoDetect, "setRaDecJ2000", ra/15.0, dec
                        )
                    except Exception as e:
                        error_handler(e, 'center the solution in KStars')

        except Exception as e:
            if error_reporting == 'messagebox':
                error_handler(e, 'plate-solving')
            self._ui.statusbar.showMessage('Error while solving: {}'.format(str(e)))
            self._ui.solveStatusLabel.setText('Error while solving: {}'.format(str(e)))

    def _target_altaz(self):
        if self._target_pos is None:
            raise RuntimeError('No target set')

        if 'target_altaz' not in self._sync_result.cache:
            # Need to compute
            self._sync_result.cache['target_altaz'] = (
                self._backend.equatorialToHorizontal(self._target_pos.ra, self._target_pos.dec)
            )

        return self._sync_result.cache['target_altaz']

    def guideUpdate(self, scope_pos, pos_src):
        if self._sync_result is None:
            return # Can't do this without a sync
        if self._target_pos is None:
            return # Can't do this without a target
        try:
            target_altaz = self._target_altaz()
            scope_altaz = self._backend.equatorialToHorizontal(scope_pos.ra, scope_pos.dec)

            dAlt = target_altaz.alt - scope_altaz.alt
            dAz = target_altaz.az - scope_altaz.az
            time_since_last_calibration = (datetime.datetime.utcnow() - self._backend.lastImuCalibration[1]).total_seconds()

            # Validity: 0s => ~1, 5s => 0.5, 10s => ~0
            validity_score = (math.tanh((5.0 - time_since_last_calibration)/2.0) + 1.0)/2.0
            arrAlt = '↑' if (dAlt > 0) else '↓'
            arrAz = '→' if (dAz < 0) else '←'
            validity_color = int(128 + (255 - 128) * validity_score)
            textColorHex = QtGui.QColor(validity_color, validity_color, validity_color).name()
            guideLabelText = (
                f'<font color="{textColorHex}">'
                f'  {arrAlt}{abs(dAlt):.1f}° {arrAz}{abs(dAz):.1f}°'
                f'</font>'
                f'<font size="8pt" color="red">'
                f'  Cal. {int(time_since_last_calibration)}s old.'
                f'</font>'
            )

            self._guide_dialog.targetGuideLabel.setText(guideLabelText)
            epv = self._ui_eyepieceview.eyepieceView
            epv.updateGuideLabel(guideLabelText)
        except Exception as e:
            silent_error_handler(e, 'updating guide label')


    def debugEmbed(self):
        from IPython import embed
        embed(header='Debug embed in MainWindow')

    def eyepieceViewDebugEmbed(self):
        self._ui_eyepieceview.eyepieceView.debug_embed()

    def eyepieceViewShow(self):
        self._eyepieceview.show()

class Ui_MainWindow(Ui_MainWindow_BasicSetup):

    def __init__(self, eyepieceViewDialog):
        super().__init__()
        self._ui_eyepieceview = eyepieceViewDialog._ui

    def setupUi(self, MainWindow, restore_file='uistate.json'):
        super().setupUi(MainWindow)
        self._populate_alignment_targets()
        self._populate_usfs_options()
        if restore_file is not None:
            try:
                restore_state(restore_file, self, self._ui_eyepieceview)
            except Exception as e:
                logger.error('Encountered error while trying to restore UI state from {}: {}'.format(restore_file, str(e)))
        self.imuDeviceComboBox.setCurrentText(self.imuDeviceComboBox.currentText()) # Activate some signals, hopefully?

    def shutDown(self):
        try:
            save_state("uistate.json", self, self._ui_eyepieceview)
        except Exception as e:
            logger.error('Encountered error while trying to save UI state: {}'.format(str(e)))

    def _populate_alignment_targets(self):
        for alignment_target in sorted(align.ALIGN_TARGETS):
            self.alignComboBox.addItem(alignment_target)
        self.alignComboBox.addItem('Custom')
        self.alignComboBox.addItem('ask KStars')

    def _populate_usfs_options(self):
        from usfsimu import AccelParams, GyroParams

        self.imuAccelBWComboBox.clear()
        self.imuGyroBWComboBox.clear()
        self.imuAccelRangeComboBox.clear()
        self.imuGyroRangeComboBox.clear()

        self.usfs_property_map = {}
        def process(params_enum, prefix, map_key, combobox):
            self.usfs_property_map[map_key] = {
                'ui': combobox,
                'map': {},
            }
            for param in params_enum:
                if param.name.startswith(prefix):
                    key = param.name[len(prefix):]
                    combobox.addItem(key)
                    self.usfs_property_map[map_key]['map'][key] = param
                    if param.name.endswith('Default'):
                        for param_ in params_enum:
                            if not param_.name.startswith(prefix):
                                continue
                            if param_.name.endswith('Default'):
                                continue
                            if param_.value == param.value:
                                break
                        if param_.value == param.value:
                            combobox.setCurrentText(param_.name[len(prefix):])
        process(AccelParams, 'BW_', 'acc_lpf_bw', self.imuAccelBWComboBox)
        process(AccelParams, 'FS_', 'acc_fs', self.imuAccelRangeComboBox)
        process(GyroParams, 'BW_', 'gyro_lpf_bw', self.imuGyroBWComboBox)
        process(GyroParams, 'FS_', 'gyro_fs', self.imuGyroRangeComboBox)
