import logging
logger = logging.getLogger('UIState')

import json
from PyQt5.QtWidgets import *

SAVE_WIDGETS = {
    'indiHostLineEdit',
    'indiDeviceLineEdit',
    'indiPortSpinBox',
    'imuSerialPortLineEdit',
    'imuBaudRateComboBox',
    'exposureSpinBox',
    'alignComboBox',
    'alignRALineEdit',
    'alignDecLineEdit',
    'alignArcsecPerPixelHintSpinBox',
    'siteLatitudeLineEdit',
    'siteLongitudeLineEdit',
    'siteElevationSpinBox',
    'siteTemperatureSpinBox',
    'platformTimeSpinBox',
    'smudgeThresholdSpinBox',
    'solveToKStarsCheckBox',
    'sepCheckBox',
    'sepThreshold',
    'fovSpinBox',
    'focuserAngleSpinBox',
    'imageSizeSpinBox',
    'eyepieceViewportOnScopeCheckBox',
    'eyepieceViewportAngleSpinBox',
    'imuAccelBWComboBox',
    'imuGyroBWComboBox',
    'imuAccelRangeComboBox',
    'imuGyroRangeComboBox',
    'imuAccelMergeRateSpinBox',
    'imuMagMergeRateSpinBox',
    'tempSourceComboBox',
    'binningSpinBox',
    'starCountCutoffSpinBox',
    'imuLowpassCheckBox',
    'imuLowpassMemorySpinBox',
    'imuI2cPortComboBox',
    'imuDeviceComboBox',
    'altAzTrackingCheckBox',
}

def save_widget(state, widget_name, widget):
    if isinstance(widget, QComboBox):
        index = widget.currentIndex()
        text = widget.itemText(index)
        state[widget_name] = {'index': index, 'text': text}
    elif isinstance(widget, QLineEdit):
        state[widget_name] = widget.text()
    elif isinstance(widget, QCheckBox):
        state[widget_name] = widget.isChecked() # Bi-state
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        state[widget_name] = widget.value()
    else:
        raise NotImplementedError(
            'Do not know how to save widget of type {}'.format(type(widget))
        )

    return state

def restore_widget(widget_name, widget, widget_state):
    if isinstance(widget, QComboBox):
        assert type(widget_state) is dict
        text = widget_state['text']
        cbxIndex = widget.findText(text)
        if cbxIndex == -1:
            widget.insertItems(0, [text])
            cbxIndex = widget.findText(text)
            widget.setCurrentIndex(cbxIndex)
        else:
            widget.setCurrentIndex(cbxIndex)
    elif isinstance(widget, QLineEdit):
        widget.setText(widget_state)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(widget_state)
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(widget_state)
    else:
        raise NotImplementedError(
            'Do not know how to restore widget of type {}'.format(type(widget))
        )

def get_state(*uis):
    widgets = set(SAVE_WIDGETS)
    saved_widgets = set()
    state = {}
    for widget in widgets:
        for ui in uis:
            if widget in ui.__dict__:
                state = save_widget(state, widget, ui.__dict__[widget])
                saved_widgets.add(widget)
                break
    if len(widgets - saved_widgets) > 0:
        logger.error('Note: The state of the following widgets was not saved: {}. Inspect the names of the widgets!'.format(list(widgets - saved_widgets)))
        # FIXME: Assert fail?
    return state

def set_state(state, *uis):
    widgets = set(state)
    loaded_widgets = set()
    for widget, widget_state in state.items():
        for ui in uis:
            if widget in ui.__dict__:
                restore_widget(widget, ui.__dict__[widget], widget_state)
                loaded_widgets.add(widget)
                break
    if len(widgets - loaded_widgets) > 0:
        logger.error('Note: The state of the following widgets was not restored: {}. Inspect the names of the widgets!'.format(list(widgets - loaded_widgets)))

def save_state(save_file, *uis):
    state = get_state(*uis)
    with open(save_file, 'w') as jf:
        json.dump(state, jf)

def restore_state(restore_file, *uis):
    with open(restore_file, 'r') as jf:
        state = json.load(jf)
    set_state(state, *uis)
