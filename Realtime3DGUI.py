import ctypes as c
import numpy as np
import matplotlib.pyplot as plt
import time
from RealtimeFlowOCT.PyScanPattern.Patterns import Figure8ScanPattern, RasterScanPattern, RoseScanPattern, \
    BidirectionalRasterScanPattern

from RealtimeOCT import RealtimeOCTController

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QGroupBox, QComboBox

from PyQt5 import uic
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QTimer
from pyqtgraph import PlotWidget, ImageView
import pyqtgraph as pyqtgraph
import threading

from Widgets.graph import OCTViewer, SpectrumPlotWidget, RunningPlotWidget

import sys
import os

import multiprocessing as mp
from queue import Empty, Full

CAM = 'img1'
AO_X = 'Dev1/ao1'
AO_Y = 'Dev1/ao2'
AO_LT = 'Dev1/ao0'
AO_FT = 'Dev1/ao3'

ALINE_SIZE = 2048
PI = np.pi
TRIGGER_GAIN = 4
NUMBER_OF_IMAQ_BUFFERS = 8
INTPDK = 0.305

ROI_OFFSET = 20

d3 = 16
NUMBER_OF_ALINES_PER_B = d3
NUMBER_OF_BLINES = d3
ROI_SIZE = d3

UPSAMPLE_FACTOR = 3

REFRESH_RATE = 220  # hz

ASYNC_WAIT_DEBUG = 0

PLOT_RANGE = 4

def reshape_unidirectional_frame(A, z, x, b, dtype=np.complex64):
    """
    Assumes shape of form [z, x, b]
    """
    reshaped = np.empty([z, x, b], dtype=dtype)
    seek = 0
    for j in range(b):
        for i in range(x):
            reshaped[:, i, j] = A[seek * z:seek * z + z]
            seek += 1
    return reshaped


def reshape_bidirectional_frame(A, z, x, b, dtype=np.complex64):
    """
    Assumes shape of form [z, x, b]
    """
    reshaped = np.empty([z, x, b], dtype=dtype)
    seek = 0
    lr = True
    for j in range(b):
        if lr:
            for i in range(x):
                reshaped[:, i, j] = A[seek * z:seek * z + z]
                seek += 1
        else:
            for i in range(x)[::-1]:
                reshaped[:, i, j] = A[seek * z:seek * z + z]
                seek += 1
        lr = not lr
    return reshaped


def hann2d(dim):
    xw = np.hanning(dim[0])
    yw = np.hanning(dim[1])
    xwin = np.repeat(xw[:, np.newaxis], dim[1], axis=1)
    ywin = np.transpose(np.repeat(yw[:, np.newaxis], dim[0], axis=1))
    hanningwindow = xwin * ywin
    return hanningwindow


def hanning_cube(dim):
    w = np.hanning(dim)
    L = w.shape[0]
    m1 = np.outer(np.ravel(w), np.ravel(w))
    win1 = np.tile(m1, np.hstack([L, 1, 1]))
    m2 = np.outer(np.ravel(w), np.ones([1, L]))
    win2 = np.tile(m2, np.hstack([L, 1, 1]))
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    win = np.multiply(win1, win2)
    return win

def blackman_cube(dim, pad=0):
    w = np.pad(np.blackman(dim), pad)
    L = w.shape[0]
    m1 = np.outer(np.ravel(w), np.ravel(w))
    win1 = np.tile(m1, np.hstack([L, 1, 1]))
    m2 = np.outer(np.ravel(w), np.ones([1, L]))
    win2 = np.tile(m2, np.hstack([L, 1, 1]))
    win2 = np.transpose(win2, np.hstack([1, 2, 0]))
    win = np.multiply(win1, win2)
    return win


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.mainWidget = QtWidgets.QWidget()

        self._controller = RealtimeOCTController(CAM, AO_X, AO_Y, AO_LT, AO_FT)

        # Plots

        self.tn_view = OCTViewer(title='Frame at dt')
        self.r_view = OCTViewer(title='Correlogram')
        self.bg_plot = SpectrumPlotWidget(title='Average raw spectrum')

        self.mot_x_plot = RunningPlotWidget(window_length=128, miny=-PLOT_RANGE, maxy=PLOT_RANGE, title='x', legend=['x'])
        self.mot_y_plot = RunningPlotWidget(window_length=128, miny=-PLOT_RANGE, maxy=PLOT_RANGE, title='y', legend=['y'])
        self.mot_z_plot = RunningPlotWidget(window_length=128, miny=-PLOT_RANGE, maxy=PLOT_RANGE, title='z', legend=['z'])

        self.mainlayout = QtWidgets.QGridLayout()

        self.mainlayout.addWidget(self.r_view, 0, 1, 1, 1)
        self.mainlayout.addWidget(self.tn_view, 0, 2, 1, 1)

        self.mainlayout.addWidget(self.bg_plot, 0, 3, 1, 1)

        self.mainlayout.addWidget(self.mot_x_plot, 1, 1, 1, 1)
        self.mainlayout.addWidget(self.mot_y_plot, 1, 2, 1, 1)

        self.mainlayout.addWidget(self.mot_z_plot, 1, 3, 1, 1)

        # Controls ----

        self._control_panel = QtWidgets.QWidget()
        self._control_layout = QtWidgets.QFormLayout()

        self._start_scan_button = QtWidgets.QPushButton()
        self._start_scan_button.setText("Start scan")
        self._control_layout.addRow(self._start_scan_button)
        self._start_scan_button.clicked.connect(self._start_scan)

        self._stop_scan_button = QtWidgets.QPushButton()
        self._stop_scan_button.setText("Stop scan")
        self._control_layout.addRow(self._stop_scan_button)
        self._stop_scan_button.clicked.connect(self._stop_scan)

        self._start_output_button = QtWidgets.QPushButton()
        self._start_output_button.setText("Start output")
        self._control_layout.addRow(self._start_output_button)
        self._start_output_button.clicked.connect(self._start_output)

        self._stop_output_button = QtWidgets.QPushButton()
        self._stop_output_button.setText("Stop output")
        self._control_layout.addRow(self._stop_output_button)
        self._stop_output_button.clicked.connect(self._stop_output)

        # self._nacq_spin = QtWidgets.QSpinBox()
        # self._nacq_spin.setRange(100, 100000000)
        # self._nacq_spin.setValue(10000)
        # self._control_layout.addRow(QtWidgets.QLabel("N to acquire"), self._nacq_spin)

        # self._fname_edit = QtWidgets.QLineEdit()
        # self._fname_edit.setText("output")
        # self._control_layout.addRow(QtWidgets.QLabel("File name"), self._fname_edit)
        #
        self._ref_button = QtWidgets.QPushButton()
        self._ref_button.setText("Update reference")
        self._control_layout.addRow(self._ref_button)
        self._ref_button.pressed.connect(self._controller.update_motion_reference)

        self._adist_spin = QtWidgets.QDoubleSpinBox()
        self._adist_spin.setDecimals(4)
        self._adist_spin.setRange(0.00001, 0.1)
        self._adist_spin.setValue(0.0015)
        self._adist_spin.setSingleStep(0.0001)
        self._control_layout.addRow(QtWidgets.QLabel("A-line spacing (mm)"), self._adist_spin)
        self._adist_spin.valueChanged.connect(self.set_scan_pattern)

        self._bidirectional_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Bidirectional raster scan"), self._bidirectional_check)
        self._bidirectional_check.toggled.connect(self.set_scan_pattern)

        self._flyback_duty_spin = QtWidgets.QDoubleSpinBox()
        self._flyback_duty_spin.setRange(0.01, 0.99)
        self._flyback_duty_spin.setValue(0.05)
        self._flyback_duty_spin.setSingleStep(0.01)
        self._control_layout.addRow(QtWidgets.QLabel("Flyback duty"), self._flyback_duty_spin)
        self._flyback_duty_spin.valueChanged.connect(self.set_scan_pattern)

        self._exposure_percentage_spin = QtWidgets.QDoubleSpinBox()
        self._exposure_percentage_spin.setRange(0.01, 0.99)
        self._exposure_percentage_spin.setValue(0.75)
        self._exposure_percentage_spin.setSingleStep(0.01)
        self._control_layout.addRow(QtWidgets.QLabel("Exposure %"), self._exposure_percentage_spin)
        self._exposure_percentage_spin.valueChanged.connect(self.set_scan_pattern)

        self._npeak_spin = QtWidgets.QSpinBox()
        self._npeak_spin.setRange(0, int((NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) / 2))
        self._npeak_spin.setValue(4)
        self._control_layout.addRow(QtWidgets.QLabel("Centroid peak N"), self._npeak_spin)
        self._npeak_spin.valueChanged.connect(self._update_motion_parameters)

        self._hpf_spin = QtWidgets.QSpinBox()
        self._hpf_spin.setRange(0, int((NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) / 2))
        self._hpf_spin.setValue(int(1.5 * UPSAMPLE_FACTOR))
        self._control_layout.addRow(QtWidgets.QLabel("HPF width"), self._hpf_spin)
        self._hpf_spin.valueChanged.connect(self._update_motion_parameters)

        self._mip_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Display MIP"), self._mip_check)

        self._xsection_radio = QtWidgets.QRadioButton()
        self._xsection_radio.setChecked(True)
        self._control_layout.addRow(QtWidgets.QLabel("slow axis cross-section"), self._xsection_radio)
        self._ysection_radio = QtWidgets.QRadioButton()
        self._control_layout.addRow(QtWidgets.QLabel("Fast axis cross-section"), self._ysection_radio)
        self._zsection_radio = QtWidgets.QRadioButton()
        self._control_layout.addRow(QtWidgets.QLabel("Axial cross-section"), self._zsection_radio)

        self._slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slice_slider.setRange(0, (NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) - 1)
        self._slice_slider.setSingleStep(1)
        self._control_layout.addRow(QtWidgets.QLabel("Slice"), self._slice_slider)

        self._trigger_offset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._trigger_offset_slider.setValue(8)
        self._trigger_offset_slider.setSingleStep(1)
        self._trigger_offset_slider.valueChanged.connect(self._update_scan_pattern)
        self._trigger_offset_label = QtWidgets.QLabel("Trigger skew (samples)")
        self._control_layout.addRow(self._trigger_offset_label, self._trigger_offset_slider)

        self._x_factor_spin = QtWidgets.QDoubleSpinBox()
        self._x_factor_spin.setRange(-5, 5)
        self._x_factor_spin.setValue(-0.5)
        self._x_factor_spin.setSingleStep(0.1)
        self._control_layout.addRow(QtWidgets.QLabel("X DAC scale factor"), self._x_factor_spin)
        self._x_factor_spin.valueChanged.connect(self._update_motion_parameters)

        self._y_factor_spin = QtWidgets.QDoubleSpinBox()
        self._y_factor_spin.setRange(-5, 5)
        self._y_factor_spin.setValue(-0.5)
        self._y_factor_spin.setSingleStep(0.1)
        self._control_layout.addRow(QtWidgets.QLabel("Y DAC scale factor"), self._y_factor_spin)
        self._y_factor_spin.valueChanged.connect(self._update_motion_parameters)

        self._z_factor_spin = QtWidgets.QDoubleSpinBox()
        self._z_factor_spin.setRange(-5, 5)
        self._z_factor_spin.setValue(-0.5)
        self._z_factor_spin.setSingleStep(0.1)
        self._control_layout.addRow(QtWidgets.QLabel("Z DAC scale factor"), self._z_factor_spin)
        self._z_factor_spin.valueChanged.connect(self._update_motion_parameters)

        self._d_spin = QtWidgets.QDoubleSpinBox()
        self._d_spin.setDecimals(3)
        self._d_spin.setValue(0.99)
        self._d_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF pos decay (d)"), self._d_spin)
        self._d_spin.valueChanged.connect(self._update_motion_parameters)

        self._g_spin = QtWidgets.QDoubleSpinBox()
        self._g_spin.setDecimals(3)
        self._g_spin.setValue(0.99)
        self._g_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF v decay (g)"), self._g_spin)
        self._g_spin.valueChanged.connect(self._update_motion_parameters)

        self._q_spin = QtWidgets.QDoubleSpinBox()
        self._q_spin.setDecimals(3)
        self._q_spin.setValue(0.0100)
        self._q_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF process noise (q)"), self._q_spin)
        self._q_spin.valueChanged.connect(self._update_motion_parameters)

        self._r_spin = QtWidgets.QDoubleSpinBox()
        self._r_spin.setDecimals(3)
        self._r_spin.setValue(4)
        self._r_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF meas noise (r)"), self._r_spin)
        self._r_spin.valueChanged.connect(self._update_motion_parameters)

        self._d = np.ones(3).astype(np.float64) * self._d_spin.value()
        self._g = np.ones(3).astype(np.float64) * self._g_spin.value()
        self._q = np.ones(3).astype(np.float64) * self._q_spin.value()
        self._r = np.ones(3).astype(np.float64) * self._r_spin.value()

        self._control_panel.setLayout(self._control_layout)

        self.mainlayout.addWidget(self._control_panel, 0, 0, 2, 1)

        # -------

        self.mainWidget.setLayout(self.mainlayout)
        self.setCentralWidget(self.mainWidget)

        self._define_scan_pattern()
        time.sleep(ASYNC_WAIT_DEBUG)

        self._window = np.hanning(2048).astype(np.float32)  # A-line window

        self._controller.configure(self._pat.get_sample_rate(), ALINE_SIZE, NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES,
                                   NUMBER_OF_IMAQ_BUFFERS, roi_offset=ROI_OFFSET, roi_size=ROI_SIZE)
        time.sleep(ASYNC_WAIT_DEBUG)

        self._controller.set_processing(INTPDK, self._window)
        time.sleep(ASYNC_WAIT_DEBUG)

        self._update_scan_pattern()
        time.sleep(ASYNC_WAIT_DEBUG)

        self._spectrum_buffer = np.empty(ALINE_SIZE, dtype=np.float32)
        self._grab_buffer = np.empty(ROI_SIZE * NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES, dtype=np.complex64)
        self._grab_buffer_upsampled = np.empty(ROI_SIZE * UPSAMPLE_FACTOR * NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR * NUMBER_OF_BLINES * UPSAMPLE_FACTOR, dtype=np.complex64)

        self._mot_buffer = np.empty(4, dtype=np.float64)

        self._motion_output_enabled = False
        self._timer = QTimer()

    def _start_scan(self):
        self._controller.start_scan()
        self._timer.timeout.connect(self._plot)
        self._timer.start(1 / REFRESH_RATE * 1000)

    def _stop_scan(self):
        self._timer.stop()
        self._controller.stop_scan()

    def _stop_output(self):
        self._controller.stop_motion_output()
        self._motion_output_enabled = False

    def _start_output(self):
        if isinstance(self._pat, BidirectionalRasterScanPattern):
            bd = True
        else:
            bd = False
        self._d = np.ones(3).astype(np.float64) * self._d_spin.value()
        self._g = np.ones(3).astype(np.float64) * self._g_spin.value()
        self._q = np.ones(3).astype(np.float64) * self._q_spin.value()
        self._r = np.ones(3).astype(np.float64) * self._r_spin.value()
        self._generate_windows()
        self._controller.start_motion_output(
            np.array([NUMBER_OF_ALINES_PER_B, NUMBER_OF_ALINES_PER_B, NUMBER_OF_ALINES_PER_B]).astype(int),
            np.array([self._x_factor_spin.value(), self._y_factor_spin.value(), self._z_factor_spin.value()]).astype(np.float64),
            UPSAMPLE_FACTOR, self._npeak_spin.value(), self._spectral_window3d, self._spatial_window3d, self._d, self._g, self._q, self._r, bidirectional=bd)
        self._motion_output_enabled = True

    def _update_motion_parameters(self):
        if self._motion_output_enabled:
            if isinstance(self._pat, BidirectionalRasterScanPattern):
                bd = True
            else:
                bd = False
            self._generate_windows()
            self._d = np.ones(3).astype(np.float64) * self._d_spin.value()
            self._g = np.ones(3).astype(np.float64) * self._g_spin.value()
            self._q = np.ones(3).astype(np.float64) * self._q_spin.value()
            self._r = np.ones(3).astype(np.float64) * self._r_spin.value()
            self._controller.update_motion_parameters(np.array([self._x_factor_spin.value(), self._y_factor_spin.value(), self._z_factor_spin.value()]).astype(np.float64),
                                                      int(self._npeak_spin.value()), self._spectral_window3d, self._spatial_window3d, self._d, self._g, self._q, self._r, bidirectional=bd)

    def _plot(self):
        got = self._controller.grab_frame(self._grab_buffer)
        try:
            if got > -1 and not np.isnan(np.sum(self._grab_buffer)):
                if isinstance(self._pat, BidirectionalRasterScanPattern):
                    self._display_buffer = np.abs(reshape_bidirectional_frame(self._grab_buffer, ROI_SIZE, NUMBER_OF_ALINES_PER_B, NUMBER_OF_BLINES))
                else:
                    self._display_buffer = np.abs(reshape_unidirectional_frame(self._grab_buffer, ROI_SIZE, NUMBER_OF_ALINES_PER_B, NUMBER_OF_BLINES))

                if self._xsection_radio.isChecked():
                    if self._mip_check.isChecked():
                        self._display_buffer = np.max(self._display_buffer, axis=1)
                    else:
                        self._display_buffer = self._display_buffer[:, np.floor_divide(self._slice_slider.value(), UPSAMPLE_FACTOR), :]
                elif self._ysection_radio.isChecked():
                    if self._mip_check.isChecked():
                        self._display_buffer = np.max(self._display_buffer, axis=2)
                    else:
                        self._display_buffer = self._display_buffer[:, :, np.floor_divide(self._slice_slider.value(), UPSAMPLE_FACTOR)]
                else:
                    if self._mip_check.isChecked():
                        self._display_buffer = np.max(self._display_buffer, axis=0)
                    else:
                        self._display_buffer = self._display_buffer[np.floor_divide(self._slice_slider.value(), UPSAMPLE_FACTOR), :, :]

                self.tn_view.setImage(self._display_buffer)
                self._controller.grab_spectrum(self._spectrum_buffer)
                self.bg_plot.set_spectrum_data(self._spectrum_buffer)

                if self._motion_output_enabled:

                    # Copying the 3D correlogram out of the working buffer is VERY slow

                    # self._controller.grab_motion_correlogram(self._grab_buffer_upsampled)
                    # self._display_buffer_upsampled = np.abs(reshape_unidirectional_frame(self._grab_buffer_upsampled, ROI_SIZE * UPSAMPLE_FACTOR, NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR, NUMBER_OF_BLINES * UPSAMPLE_FACTOR))
                    #
                    # if self._xsection_radio.isChecked():
                    #     if self._mip_check.isChecked():
                    #         self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=1)
                    #     else:
                    #         self._display_buffer_upsampled = self._display_buffer_upsampled[:, self._slice_slider.value(), :]
                    # elif self._ysection_radio.isChecked():
                    #     if self._mip_check.isChecked():
                    #         self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=2)
                    #     else:
                    #         self._display_buffer_upsampled = self._display_buffer_upsampled[:, :, self._slice_slider.value()]
                    # else:
                    #     if self._mip_check.isChecked():
                    #         self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=0)
                    #     else:
                    #         self._display_buffer_upsampled = self._display_buffer_upsampled[self._slice_slider.value(), :, :]
                    #
                    # self.r_view.setImage(np.fft.fftshift(self._display_buffer_upsampled))
                    # self.r_view.setImage(self._display_buffer_upsampled)
                    if not self._controller.grab_motion_vector(self._mot_buffer):  # Returns 0 if dequeue is successful
                        self.mot_x_plot.append_to_plot([self._mot_buffer[0]])
                        self.mot_y_plot.append_to_plot([self._mot_buffer[1]])
                        self.mot_z_plot.append_to_plot([self._mot_buffer[2]])
            else:
                # print('Frame contains NaN')
                pass
        except RuntimeWarning:
            # print('Invalid data')
            pass

    def _generate_windows(self):
        pad = int(((NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) - NUMBER_OF_ALINES_PER_B) / 2)
        self._spatial_window3d = blackman_cube(NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR, pad=pad)
        self._spectral_window3d = np.ones(np.shape(self._spatial_window3d))

        self._spectral_window3d[0:self._hpf_spin.value(), 0:self._hpf_spin.value(), 0:self._hpf_spin.value()] = 0
        self._spectral_window3d = np.roll(self._spectral_window3d, int(-self._hpf_spin.value() / 2), axis=0)
        self._spectral_window3d = np.roll(self._spectral_window3d, int(-self._hpf_spin.value() / 2), axis=1)
        self._spectral_window3d = np.roll(self._spectral_window3d, int(-self._hpf_spin.value() / 2), axis=2)

        self._spatial_window3d = self._spatial_window3d.astype(np.float32).flatten()
        self._spectral_window3d = self._spectral_window3d.astype(np.float32).flatten()

    def _define_scan_pattern(self):
        fovwidth = (NUMBER_OF_ALINES_PER_B - 1) * self._adist_spin.value()
        exp = self._exposure_percentage_spin.value()
        fb = self._flyback_duty_spin.value()
        if self._bidirectional_check.isChecked():
            self._pat = BidirectionalRasterScanPattern()
        else:
            self._pat = RasterScanPattern()

        self._pat.generate(alines=NUMBER_OF_ALINES_PER_B, blines=NUMBER_OF_BLINES, fov=[fovwidth, fovwidth],
                           samples_on=1, samples_off=1, exposure_percentage=exp, flyback_duty=fb)
        self._trigger_offset_slider.setRange(-np.floor_divide(len(self._pat.get_line_trig()), 8),
                                             np.floor_divide(len(self._pat.get_line_trig()), 8))
        print("Generated", type(self._pat), "raster scan rate", self._pat.get_pattern_rate(), 'Hz')

    def _update_scan_pattern(self):
        self._trigger_offset_label.setText('Trigger skew (' + str(self._trigger_offset_slider.value()) + ')')
        self._xsig = self._pat.get_x() * 22
        self._ysig = self._pat.get_y() * 18
        self._ltsig = np.roll(self._pat.get_line_trig() * TRIGGER_GAIN, self._trigger_offset_slider.value())
        self._ftsig = np.roll(self._pat.get_frame_trig() * TRIGGER_GAIN, self._trigger_offset_slider.value())
        # self._ftsig = np.zeros(len(self._pat.get_frame_trig()))
        # self._ftsig[-2::] = TRIGGER_GAIN
        self._controller.set_scan(self._xsig, self._ysig, self._ltsig, self._ftsig)

    def set_scan_pattern(self):
        self._define_scan_pattern()
        self._update_scan_pattern()

    def closeEvent(self, event):
        self._timer.stop()
        self._controller.stop_scan()
        # self._controller.stop_motion_output()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
