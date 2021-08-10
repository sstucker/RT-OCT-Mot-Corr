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
TRIGGER_GAIN = 6
NUMBER_OF_IMAQ_BUFFERS = 8
INTPDK = 0.305

ROI_OFFSET = 0

d3 = 16
NUMBER_OF_ALINES_PER_B = d3
NUMBER_OF_BLINES = d3
ROI_SIZE = d3

UPSAMPLE_FACTOR = 4
NPEAK = 2

REFRESH_RATE = 200  # hz

ASYNC_WAIT_DEBUG = 3


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
    reshaped = np.roll(reshaped, -2, axis=1)
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


def hann_cube(dim):
    w = np.hanning(dim)
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

        self.mot_x_plot = RunningPlotWidget(window_length=128, miny=-16, maxy=16, title='x', legend=['x'])
        self.mot_y_plot = RunningPlotWidget(window_length=128, miny=-16, maxy=16, title='y', legend=['y'])
        self.mot_z_plot = RunningPlotWidget(window_length=128, miny=-16, maxy=16, title='z', legend=['z'])

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

        # self._q_spin = QtWidgets.QDoubleSpinBox()
        # self._q_spin.setDecimals(5)
        # self._q_spin.setValue(0.0100)
        # self._q_spin.setRange(-9999, 9999)
        # self._control_layout.addRow(QtWidgets.QLabel("q"), self._q_spin)
        #
        # self._r_spin = QtWidgets.QDoubleSpinBox()
        # self._r_spin.setValue(4)
        # self._r_spin.setRange(-9999, 9999)
        # self._control_layout.addRow(QtWidgets.QLabel("r"), self._r_spin)

        # self._rot_spin = QtWidgets.QDoubleSpinBox()
        # self._rot_spin.setValue(42.00)
        # self._rot_spin.setDecimals(2)
        # self._rot_spin.setSingleStep(0.01)
        # self._rot_spin.setRange(-360, 360)
        # self._control_layout.addRow(QtWidgets.QLabel("Rotation (deg)"), self._rot_spin)

        self._adist_spin = QtWidgets.QDoubleSpinBox()
        self._adist_spin.setDecimals(5)
        self._adist_spin.setRange(0.00001, 0.1)
        self._adist_spin.setValue(0.0005)
        self._adist_spin.setSingleStep(0.0001)
        self._control_layout.addRow(QtWidgets.QLabel("A-line spacing (mm)"), self._adist_spin)
        self._adist_spin.valueChanged.connect(self.set_scan_pattern)

        self._bidirectional_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Bidirectional raster scan"), self._bidirectional_check)
        self._bidirectional_check.toggled.connect(self.set_scan_pattern)

        self._flyback_duty_spin = QtWidgets.QDoubleSpinBox()
        self._flyback_duty_spin.setRange(0.1, 0.9)
        self._flyback_duty_spin.setValue(0.1)
        self._flyback_duty_spin.setSingleStep(0.01)
        self._control_layout.addRow(QtWidgets.QLabel("Flyback duty"), self._flyback_duty_spin)
        self._flyback_duty_spin.valueChanged.connect(self.set_scan_pattern)

        self._exposure_percentage_spin = QtWidgets.QDoubleSpinBox()
        self._exposure_percentage_spin.setRange(0.1, 0.9)
        self._exposure_percentage_spin.setValue(0.7)
        self._exposure_percentage_spin.setSingleStep(0.01)
        self._control_layout.addRow(QtWidgets.QLabel("Exposure %"), self._exposure_percentage_spin)
        self._exposure_percentage_spin.valueChanged.connect(self.set_scan_pattern)

        # self._npeak_spin = QtWidgets.QSpinBox()
        # self._npeak_spin.setRange(0, 4)
        # self._npeak_spin.setValue(2)
        # self._control_layout.addRow(QtWidgets.QLabel("Centroid peak N"), self._npeak_spin)

        # self._varwin_spin = QtWidgets.QSpinBox()
        # self._varwin_spin.setRange(0, 1000)
        # self._varwin_spin.setValue(8)
        # self._control_layout.addRow(QtWidgets.QLabel("Variance window N"), self._varwin_spin)

        # self._d_spin = QtWidgets.QDoubleSpinBox()
        # self._d_spin.setValue(1)
        # self._control_layout.addRow(QtWidgets.QLabel("d"), self._d_spin)
        #
        # self._g_spin = QtWidgets.QDoubleSpinBox()
        # self._g_spin.setValue(1)
        # self._control_layout.addRow(QtWidgets.QLabel("g"), self._g_spin)

        # self._zstart_spin = QtWidgets.QSpinBox()
        # self._zstart_spin.setValue(10)
        # self._control_layout.addRow(QtWidgets.QLabel("Z-start ROI"), self._zstart_spin)
        #
        # self._x_factor_spin = QtWidgets.QDoubleSpinBox()
        # self._x_factor_spin.setValue(-1)
        # self._x_factor_spin.setRange(-5, 5)
        # self._control_layout.addRow(QtWidgets.QLabel("X conv factor"), self._x_factor_spin)
        #
        # self._y_factor_spin = QtWidgets.QDoubleSpinBox()
        # self._y_factor_spin.setValue(-1)
        # self._y_factor_spin.setRange(-5, 5)
        # self._control_layout.addRow(QtWidgets.QLabel("Y conv factor"), self._y_factor_spin)
        #
        # self._z_factor_spin = QtWidgets.QDoubleSpinBox()
        # self._z_factor_spin.setValue(-1)
        # self._z_factor_spin.setRange(-5, 5)
        # self._control_layout.addRow(QtWidgets.QLabel("Z conv factor"), self._z_factor_spin)

        self._mip_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Display MIP"), self._mip_check)

        self._xsection_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Display fast cross-section"), self._xsection_check)

        self._control_panel.setLayout(self._control_layout)

        self.mainlayout.addWidget(self._control_panel, 0, 0, 2, 1)

        # -------

        self.mainWidget.setLayout(self.mainlayout)
        self.setCentralWidget(self.mainWidget)

        self._define_scan_pattern()
        time.sleep(ASYNC_WAIT_DEBUG)

        self._window3d = hann_cube(NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR).astype(np.float32).flatten()
        self._window3d = np.ones(len(self._window3d)).astype(np.float32)
        # plt.plot(self._window3d)
        # plt.show()

        self._window = np.hanning(2048).astype(np.float32)
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
        self._controller.start_motion_output(
            np.array([NUMBER_OF_ALINES_PER_B, NUMBER_OF_ALINES_PER_B, NUMBER_OF_ALINES_PER_B]).astype(int),
            UPSAMPLE_FACTOR, NPEAK, self._window3d)
        self._motion_output_enabled = True

    def _plot(self):
        got = self._controller.grab_frame(self._grab_buffer)
        if got > -1 and not np.isnan(np.sum(self._grab_buffer)):
            if isinstance(self._pat, BidirectionalRasterScanPattern):
                self._display_buffer = np.abs(reshape_bidirectional_frame(self._grab_buffer, ROI_SIZE, NUMBER_OF_ALINES_PER_B, NUMBER_OF_BLINES))
            else:
                self._display_buffer = np.abs(reshape_unidirectional_frame(self._grab_buffer, ROI_SIZE, NUMBER_OF_ALINES_PER_B, NUMBER_OF_BLINES))

            if self._mip_check.isChecked():
                self._display_buffer = np.max(self._display_buffer, axis=0)
            else:
                if self._xsection_check.isChecked():
                    self._display_buffer = self._display_buffer[:, :, int(NUMBER_OF_BLINES / 2)]
                else:
                    self._display_buffer = self._display_buffer[:, int(NUMBER_OF_ALINES_PER_B / 2), :]
            self.tn_view.setImage(self._display_buffer)
            self._controller.grab_spectrum(self._spectrum_buffer)
            self.bg_plot.set_spectrum_data(self._spectrum_buffer)
            if self._motion_output_enabled:
                self._controller.grab_motion_correlogram(self._grab_buffer_upsampled)
                self._display_buffer_upsampled = np.abs(reshape_unidirectional_frame(self._grab_buffer_upsampled, ROI_SIZE * UPSAMPLE_FACTOR, NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR, NUMBER_OF_BLINES * UPSAMPLE_FACTOR))
                if self._mip_check.isChecked():
                    self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=0)
                else:
                    if self._xsection_check.isChecked():
                        self._display_buffer_upsampled = self._display_buffer_upsampled[:, :, int((NUMBER_OF_BLINES * UPSAMPLE_FACTOR) / 2)]
                    else:
                        self._display_buffer_upsampled = self._display_buffer_upsampled[:, int((NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) / 2), :]
                self.r_view.setImage(np.fft.fftshift(self._display_buffer_upsampled))
                if self._controller.grab_motion_vector(self._mot_buffer):  # Returns true if correlation result is ready
                    # print(self._mot_buffer)
                    self.mot_x_plot.append_to_plot([self._mot_buffer[0]])
                    self.mot_y_plot.append_to_plot([self._mot_buffer[1]])
                    self.mot_z_plot.append_to_plot([self._mot_buffer[2]])
        else:
            print('Frame contains NaN')

    def _define_scan_pattern(self):
        fovwidth = (NUMBER_OF_ALINES_PER_B - 1) * self._adist_spin.value()
        exp = self._exposure_percentage_spin.value()
        fb = self._flyback_duty_spin.value()
        if self._bidirectional_check.isChecked():
            self._pat = BidirectionalRasterScanPattern()
        else:
            self._pat = RasterScanPattern()

        self._pat.generate(alines=NUMBER_OF_ALINES_PER_B, blines=NUMBER_OF_BLINES, fov=[fovwidth, fovwidth],
                           samples_on=2, exposure_percentage=exp, flyback_duty=fb)
        print("Generated", type(self._pat), "raster scan rate", self._pat.get_pattern_rate(), 'Hz')

    def _update_scan_pattern(self):
        self._xsig = self._pat.get_x() * 22
        self._ysig = self._pat.get_y() * 18
        self._ltsig = self._pat.get_line_trig() * TRIGGER_GAIN
        self._ftsig = self._pat.get_frame_trig() * TRIGGER_GAIN
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
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
