import ctypes as c
import numpy as np
import matplotlib.pyplot as plt
import time
from pyscanpatterns.scanpatterns import RasterScanPattern

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

# from CvStream import CvStream

V_TO_MM = 14.06  # Calibrated 1/27/2022

CAM = 'img1'
AO_X = 'Dev1/ao1'
AO_Y = 'Dev1/ao2'
AO_LT = 'Dev1/ao0'
AO_FT = 'Dev1/ao3'

AO_DX = 'Dev1/ao4'
AO_DY = 'Dev1/ao5'
AO_DZ = 'Dev1/ao6'

ALINE_SIZE = 2048
PI = np.pi
TRIGGER_GAIN = 5
NUMBER_OF_IMAQ_BUFFERS = 4
INTPDK = 0.305

ROI_OFFSET = 60

d3 = 16
NUMBER_OF_ALINES_PER_B = d3
NUMBER_OF_BLINES = d3
ROI_SIZE = d3

N_LAG = 2
UPSAMPLE_FACTOR = 2

REFRESH_RATE = 197.6  # hz
WIN_LEN = 128

ASYNC_WAIT_DEBUG = 0

PLOT_RANGE = 6

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
        # self.bg_plot = SpectrumPlotWidget(title='Average raw spectrum')

        plt_x = np.arange(0, WIN_LEN) * (1 / REFRESH_RATE)

        self.mot_x_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE, maxy=PLOT_RANGE, title='x', legend=['x'], labels={'left': 'px', 'bottom': 'Time (s)'}, x=plt_x)
        self.mot_y_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE, maxy=PLOT_RANGE, title='y', legend=['y'], labels={'left': 'px', 'bottom': 'Time (s)'}, x=plt_x)
        # self.mot_z_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE, maxy=PLOT_RANGE, title='z', legend=['z'], labels={'left': 'px', 'bottom': 'Time (s)'}, x=plt_x)

        self.mot_dx_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE / 2, maxy=PLOT_RANGE / 2, title='dx', legend=['dx'], labels={'left': 'px/frame', 'bottom': 'Time (s)'}, x=plt_x)
        self.mot_dy_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE / 2, maxy=PLOT_RANGE / 2, title='dy', legend=['dy'], labels={'left': 'px/frame', 'bottom': 'Time (s)'}, x=plt_x)
        # self.mot_dz_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE / 2, maxy=PLOT_RANGE / 2, title='dz', legend=['dz'], labels={'left': 'px', 'bottom': 'Time (s)'}, x=plt_x)

        self.mot_x_output_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE / 4, maxy=PLOT_RANGE / 4, title='x output', legend=['x'], labels={'left': 'V', 'bottom': 'Time (s)'}, x=plt_x)
        self.mot_y_output_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE / 4, maxy=PLOT_RANGE / 4, title='y output', legend=['y'], labels={'left': 'V', 'bottom': 'Time (s)'}, x=plt_x)
        # self.mot_z_output_plot = RunningPlotWidget(window_length=WIN_LEN, miny=-PLOT_RANGE / 4, maxy=PLOT_RANGE / 4, title='z output', legend=['z'], labels={'left': 'V', 'bottom': 'Time (s)'}, x=plt_x)

        self.mainlayout = QtWidgets.QGridLayout()

        self.mainlayout.addWidget(self.r_view, 0, 1, 2, 1)
        self.mainlayout.addWidget(self.tn_view, 0, 2, 2, 1)

        # self.mainlayout.addWidget(self.bg_plot, 0, 3, 2, 1)

        self.mainlayout.addWidget(self.mot_x_plot, 2, 1, 1, 1)
        self.mainlayout.addWidget(self.mot_y_plot, 2, 2, 1, 1)
        # self.mainlayout.addWidget(self.mot_z_plot, 2, 3, 1, 1)

        self.mainlayout.addWidget(self.mot_dx_plot, 3, 1, 1, 1)
        self.mainlayout.addWidget(self.mot_dy_plot, 3, 2, 1, 1)
        # self.mainlayout.addWidget(self.mot_dz_plot, 3, 3, 1, 1)

        self.mainlayout.addWidget(self.mot_x_output_plot, 4, 1, 1, 1)
        self.mainlayout.addWidget(self.mot_y_output_plot, 4, 2, 1, 1)
        # self.mainlayout.addWidget(self.mot_z_output_plot, 4, 3, 1, 1)

        # Controls ----

        self._control_panel = QtWidgets.QWidget()
        self._control_layout = QtWidgets.QFormLayout()

        self._ref_button = QtWidgets.QPushButton()
        self._ref_button.setText("Update reference")
        self._control_layout.addRow(self._ref_button)
        self._ref_button.pressed.connect(self._controller.update_motion_reference)

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

        self._control_layout.addRow(QtWidgets.QLabel("<h3>Record</h3>"))  # -----------------------------------------

        self._n_stim_spin = QtWidgets.QSpinBox()
        self._n_stim_spin.setRange(1, 100)
        self._n_stim_spin.setValue(5)
        self._control_layout.addRow(QtWidgets.QLabel("Number of stimuli"), self._n_stim_spin)

        self._rec_sec_spin = QtWidgets.QSpinBox()
        self._rec_sec_spin.setRange(1, 600)
        self._rec_sec_spin.setValue(15)
        self._control_layout.addRow(QtWidgets.QLabel("Seconds per stimuli"), self._rec_sec_spin)

        self._fname_edit = QtWidgets.QLineEdit()
        self._fname_edit.setText("D:/phantom_phase_corr_1_27_22/output")
        self._fname_edit.setMinimumWidth(200)
        self._control_layout.addRow(QtWidgets.QLabel("Recording name"), self._fname_edit)

        self._rec_button = QtWidgets.QPushButton()
        self._rec_button.setText("Run experiment")
        self._control_layout.addRow(self._rec_button)
        self._rec_button.pressed.connect(self._start_recording)

        self._control_layout.addRow(QtWidgets.QLabel("<h3>OCT Scan</h3>"))  # -----------------------------------------

        self._adist_spin = QtWidgets.QDoubleSpinBox()
        self._adist_spin.setDecimals(6)
        self._adist_spin.setRange(0.00001, 0.1)
        self._adist_spin.setValue(0.001)
        self._adist_spin.setSingleStep(0.0001)
        self._control_layout.addRow(QtWidgets.QLabel("A-line spacing (mm)"), self._adist_spin)
        self._adist_spin.valueChanged.connect(self.set_scan_pattern)

        self._bidirectional_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Bidirectional raster scan"), self._bidirectional_check)
        self._bidirectional_check.toggled.connect(self.set_scan_pattern)

        self._flyback_duty_spin = QtWidgets.QDoubleSpinBox()
        self._flyback_duty_spin.setRange(0.01, 0.99)
        self._flyback_duty_spin.setValue(0.2)
        self._flyback_duty_spin.setSingleStep(0.01)
        self._control_layout.addRow(QtWidgets.QLabel("Flyback duty"), self._flyback_duty_spin)
        self._flyback_duty_spin.valueChanged.connect(self.set_scan_pattern)

        self._exposure_percentage_spin = QtWidgets.QDoubleSpinBox()
        self._exposure_percentage_spin.setRange(0.01, 0.99)
        self._exposure_percentage_spin.setValue(0.70)
        self._exposure_percentage_spin.setSingleStep(0.01)
        self._control_layout.addRow(QtWidgets.QLabel("Exposure %"), self._exposure_percentage_spin)
        self._exposure_percentage_spin.valueChanged.connect(self.set_scan_pattern)

        self._line_trigger_offset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._line_trigger_offset_slider.setSingleStep(1)
        self._line_trigger_offset_slider.setRange(-100, 100)
        self._line_trigger_offset_slider.setValue(13)
        self._line_trigger_offset_slider.valueChanged.connect(self._update_scan_pattern)
        self._line_trigger_offset_label = QtWidgets.QLabel("Line trigger skew (samples)")
        self._control_layout.addRow(self._line_trigger_offset_label, self._line_trigger_offset_slider)

        self._frame_trigger_offset_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._frame_trigger_offset_slider.setSingleStep(1)
        self._frame_trigger_offset_slider.setRange(-100, 100)
        self._frame_trigger_offset_slider.setValue(0)
        self._frame_trigger_offset_slider.valueChanged.connect(self._update_scan_pattern)
        self._frame_trigger_offset_label = QtWidgets.QLabel("Frame trigger skew (samples)")
        self._control_layout.addRow(self._frame_trigger_offset_label, self._frame_trigger_offset_slider)

        self._x_offset_spin = QtWidgets.QDoubleSpinBox()
        self._x_offset_spin.setRange(-5, 5)
        self._x_offset_spin.setValue(0)
        self._x_offset_spin.setDecimals(3)
        self._x_offset_spin.setSingleStep(0.001)
        self._control_layout.addRow(QtWidgets.QLabel("X shift (V/mm)"), self._x_offset_spin)
        self._x_offset_spin.valueChanged.connect(self.set_scan_pattern)

        self._y_offset_spin = QtWidgets.QDoubleSpinBox()
        self._y_offset_spin.setRange(-5, 5)
        self._y_offset_spin.setValue(0)
        self._y_offset_spin.setDecimals(3)
        self._y_offset_spin.setSingleStep(0.001)
        self._control_layout.addRow(QtWidgets.QLabel("Y shift"), self._y_offset_spin)
        self._y_offset_spin.valueChanged.connect(self.set_scan_pattern)

        self._angle_spin = QtWidgets.QDoubleSpinBox()
        self._angle_spin.setRange(-360, 360)
        self._angle_spin.setValue(0)
        self._angle_spin.setDecimals(1)
        self._angle_spin.setSingleStep(1)
        self._control_layout.addRow(QtWidgets.QLabel("Rotation"), self._angle_spin)
        self._angle_spin.valueChanged.connect(self.set_scan_pattern)

        self._control_layout.addRow(QtWidgets.QLabel("<h3>Display</h3>"))  # ------------------------------------------

        self._show_corr_check = QtWidgets.QCheckBox()
        self._show_corr_check.setChecked(False)
        self._control_layout.addRow(QtWidgets.QLabel("Draw correlogram"), self._show_corr_check)

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
        self._control_layout.addRow(QtWidgets.QLabel("Slice to view"), self._slice_slider)

        self._mip_check = QtWidgets.QCheckBox()
        self._control_layout.addRow(QtWidgets.QLabel("Display MIP"), self._mip_check)

        self._control_layout.addRow(QtWidgets.QLabel("<h3>Motion quantification & correction</h3>"))  # ----------------

        self._dac_output_check = QtWidgets.QCheckBox()
        self._dac_output_check.setChecked(False)
        self._control_layout.addRow(QtWidgets.QLabel("Enable DAC output"), self._dac_output_check)
        self._dac_output_check.stateChanged.connect(self._update_motion_parameters)

        self._npeak_spin = QtWidgets.QSpinBox()
        self._npeak_spin.setRange(0, int((NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) / 2))
        self._npeak_spin.setValue(1)
        self._control_layout.addRow(QtWidgets.QLabel("Centroid peak N"), self._npeak_spin)
        self._npeak_spin.valueChanged.connect(self._update_motion_parameters)

        self._hpf_spin = QtWidgets.QSpinBox()
        self._hpf_spin.setRange(0, NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR)
        self._hpf_spin.setValue(int((NUMBER_OF_ALINES_PER_B / 8) * UPSAMPLE_FACTOR))
        self._control_layout.addRow(QtWidgets.QLabel("HPF width"), self._hpf_spin)
        self._hpf_spin.valueChanged.connect(self._update_motion_parameters)

        self._x_factor_spin = QtWidgets.QDoubleSpinBox()
        self._x_factor_spin.setRange(-1000, 1000)
        self._x_factor_spin.setValue(-80)
        self._x_factor_spin.setDecimals(2)
        self._x_factor_spin.setSingleStep(0.1)
        self._control_layout.addRow(QtWidgets.QLabel("X feedback scale factor"), self._x_factor_spin)
        self._x_factor_spin.valueChanged.connect(self._update_motion_parameters)

        self._y_factor_spin = QtWidgets.QDoubleSpinBox()
        self._y_factor_spin.setRange(-1000, 1000)
        self._y_factor_spin.setValue(70)
        self._y_factor_spin.setDecimals(2)
        self._y_factor_spin.setSingleStep(0.1)
        self._control_layout.addRow(QtWidgets.QLabel("Y feedback scale factor (V/mm)"), self._y_factor_spin)
        self._y_factor_spin.valueChanged.connect(self._update_motion_parameters)

        self._z_factor_spin = QtWidgets.QDoubleSpinBox()
        self._z_factor_spin.setRange(-1000, 1000)
        self._z_factor_spin.setValue(0)
        self._z_factor_spin.setDecimals(2)
        self._z_factor_spin.setSingleStep(0.1)
        self._control_layout.addRow(QtWidgets.QLabel("Z feedback scale factor (V/mm)"), self._z_factor_spin)
        self._z_factor_spin.valueChanged.connect(self._update_motion_parameters)

        self._3dwindow_check = QtWidgets.QCheckBox()
        self._3dwindow_check.setChecked(True)
        self._3dwindow_check.stateChanged.connect(self._update_motion_parameters)
        self._control_layout.addRow(QtWidgets.QLabel("Apply window prior to FFT"), self._3dwindow_check)

        self._control_layout.addRow(QtWidgets.QLabel("<h3>Kalman filter</h3>"))  # -------------------------

        self._dt_spin = QtWidgets.QDoubleSpinBox()
        self._dt_spin.setValue(1)
        self._dt_spin.setDecimals(4)
        self._dt_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF dt"), self._dt_spin)
        self._dt_spin.valueChanged.connect(self._update_motion_parameters)

        self._e_spin = QtWidgets.QDoubleSpinBox()
        self._e_spin.setValue(1)
        self._e_spin.setDecimals(4)
        self._e_spin.setSingleStep(0.0001)
        self._e_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF position decay (e)"), self._e_spin)
        self._e_spin.valueChanged.connect(self._update_motion_parameters)

        self._f_spin = QtWidgets.QDoubleSpinBox()
        self._f_spin.setValue(1)
        self._f_spin.setDecimals(4)
        self._f_spin.setSingleStep(0.0001)
        self._f_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF velocity decay (f)"), self._f_spin)
        self._f_spin.valueChanged.connect(self._update_motion_parameters)

        self._g_spin = QtWidgets.QDoubleSpinBox()
        self._g_spin.setValue(1)
        self._g_spin.setDecimals(4)
        self._g_spin.setSingleStep(0.0001)
        self._g_spin.setRange(-9999, 9999)
        self._control_layout.addRow(QtWidgets.QLabel("KF accel decay (g)"), self._g_spin)
        self._g_spin.valueChanged.connect(self._update_motion_parameters)

        self._q_spin = QtWidgets.QDoubleSpinBox()
        self._q_spin.setDecimals(2)
        self._q_spin.setRange(-9999, 9999)
        self._q_spin.setValue(1)
        self._control_layout.addRow(QtWidgets.QLabel("KF process noise (q)"), self._q_spin)
        self._q_spin.valueChanged.connect(self._update_motion_parameters)

        self._r1_spin = QtWidgets.QDoubleSpinBox()
        self._r1_spin.setDecimals(2)
        self._r1_spin.setRange(-10000000, 10000000)
        self._r1_spin.setValue(1000)
        self._control_layout.addRow(QtWidgets.QLabel("KF position meas noise (r1)"), self._r1_spin)
        self._r1_spin.valueChanged.connect(self._update_motion_parameters)

        self._r2_spin = QtWidgets.QDoubleSpinBox()
        self._r2_spin.setDecimals(2)
        self._r2_spin.setRange(-10000000, 10000000)
        self._r2_spin.setValue(10)
        self._control_layout.addRow(QtWidgets.QLabel("KF velocity meas noise (r2)"), self._r2_spin)
        self._r2_spin.valueChanged.connect(self._update_motion_parameters)

        self._control_panel.setLayout(self._control_layout)

        self.mainlayout.addWidget(self._control_panel, 0, 0, 5, 1)

        # -------

        self.mainWidget.setLayout(self.mainlayout)
        self.setCentralWidget(self.mainWidget)

        self._define_scan_pattern()
        time.sleep(ASYNC_WAIT_DEBUG)

        self._window = np.hanning(2048).astype(np.float32)  # A-line window

        self._controller.configure(self._pat.sample_rate, ALINE_SIZE, NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES,
                                   NUMBER_OF_IMAQ_BUFFERS, roi_offset=ROI_OFFSET, roi_size=ROI_SIZE)
        time.sleep(ASYNC_WAIT_DEBUG)

        self._controller.set_processing(INTPDK, self._window)
        time.sleep(ASYNC_WAIT_DEBUG)

        self._update_scan_pattern()
        time.sleep(ASYNC_WAIT_DEBUG)

        # self._cvstream = None
        self._rec_mot = []
        self._rec_img = []

        self._is_recording = False

        self._recording_n = -1
        self._recorded_n = 0

        self._t_rec_start = -1E9

        self._spectrum_buffer = np.empty(ALINE_SIZE, dtype=np.float32)
        self._grab_buffer = np.empty(ROI_SIZE * NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES, dtype=np.complex64)
        self._grab_buffer_upsampled = np.empty(ROI_SIZE * UPSAMPLE_FACTOR * NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR * NUMBER_OF_BLINES * UPSAMPLE_FACTOR, dtype=np.complex64)
        self._grab_buffer_upsampled_2d = np.empty(NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR * NUMBER_OF_BLINES * UPSAMPLE_FACTOR, dtype=np.complex64)

        self._mot_buffer = np.zeros(11, dtype=np.float64)

        self._motion_quant_enabled = False
        self._timer = QTimer()

        while not self._controller.is_ready_to_scan():
            time.sleep(0.1)

    def _start_recording(self):
        self._sec_to_record = (self._n_stim_spin.value() + 1) * self._rec_sec_spin.value() + 3
        print('Recording for', self._sec_to_record, 'seconds')
        # self._cvstream = CvStream(fps=18)
        self._rec_mot = []
        self._rec_img = []
        self._t_rec_start = time.time()
        self._recording_n = -1
        self._recorded_n = 0
        # self._cvstream.start(self._fname_edit.text())
        self._is_recording = True
        self._rec_button.setEnabled(False)
        self._fname_edit.setEnabled(False)
        self._rec_sec_spin.setEnabled(False)
        self._controller.run_motion_experiment(self._n_stim_spin.value(), self._rec_sec_spin.value())

    def _stop_recording(self):
        n_recorded = len(self._rec_mot)
        elapsed = time.time() - self._t_rec_start
        print('Recorded', n_recorded, 'frames in', str(elapsed)[0:6], 's at', str(n_recorded / elapsed)[0:6], 'hz')
        # self._cvstream.stop()
        np.save(self._fname_edit.text() + '_img', np.array(self._rec_img))
        np.save(self._fname_edit.text() + '_mot', np.array(self._rec_mot))
        plt.plot(np.array(self._rec_mot)[:, 10], '-k')
        plt.plot(np.array(self._rec_mot)[:, 9], '-g')
        plt.plot(np.array(self._rec_mot)[:, 6], '--r')
        plt.plot(np.array(self._rec_mot)[:, 7], '--b')
        plt.show()
        self._is_recording = False
        self._rec_button.setEnabled(True)
        self._fname_edit.setEnabled(True)
        self._rec_sec_spin.setEnabled(True)

    def _start_scan(self):
        self._controller.start_scan()
        self._timer.timeout.connect(self._update)
        self._timer.start(1 / (1.5 * REFRESH_RATE))

    def _stop_scan(self):
        self._timer.stop()
        self._controller.stop_scan()

    def _stop_output(self):
        self._controller.stop_motion_quant()
        self._motion_quant_enabled = False

    def _start_output(self):
        self._get_kf_params()
        self._generate_windows()
        self._get_scale_factors()
        self._controller.configure_motion_output(AO_DX, AO_DY, AO_DZ, self._scale_factors, self._dac_output_check.isChecked())
        self._controller.start_motion_quant(
            np.array([NUMBER_OF_ALINES_PER_B, NUMBER_OF_ALINES_PER_B, NUMBER_OF_ALINES_PER_B]).astype(int),
            UPSAMPLE_FACTOR, self._npeak_spin.value(), self._spectral_window3d, self._spatial_window3d,
            self._e, self._f,  self._g, self._q, self._r1, self._r2, self._dt, N_LAG)
        self._motion_quant_enabled = True

    def _update_motion_parameters(self):
        if self._motion_quant_enabled:
            self._generate_windows()
            self._get_kf_params()
            self._get_scale_factors()
            self._controller.configure_motion_output(AO_DX, AO_DY, AO_DZ, self._scale_factors,
                                                     self._dac_output_check.isChecked())
            self._controller.update_motion_parameters(int(self._npeak_spin.value()), self._spectral_window3d,
                                                      self._spatial_window3d, self._e, self._f, self._g, self._q,
                                                      self._r1, self._r2, self._dt)

    def _update(self):
        if self._is_recording:
            if time.time() - self._t_rec_start > self._sec_to_record:
                self._stop_recording()
        got = self._controller.grab_frame(self._grab_buffer)
        try:
            if got > -1 and not np.isnan(np.sum(self._grab_buffer)):
                self._display_buffer = np.abs(reshape_unidirectional_frame(self._grab_buffer, ROI_SIZE, NUMBER_OF_ALINES_PER_B, NUMBER_OF_BLINES))
                if self._motion_quant_enabled:

                    # -- MIP3 --------------------------------------------------------------------------

                    # self._controller.grab_motion_correlogram(self._grab_buffer_upsampled_2d)
                    # # self._controller.grab_motion_frame(self._grab_buffer_upsampled_2d)
                    # self._display_buffer_upsampled = np.abs(np.reshape(self._grab_buffer_upsampled_2d, [NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR, NUMBER_OF_BLINES * UPSAMPLE_FACTOR]))
                    # self.r_view.setImage(np.fft.ifftshift(self._display_buffer_upsampled))
                    # # self.r_view.setImage(np.fft.ifft2(self._display_buffer_upsampled)))

                    # Copying the 3D correlogram out of the working buffer is VERY slow
                    if self._show_corr_check.isChecked():
                        self._controller.grab_motion_correlogram(self._grab_buffer_upsampled)
                        self._display_buffer_upsampled = np.abs(reshape_unidirectional_frame(self._grab_buffer_upsampled, ROI_SIZE * UPSAMPLE_FACTOR, NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR, NUMBER_OF_BLINES * UPSAMPLE_FACTOR))

                        self.r_view.setLevels([np.min(self._display_buffer_upsampled), np.max(self._display_buffer_upsampled)])
                        if self._xsection_radio.isChecked():
                            if self._mip_check.isChecked():
                                self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=1)
                            else:
                                self._display_buffer_upsampled = self._display_buffer_upsampled[:, self._slice_slider.value(), :]
                        elif self._ysection_radio.isChecked():
                            if self._mip_check.isChecked():
                                self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=2)
                            else:
                                self._display_buffer_upsampled = self._display_buffer_upsampled[:, :, self._slice_slider.value()]
                        else:
                            if self._mip_check.isChecked():
                                self._display_buffer_upsampled = np.max(self._display_buffer_upsampled, axis=0)
                            else:
                                self._display_buffer_upsampled = self._display_buffer_upsampled[self._slice_slider.value(), :, :]

                        self.r_view.setImage(np.fft.fftshift(self._display_buffer_upsampled))

                    # Grab motion vector
                    if not self._controller.grab_motion_vector(self._mot_buffer):  # Returns 0 if dequeue is successful
                        if self._is_recording:
                            self._rec_img.append(np.copy(self._display_buffer))
                            self._rec_mot.append(np.copy(self._mot_buffer))
                            self._recorded_n += 1
                        else:
                            self.mot_x_plot.append_to_plot([self._mot_buffer[0]])
                            self.mot_y_plot.append_to_plot([self._mot_buffer[1]])
                            # self.mot_z_plot.append_to_plot([self._mot_buffer[2]])
                            self.mot_dx_plot.append_to_plot([self._mot_buffer[3]])
                            self.mot_dy_plot.append_to_plot([self._mot_buffer[4]])
                            # self.mot_dz_plot.append_to_plot([self._mot_buffer[5]])
                        self.mot_x_output_plot.append_to_plot([self._mot_buffer[6]])
                        self.mot_y_output_plot.append_to_plot([self._mot_buffer[7]])
                        # self.mot_z_output_plot.append_to_plot([self._mot_buffer[8]])

                if not self._is_recording:  # Only draw view if not recording
                    # self._display_buffer = np.roll(self._display_buffer, -2, axis=1)
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
                    else:  # Axial
                        if self._mip_check.isChecked():
                            self._display_buffer = np.max(self._display_buffer, axis=0)
                        else:
                            self._display_buffer = self._display_buffer[np.floor_divide(self._slice_slider.value(), UPSAMPLE_FACTOR), :, :]

                    self.tn_view.setImage(self._display_buffer)

                # self._controller.grab_spectrum(self._spectrum_buffer)
                # self.bg_plot.set_spectrum_data(self._spectrum_buffer)

            else:
                # print('Frame contains NaN')
                pass
        except RuntimeWarning:
            # print('Invalid data')
            pass

    def _generate_windows(self):
        pad = int(((NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR) - NUMBER_OF_ALINES_PER_B) / 2)
        self._spatial_window3d = blackman_cube(NUMBER_OF_ALINES_PER_B, pad=pad)

        self._spatial_window3d = self._spatial_window3d.astype(np.float32).flatten()
        if not self._3dwindow_check.isChecked():
            self._spatial_window3d = np.ones(len(self._spatial_window3d)).astype(np.float32).flatten()

        self._spectral_window3d = np.ones([NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR,
                                           NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR,
                                           NUMBER_OF_ALINES_PER_B * UPSAMPLE_FACTOR])

        self._spectral_window3d[0:self._hpf_spin.value(), 0:self._hpf_spin.value(), 0:self._hpf_spin.value()] = 0
        self._spectral_window3d = np.roll(self._spectral_window3d, int(-self._hpf_spin.value() / 2), axis=0)
        self._spectral_window3d = np.roll(self._spectral_window3d, int(-self._hpf_spin.value() / 2), axis=1)
        self._spectral_window3d = np.roll(self._spectral_window3d, int(-self._hpf_spin.value() / 2), axis=2)

        # plt.subplot(1,2,1)
        # plt.imshow(np.min(self._spectral_window3d, axis=0))

        plt.show()

        self._spectral_window3d = self._spectral_window3d.astype(np.float32).flatten()



    def _get_scale_factors(self):  # Amp scale factor * V/mm * mm/pixel = volts/pixel displacement
        self._scale_factors = np.array([
            1 * self._x_factor_spin.value() * self._adist_spin.value(),
            1 * self._y_factor_spin.value() * self._adist_spin.value(),
            1 * self._z_factor_spin.value() * self._adist_spin.value()
        ]).astype(np.float64)

    def _get_kf_params(self):
        self._e = np.ones(3).astype(np.float64) * self._e_spin.value()
        self._f = np.ones(3).astype(np.float64) * self._f_spin.value()
        self._g = np.ones(3).astype(np.float64) * self._g_spin.value()
        self._q = np.ones(3).astype(np.float64) * self._q_spin.value()
        self._r1 = np.ones(3).astype(np.float64) * self._r1_spin.value()
        self._r2 = np.ones(3).astype(np.float64) * self._r2_spin.value()
        self._dt = self._dt_spin.value()

    def _define_scan_pattern(self):
        fovwidth = (NUMBER_OF_ALINES_PER_B - 1) * self._adist_spin.value()
        exp = self._exposure_percentage_spin.value()
        fb = self._flyback_duty_spin.value()
        self._pat = RasterScanPattern()

        self._pat.generate(alines=NUMBER_OF_ALINES_PER_B, blines=NUMBER_OF_BLINES, max_trigger_rate=75900,
                           fov=[fovwidth, fovwidth], samples_on=1, samples_off=2, exposure_fraction=exp, flyback_duty=fb,
                           rotation_rad=self._angle_spin.value() * (np.pi / 180), bidirectional=False)
        self._line_trigger_offset_slider.setRange(-np.floor_divide(len(self._pat.line_trigger), 8),
                                             np.floor_divide(len(self._pat.line_trigger), 8))
        print("Generated", type(self._pat), "raster scan rate", self._pat.pattern_rate, 'Hz')

    def _update_scan_pattern(self):
        self._xsig = (self._pat.x + self._x_offset_spin.value()) * V_TO_MM
        self._ysig = (self._pat.y + self._y_offset_spin.value()) * V_TO_MM
        self._ltsig = self._pat.line_trigger * TRIGGER_GAIN

        # plt.subplot(1,2,1)
        # plt.plot(self._xsig)
        # plt.plot(self._ysig)
        # plt.plot(self._ltsig)
        # plt.subplot(1,2,2)
        # plt.plot(self._pat.image_mask)
        # plt.show()

        self._controller.set_scan(self._xsig, self._ysig, self._ltsig, self._pat.sample_rate, self._pat.points_in_scan, self._pat.points_in_image, self._pat.image_mask)

    def set_scan_pattern(self):
        self._define_scan_pattern()
        self._update_scan_pattern()

    def closeEvent(self, event):
        self._timer.stop()
        self._controller.stop_scan()
        # self._controller.stop_motion_quant()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
