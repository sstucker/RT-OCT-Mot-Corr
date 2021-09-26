import ctypes as c
import pathlib
import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
import cv2
import os
import time


c_bool_p = ndpointer(dtype=np.bool, ndim=1, flags='C_CONTIGUOUS')
c_int_p = ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
c_bool_p = ndpointer(dtype=np.bool, ndim=1, flags='C_CONTIGUOUS')
c_uint16_p = ndpointer(dtype=np.uint16, ndim=1, flags='C_CONTIGUOUS')
c_uint32_p = ndpointer(dtype=np.uint32, ndim=1, flags='C_CONTIGUOUS')
c_float_p = ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_double_p = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
c_complex64_p = ndpointer(dtype=np.complex64, ndim=1, flags='C_CONTIGUOUS')

ControllerHandle = c.POINTER(c.c_char)
PlanHandle = c.POINTER(c.c_char)

os.chdir(r"C:\Users\OCT\Dev\RealtimeOCT\x64\bin")
LIB_PATH = r"C:\Users\OCT\Dev\RealtimeOCT\x64\bin\RealtimeOCT.dll"

lib = c.CDLL(LIB_PATH)

lib.RTOCT_open.argtypes = (c.c_char_p, c.c_char_p, c.c_char_p, c.c_char_p, c.c_char_p)
lib.RTOCT_open.restype = ControllerHandle
lib.RTOCT_is_scanning.argtypes = [ControllerHandle]
lib.RTOCT_is_scanning.restype = c.c_bool
lib.RTOCT_is_ready_to_scan.argtypes = [ControllerHandle]
lib.RTOCT_is_ready_to_scan.restype = c.c_bool

lib.RTOCT_configure.argtypes = (ControllerHandle, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int)
lib.RTOCT_setProcessing.argtypes = (ControllerHandle, c.c_double, c_float_p)
lib.RTOCT_setScan.argtypes = (ControllerHandle, c_double_p, c_double_p, c_double_p, c_double_p, c.c_int)
lib.RTOCT_startScan.argtypes = [ControllerHandle]
lib.RTOCT_stopScan.argtypes = [ControllerHandle]
lib.RTOCT_startSave.argtypes = (ControllerHandle, c.c_char_p, c.c_int)
lib.RTOCT_saveN.argtypes = (ControllerHandle, c.c_char_p, c.c_int, c.c_int)
lib.RTOCT_stopSave.argtypes = [ControllerHandle]
lib.RTOCT_grabFrame.argtypes = [ControllerHandle, c_complex64_p]
lib.RTOCT_grabFrame.restype = c.c_int
lib.RTOCT_grabSpectrum.argtypes = [ControllerHandle, c_float_p]
lib.RTOCT_close.argtypes = [ControllerHandle]

# -- MOTION QUANT -------------------------------------

lib.RTOCT_start_motion_output.argtypes = [ControllerHandle, c_int_p, c_double_p, c.c_int, c.c_int, c_float_p, c_float_p, c.c_bool, c_double_p, c_double_p, c_double_p, c_double_p]
lib.RTOCT_stop_motion_output.argtypes = [ControllerHandle]
lib.RTOCT_update_motion_reference.argtypes = [ControllerHandle]
lib.RTOCT_grab_motion_correlogram.argtypes = [ControllerHandle, c_complex64_p]
lib.RTOCT_grab_motion_frame.argtypes = [ControllerHandle, c_complex64_p]
lib.RTOCT_update_motion_parameters.argtypes = [ControllerHandle, c_double_p, c.c_int, c_float_p, c_float_p, c.c_bool, c_double_p, c_double_p, c_double_p, c_double_p]

lib.RTOCT_grab_motion_vector.restype = c.c_int
lib.RTOCT_grab_motion_vector.argtypes = [ControllerHandle, c_double_p]

lib.PCPLAN3D_create.restype = PlanHandle

lib.PCPLAN3D_close.argtypes = [PlanHandle]

lib.PCPLAN3D_set_reference.argtypes = (PlanHandle, c_complex64_p)
lib.PCPLAN3D_get_displacement.argtypes = (PlanHandle, c_complex64_p, c_double_p)

lib.PCPLAN3D_get_r.argtypes = (PlanHandle, c_complex64_p)
lib.PCPLAN3D_get_R.argtypes = (PlanHandle, c_complex64_p)
lib.PCPLAN3D_get_t0.argtypes = (PlanHandle, c_complex64_p)
lib.PCPLAN3D_get_tn.argtypes = (PlanHandle, c_complex64_p)


class RealtimeOCTController:

    def __init__(self, camera_name, ao_ch_x_name, ao_ch_y_name, ao_ch_lt_name, ao_ch_ft_name):
        """
        camera_name -- string. Name of cam file corresponding to line camera
        ao_ch_x_name -- string. NI-DAQ analog out channel identifier to be used for X galvo output
        ao_ch_y_name -- string. NI-DAQ analog out channel identifier to be used for Y galvo output
        ao_ch_lt_name -- string. NI-DAQ analog out channel identifier to be used for camera triggering
        ao_ch_ft_name -- string. NI-DAQ analog out channel identifier to be used for frame grab triggering
        """
        self._handle = lib.RTOCT_open(camera_name.encode('utf-8'), ao_ch_x_name.encode('utf-8'),
                                      ao_ch_y_name.encode('utf-8'), ao_ch_lt_name.encode('utf-8'),
                                      ao_ch_ft_name.encode('utf-8'))

    def __del__(self):
        lib.RTOCT_close(self._handle)

    def configure(self, dac_output_rate, aline_size, number_of_alines, number_of_imaq_buffers, roi_offset=0,
                  roi_size=None):
        """
        Configures attributes of OCT acquisition such as NI hardware channel identifiers and image size. Acquisition
        cannot be configured during a scan.
        dac_output_rate -- int. The rate at which the DAC generates the samples passed to set_scan. Defined by the
                                scan pattern.
        aline_size -- int. The number of voxels in each A-line i.e. 2048
        number_of_alines -- int. The number of A-lines in each acquisiton frame. Defined by the scan pattern.
        number_of_imaq_buffers -- int. The number of buffers to allocate for image acquisition and processing. Larger
                                  values make acquisiton more robust to dropped frames but increase memory overhead.
        roi_offset (optional) -- int. Number of voxels to discard from beginning of each spatial A-line
        roi_size (optional) -- int. Number of voxels to keep of each spatial A-line, beginning from roi_offset
        """
        if roi_size is None:
            roi_size = aline_size
        lib.RTOCT_configure(self._handle, int(dac_output_rate), int(aline_size), int(roi_offset), int(roi_size),
                            int(number_of_alines), int(number_of_imaq_buffers))

    def is_scanning(self):
        return lib.RTOCT_is_scanning(self._handle)

    def is_ready_to_scan(self):
        return lib.RTOCT_is_ready_to_scan(self._handle)

    def set_processing(self, intpdk, apod_window):
        """
        Configures attributes of SD-OCT processing. Can be called during a scan.
        intpdk -- scalar. Parameter for linear-in-wavelength -> linear-in-wavenmber interpolation.
        apod_window -- numpy array. Window which is multiplied by each spectral A-line prior to FFT i.e. Hanning window
        """
        lib.RTOCT_setProcessing(self._handle, np.double(intpdk), apod_window)

    def set_scan(self, x, y, lt, ft):
        """
        Sets the signals used to drive the galvos and trigger camera and frame grabber. Can be called during a scan.
        x -- numpy array. X galvo drive signal
        y -- numpy array. Y galvo drive signal
        lt -- numpy array. Camera A-line exposure trigger signal
        ft -- numpy array. Frame grabber trigger signal
        """
        lib.RTOCT_setScan(self._handle, x, y, lt, ft, len(x))

    def start_scan(self):
        lib.RTOCT_startScan(self._handle)

    def stop_scan(self):
        lib.RTOCT_stopScan(self._handle)

    def start_save(self, file_name, max_bytes):
        """
        Begins streaming a TIFF file to disk until scanning stops or stop_save is called.
        file_name -- string. Desired name of output file
        max_bytes -- int. Maximum size each file can be before a new one is created
        """
        lib.RTOCT_startSave(self._handle, file_name.split('.')[0].encode('utf-8'), int(max_bytes))

    def save_n(self, file_name, max_bytes, frames_to_save):
        """
        Streams frames_to_save frames to disk.
        file_name -- string. Desired name of output file
        max_bytes -- int. Maximum size each file can be before a new one is created
        frames_to_save -- int. Number of frames to save
        """
        lib.RTOCT_saveN(self._handle, file_name.split('.')[0].encode('utf-8'), int(max_bytes), int(frames_to_save))

    def stop_save(self):
        lib.RTOCT_stopSave(self._handle)

    def grab_frame(self, output):
        return lib.RTOCT_grabFrame(self._handle, output)

    def grab_spectrum(self, output):
        return lib.RTOCT_grabSpectrum(self._handle, output)

    # -- MOTION TRACKING ----------------------------------------------------------------------------------------

    def start_motion_output(self, input_dims, scale_xyz, upsampling_factor, npeak, spectral_window3d, spatial_window3d, filter_d, filter_g, filter_q, filter_r, bidirectional=False):
        lib.RTOCT_start_motion_output(self._handle, input_dims, scale_xyz, upsampling_factor, npeak, spectral_window3d, spatial_window3d, bidirectional, filter_d, filter_g, filter_q, filter_r)

    def stop_motion_output(self):
        lib.RTOCT_stop_motion_output(self._handle)

    def update_motion_reference(self):
        lib.RTOCT_update_motion_reference(self._handle)

    def update_motion_parameters(self, scale_xyz, npeak, spectral_window3d, spatial_window3d, filter_d, filter_g, filter_q, filter_r, bidirectional=False):
        lib.RTOCT_update_motion_parameters(self._handle, scale_xyz, npeak, spectral_window3d, spatial_window3d, bidirectional, filter_d, filter_g, filter_q, filter_r)

    def grab_motion_correlogram(self, out):
        lib.RTOCT_grab_motion_correlogram(self._handle, out)

    def grab_motion_frame(self, out):
        lib.RTOCT_grab_motion_frame(self._handle, out)

    def grab_motion_vector(self, out):
        return lib.RTOCT_grab_motion_vector(self._handle, out)
