
import numpy as np
import matplotlib.pyplot as plt
import time
# from RealtimeFlowOCT.PyScanPattern.Patterns import BlineRepeatedRasterScan, RasterScanPattern, RoseScanPattern, \
#     BidirectionalRasterScanPattern

from scanpatterns import RasterScanPattern

from RealtimeOCT import RealtimeOCTController

import sys
import os

# input("Press ENTER to continue.")

BYTES_PER_GB = 1073741824

CAM = 'img1'
AO_X = 'Dev1/ao1'
AO_Y = 'Dev1/ao2'
AO_LT = 'Dev1/ao0'
AO_FT = 'Dev1/ao3'

ALINE_SIZE = 2048
PI = np.pi
TRIGGER_GAIN = 4
NUMBER_OF_IMAQ_BUFFERS = 128
INTPDK = 0.305

ROI_OFFSET = 10

ALINE_REPEAT = 4
BLINE_REPEAT = 2
NUMBER_OF_ALINES_PER_B = 256
NUMBER_OF_BLINES = 256

ROI_SIZE = 200

CHUNKS = 256

N_FRAMES_TO_ACQUIRE = 2

FOVWIDTH = 0.5
ROT_DEG = 6
MM_TO_V = 14.06
SHIFT_X = -20 * 10**-3  # mm
SHIFT_Y = -60 * 10**-3  # mm

# FILENAME = r'D:\angio_2_28_22\angio_jg09_01'
FILENAME = r'D:\new_pc_test'

total_number_of_alines_per_frame = (NUMBER_OF_ALINES_PER_B * ALINE_REPEAT * BLINE_REPEAT * NUMBER_OF_BLINES) / CHUNKS

size_of_each_raw_frame_gb = (NUMBER_OF_ALINES_PER_B * ALINE_REPEAT * BLINE_REPEAT * NUMBER_OF_BLINES * ALINE_SIZE * 2) / BYTES_PER_GB  # uint16
size_of_each_processed_frame_bytes = NUMBER_OF_ALINES_PER_B * ALINE_REPEAT * BLINE_REPEAT * NUMBER_OF_BLINES * ROI_SIZE * 8  # complex64
size_of_each_processed_frame_gb = size_of_each_processed_frame_bytes / BYTES_PER_GB  # complex64

print('Each raw frame is', size_of_each_raw_frame_gb, 'GB in size (' + str(NUMBER_OF_ALINES_PER_B * ALINE_REPEAT * BLINE_REPEAT * NUMBER_OF_BLINES) + ' A-lines)')
print('Each processed frame is', size_of_each_processed_frame_gb, 'GB in size (after uint16 -> float64 conversion and', ALINE_SIZE, '->', ROI_SIZE, 'axial crop)')

print('With', CHUNKS, 'chunks, each (chunked) raw frame is', (total_number_of_alines_per_frame * ALINE_SIZE * 2) / BYTES_PER_GB,
      'GB in size (' + str(total_number_of_alines_per_frame) + ' A-lines per chunk)')

print('With', N_FRAMES_TO_ACQUIRE, 'frames to acquire, total processed acquisition size is', size_of_each_processed_frame_gb * N_FRAMES_TO_ACQUIRE, 'GB')

pattern = RasterScanPattern()

pattern.generate(max_trigger_rate=76000, alines=NUMBER_OF_ALINES_PER_B, blines=NUMBER_OF_BLINES,
                 bline_repeat=BLINE_REPEAT, aline_repeat=ALINE_REPEAT, fov=[FOVWIDTH, FOVWIDTH], samples_on=1, samples_off=2,
                 exposure_fraction=0.4, rotation_rad=ROT_DEG*(np.pi / 180), trigger_blines=CHUNKS is not None)

print('Pattern rate of', str(pattern.pattern_rate)[0:5], 'acquisition time of', (1 / pattern.pattern_rate) * N_FRAMES_TO_ACQUIRE)

# plt.subplot(1, 2, 1)
# plt.plot(pattern.x)
# plt.plot(pattern.y)
# plt.plot(pattern.line_trigger)
#
# plt.subplot(1, 2, 2)
# plt.plot(pattern.x, pattern.y)
# plt.scatter(pattern.x[pattern.line_trigger.astype(bool)], pattern.y[pattern.line_trigger.astype(bool)])
# plt.show()

controller = RealtimeOCTController(CAM, AO_X, AO_Y, AO_LT, AO_FT)
controller.configure(pattern.sample_rate, ALINE_SIZE, total_number_of_alines_per_frame, NUMBER_OF_IMAQ_BUFFERS,
                     roi_offset=ROI_OFFSET, roi_size=ROI_SIZE)
apod_window = np.hanning(ALINE_SIZE).astype(np.float32)
controller.set_processing(INTPDK, apod_window)

xsig = (pattern.x + SHIFT_X) * MM_TO_V
ysig = (pattern.y + SHIFT_Y) * MM_TO_V
ltsig = pattern.line_trigger * TRIGGER_GAIN

ftsig = pattern.frame_trigger * TRIGGER_GAIN

controller.set_scan(xsig, ysig, ltsig, ftsig)

ready_to_scan = False
while not ready_to_scan:
    print("Waiting for setup to complete...")
    time.sleep(1)
    ready_to_scan = controller.is_ready_to_scan()

controller.start_scan()

scanning = False
while not scanning:
    print("Waiting for scanning to start...")
    time.sleep(1)
    scanning = controller.is_scanning()
print('Scan started!')

if os.path.exists(FILENAME + '.RAW'):
    print('Deleting', FILENAME + '.RAW', 'because it already exists.')
    os.remove(FILENAME + '.RAW')

time.sleep(2)  # Galvo settling time

print('Attempting to acquire files with size', size_of_each_processed_frame_bytes, 'bytes,',
      size_of_each_processed_frame_bytes / BYTES_PER_GB, 'GB')
print('Attemping to acquire', N_FRAMES_TO_ACQUIRE * CHUNKS, 'total chunked frames,', N_FRAMES_TO_ACQUIRE, 'frames')
controller.save_n(FILENAME, size_of_each_processed_frame_bytes, N_FRAMES_TO_ACQUIRE * CHUNKS)
# controller.start_save(FILENAME, 2E9)

time.sleep(int((1 / pattern.pattern_rate) * (N_FRAMES_TO_ACQUIRE + 2)))

controller.stop_save()

controller.stop_scan()

scanning = True
while scanning:
    print("Waiting for scanning to stop...")
    time.sleep(1)
    scanning = controller.is_scanning()

try:
    z = np.fromfile(FILENAME + '.RAW', dtype=np.complex64)
    zlen = int(len(z) / (NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES * ROI_SIZE))
    print('Saved', zlen, 'frames')
    img = np.reshape(z, [zlen, NUMBER_OF_BLINES, NUMBER_OF_ALINES_PER_B, ROI_SIZE])
    plt.imshow(np.abs(img[zlen - 1, 0, :, :]))
    plt.show()

except FileNotFoundError:
    print("Couldn't open file")