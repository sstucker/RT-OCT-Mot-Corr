
import numpy as np
import matplotlib.pyplot as plt
import time
from RealtimeFlowOCT.PyScanPattern.Patterns import BlineRepeatedRasterScan, RasterScanPattern, RoseScanPattern, \
    BidirectionalRasterScanPattern

from RealtimeOCT import RealtimeOCTController

import sys
import os

# input("Press ENTER to continue.")

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

ROI_OFFSET = 10

BLINE_REPEAT = 2
NUMBER_OF_ALINES_PER_B = 128
NUMBER_OF_BLINES = 128
ALINE_SPACING = 0.001
ROI_SIZE = 400

fovwidth = ALINE_SPACING * (1 - NUMBER_OF_ALINES_PER_B)

# pattern = RasterScanPattern()
# pattern.generate(alines=NUMBER_OF_ALINES_PER_B, blines=NUMBER_OF_BLINES, fov=[fovwidth, fovwidth],
#                  samples_on=1, samples_off=1)

pattern = BlineRepeatedRasterScan()
pattern.generate(alines=NUMBER_OF_ALINES_PER_B, blines=NUMBER_OF_BLINES, bline_repeat=BLINE_REPEAT, fov=[fovwidth, fovwidth],
                 samples_on=1, samples_off=1)
NUMBER_OF_ALINES_PER_B = NUMBER_OF_ALINES_PER_B * BLINE_REPEAT

controller = RealtimeOCTController(CAM, AO_X, AO_Y, AO_LT, AO_FT)
controller.configure(pattern.get_sample_rate(), ALINE_SIZE, NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES, NUMBER_OF_IMAQ_BUFFERS,
                     roi_offset=ROI_OFFSET, roi_size=ROI_SIZE)
apod_window = np.hanning(ALINE_SIZE).astype(np.float32)
controller.set_processing(INTPDK, apod_window)

xsig = pattern.get_x() * 22
ysig = pattern.get_y() * 18
ltsig = pattern.get_line_trig() * TRIGGER_GAIN
ftsig = pattern.get_frame_trig() * TRIGGER_GAIN
controller.set_scan(xsig, ysig, ltsig, ftsig)

# plt.plot(xsig, ysig)
# plt.scatter(xsig[ltsig.astype(bool)], ysig[ltsig.astype(bool)])
# plt.show()

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

# f = np.empty(ALINE_SIZE * NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES).astype(np.complex64)
# for i in range(1000):
#     print(controller.grab_frame(f))
#     time.sleep(1 / pattern.get_pattern_rate())


FILENAME = r'D:\realtimeoct_acq\continuous_imaging_test'
ACQ_N = 10

time.sleep(2)  # Galvo settling time

# controller.save_n(FILENAME, 2E9, 1)
controller.start_save(FILENAME, 2E9)

time.sleep(5)

controller.stop_save()

controller.stop_scan()

try:
    z = np.fromfile(FILENAME + '.RAW', dtype=np.complex64)
    zlen = int(len(z) / (NUMBER_OF_ALINES_PER_B * NUMBER_OF_BLINES * ROI_SIZE))

    print('Saved', zlen, 'frames')

    img = np.reshape(z, [zlen, NUMBER_OF_BLINES, NUMBER_OF_ALINES_PER_B, ROI_SIZE])

    plt.imshow(np.abs(img[zlen - 1, 0, :, :]))
    plt.show()
except FileNotFoundError:
    print("Couldn't open file")