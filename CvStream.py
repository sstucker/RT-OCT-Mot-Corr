import time
import cv2
import multiprocessing as mp

# TODO make access to camera interface thread safe


def _run(process_running, filename, camera_port, fps):
    capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    avi_size = (int(capture.get(3)), int(capture.get(4)))
    writer = cv2.VideoWriter(filename.split('.')[0] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, avi_size)
    started = time.time()
    frames_written = 0
    while process_running.value:
        ret, frame = capture.read()
        if ret:
            writer.write(frame)
            frames_written += 1
    ended = time.time()
    print('Took', str(ended - started), 'to record', frames_written, 'frames at',
          str(frames_written / (ended - started))[0:7], 'frames per second.')
    writer.release()
    capture.release()
    cv2.destroyAllWindows()


def _run_for_n_seconds(filename, camera_port, fps, n_seconds):
    capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    avi_size = (int(capture.get(3)), int(capture.get(4)))
    writer = cv2.VideoWriter(filename.split('.')[0] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, avi_size)
    started = time.time()
    frames_written = 0
    while int(time.time() - started) < n_seconds:
        ret, frame = capture.read()
        if ret:
            writer.write(frame)
            frames_written += 1
    ended = time.time()
    print('Took', str(ended - started), 'to record', frames_written, 'frames at',
          str(frames_written / (ended - started))[0:7], 'frames per second.')
    writer.release()
    capture.release()
    cv2.destroyAllWindows()


def _run_for_n_frames(filename, camera_port, fps, n_frames):
    capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    avi_size = (int(capture.get(3)), int(capture.get(4)))
    writer = cv2.VideoWriter(filename.split('.')[0] + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, avi_size)
    frames_written = 0
    started = time.time()
    while frames_written < n_frames:
        ret, frame = capture.read()
        if ret:
            writer.write(frame)
            frames_written += 1
    ended = time.time()
    print('Took', str(ended - started), 'to record', frames_written, 'frames at',
          str(frames_written / (ended - started))[0:7], 'frames per second.')
    writer.release()
    capture.release()
    cv2.destroyAllWindows()


class CvStream:

    def __init__(self, camera_port=0, fps=20):
        self._process = None
        self._process_running = mp.Value("i", 0)
        self._port = camera_port
        self._fps = fps

    def start(self, filename):
        self._process = mp.Process(target=_run, args=(self._process_running, filename, self._port, self._fps))
        self._process_running.value = 1
        self._process.start()

    def stop(self):
        self._process_running.value = 0
        self._process.join()

    def capture_n_seconds(self, filename, n_seconds):
        self._process = mp.Process(target=_run_for_n_seconds, args=(filename, self._port, self._fps, n_seconds))
        self._process.start()

    def capture_n_frames(self, filename, n_frames):
        self._process = mp.Process(target=_run_for_n_frames, args=(filename, self._port, self._fps, n_frames))
        self._process.start()

    def __del__(self):
        self.stop()


if __name__ == '__main__':

    stream = CvStream(fps=20)
    print('Streaming 30 seconds of video to cvstream_demo.avi...')
    # stream.capture_n_frames('cvstream_demo', 20 * 30)
    stream.capture_n_seconds('cvstream_demo', 30)

    # stream.start('cvstream_demo')
    time.sleep(30)
    # stream.stop()

    print('Finishing up...')
