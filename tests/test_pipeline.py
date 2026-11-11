import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from scout.app import Pipeline, create_app
from scout.detectors import Detection


class FakeDetector:
    name = 'motion-baseline'
    precision = 'cpu'

    def infer(self, frame):
        return [Detection(2, 3, 20, 22, 0, .5)]


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(.02)
        return True, np.zeros((32, 32, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def wait_for_frame(pipe, timeout=4):
    deadline = time.monotonic() + timeout
    sequence = -1
    while time.monotonic() < deadline:
        sequence, frame = pipe.wait_frame(sequence)
        if frame:
            return sequence, frame
    raise AssertionError('no frame received')


class PipelineTest(unittest.TestCase):
    @patch('scout.app.create_detector', return_value=FakeDetector())
    @patch('scout.app.cv2.VideoCapture')
    def test_shared_frame_and_truthful_motion_label(self, capture, detector):
        source = FakeCapture()
        capture.return_value = source
        pipe = Pipeline('rtsp://camera/test', mode='motion')
        pipe.start()
        try:
            sequence, frame = wait_for_frame(pipe)
            self.assertTrue(frame.startswith(b'\xff\xd8'))
            stats = pipe.snapshot()
            self.assertEqual(stats['status'], 'running')
            self.assertEqual(stats['detections'][0]['label'], 'motion')
            self.assertIsNone(stats['detections'][0]['confidence'])
            self.assertIsNone(stats['display_fps'])
            self.assertEqual(stats['telemetry']['status'], 'not_configured')
            self.assertEqual(capture.call_count, 1)
            self.assertGreater(sequence, 0)
        finally:
            pipe.close()
        self.assertTrue(source.released)

    @patch('scout.app.create_detector')
    @patch('scout.app.cv2.VideoCapture')
    def test_encoding_does_not_block_capture(self, capture, make_detector):
        class CountingDetector(FakeDetector):
            def __init__(self):
                self.count = 0
            def infer(self, frame):
                self.count += 1
                return super().infer(frame)
        detector = CountingDetector()
        make_detector.return_value = detector
        capture.return_value = FakeCapture()
        started, release = threading.Event(), threading.Event()
        import cv2
        real_encode = cv2.imencode
        def blocked_encode(*args):
            started.set()
            release.wait(2)
            return real_encode(*args)
        with patch('scout.app.cv2.imencode', side_effect=blocked_encode):
            pipe = Pipeline('rtsp://camera/test', mode='motion')
            pipe.start()
            try:
                self.assertTrue(started.wait(1))
                deadline = time.monotonic() + 1
                while detector.count < 3 and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertGreaterEqual(detector.count, 3)
            finally:
                release.set()
                pipe.close()

    @patch('scout.app.create_detector', return_value=FakeDetector())
    @patch('scout.app.cv2.VideoCapture')
    def test_disconnect_discards_inflight_encoded_frame(self, capture, make_detector):
        class OneFrame(FakeCapture):
            def __init__(self):
                super().__init__()
                self.count = 0
            def read(self):
                time.sleep(.02)
                self.count += 1
                return (True, np.zeros((32, 32, 3), dtype=np.uint8)) if self.count == 1 else (False, None)
        capture.return_value = OneFrame()
        started, release = threading.Event(), threading.Event()
        import cv2
        real_encode = cv2.imencode
        def blocked_encode(*args):
            started.set()
            release.wait(2)
            return real_encode(*args)
        with patch('scout.app.cv2.imencode', side_effect=blocked_encode):
            pipe = Pipeline('rtsp://camera/test', mode='motion')
            pipe.start()
            try:
                self.assertTrue(started.wait(1))
                deadline = time.monotonic() + 1
                while pipe.snapshot()['status'] != 'reconnecting' and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(pipe.snapshot()['status'], 'reconnecting')
                release.set()
                time.sleep(.05)
                with pipe.changed:
                    self.assertIsNone(pipe.frame)
            finally:
                release.set()
                pipe.close()

    @patch('scout.app.create_detector', return_value=FakeDetector())
    @patch.dict('os.environ', {'SCOUT_PREVIEW_FPS': '0'})
    def test_rejects_invalid_preview_rate(self, detector):
        with self.assertRaises(ValueError):
            create_app('rtsp://camera/test', detector_mode='motion')

    @patch('scout.app.create_detector', return_value=FakeDetector())
    @patch('scout.app.cv2.VideoCapture')
    def test_reconnects_after_open_failure(self, capture, detector):
        failed, working = FakeCapture(False), FakeCapture()
        capture.side_effect = [failed, working]
        pipe = Pipeline('rtsp://camera/test', mode='motion')
        pipe.start()
        try:
            _, frame = wait_for_frame(pipe, timeout=5)
            self.assertTrue(frame)
            self.assertTrue(failed.released)
            self.assertEqual(pipe.snapshot()['status'], 'running')
        finally:
            pipe.close()


if __name__ == '__main__':
    unittest.main()
