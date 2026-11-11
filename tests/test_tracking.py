import threading
import time
import unittest

import cv2
import numpy as np

from scout.detectors import Detection, letterbox_geometry
from scout.tracking import FlowKalmanTracker, TrackedDetector


class TrackerTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(19)
        self.image = np.zeros((160, 240), dtype=np.uint8)
        self.image[40:110, 50:120] = rng.randint(0, 256, (70, 70), dtype=np.uint8)
        self.box = Detection(50, 40, 120, 110, 0, .9)

    def test_native_camera_pixels_are_padded_not_resized(self):
        self.assertEqual(letterbox_geometry(1280, 720, 1280, 736), (1.0, 1280, 720, 0, 8))
        self.assertEqual(letterbox_geometry(1920, 1080, 1920, 1088), (1.0, 1920, 1080, 0, 4))

    def test_tracks_detected_object_across_translated_frames(self):
        tracker = FlowKalmanTracker()
        initial = tracker.update(self.image, self.image.shape, 10.0, ([self.box], self.image, 10.0))
        self.assertEqual(len(initial), 1)
        self.assertFalse(initial[0].tracked)
        shifted = cv2.warpAffine(self.image, np.float32([[1, 0, 6], [0, 1, 3]]), (240, 160))
        tracked = tracker.update(shifted, shifted.shape, 10.05)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].track_id, initial[0].track_id)
        self.assertTrue(tracked[0].tracked)
        self.assertGreater(tracked[0].x1, initial[0].x1 + 1)
        self.assertAlmostEqual(tracked[0].x1, 56, delta=4)
        self.assertAlmostEqual(tracked[0].y1, 43, delta=3)

    def test_no_detection_means_no_object_and_tracks_expire(self):
        tracker = FlowKalmanTracker(max_age_seconds=.5)
        self.assertEqual(tracker.update(self.image, self.image.shape, 1.0), [])
        tracker.update(self.image, self.image.shape, 1.1, ([self.box], self.image, 1.1))
        self.assertEqual(tracker.update(self.image, self.image.shape, 1.7), [])
        self.assertEqual(tracker.update(self.image, self.image.shape, 2.0, ([self.box], self.image, 1.0)), [])

    def test_delayed_detection_without_flow_is_not_reintroduced(self):
        tracker = FlowKalmanTracker(max_age_seconds=1.0)
        blank = np.zeros_like(self.image)
        self.assertEqual(tracker.update(blank, blank.shape, 3.0, ([self.box], blank, 2.7)), [])

    def test_reconfirmation_keeps_id_and_clears_predicted_flag(self):
        tracker = FlowKalmanTracker()
        first = tracker.update(self.image, self.image.shape, 2.0, ([self.box], self.image, 2.0))
        tracker.update(self.image, self.image.shape, 2.05)
        second = tracker.update(self.image, self.image.shape, 2.1, ([self.box], self.image, 2.1))
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertFalse(second[0].tracked)

    def test_reset_discards_inflight_detection(self):
        started, release = threading.Event(), threading.Event()
        class SlowDetector:
            name = 'fake-yolo'
            precision = 'cpu'
            def infer(self, frame):
                started.set()
                release.wait(2)
                return [Detection(50, 40, 120, 110, 0, .9)]
        detector = TrackedDetector(SlowDetector(), interval=1)
        frame = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        try:
            detector.infer(frame)
            self.assertTrue(started.wait(1))
            detector.reset()
            release.set()
            deadline = time.monotonic() + 2
            while detector.busy and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertFalse(detector.busy)
            self.assertEqual(detector.status()['detector_status'], 'starting')
            self.assertEqual(detector.infer(frame), [])
        finally:
            release.set()
            detector.close()

    def test_detection_runs_off_the_capture_thread(self):
        started, release = threading.Event(), threading.Event()
        class SlowDetector:
            name = 'fake-yolo'
            precision = 'cpu'
            def infer(self, frame):
                started.set()
                release.wait(2)
                return [Detection(50, 40, 120, 110, 0, .9)]
        detector = TrackedDetector(SlowDetector(), interval=1)
        frame = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        try:
            self.assertEqual(detector.infer(frame), [])
            self.assertTrue(started.wait(1))
            t0 = time.monotonic()
            detector.infer(frame)
            self.assertLess(time.monotonic() - t0, .3)
            release.set()
            deadline = time.monotonic() + 2
            found = []
            while time.monotonic() < deadline and not found:
                time.sleep(.01)
                found = detector.infer(frame)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].class_id, 0)
        finally:
            release.set()
            detector.close()


if __name__ == '__main__':
    unittest.main()
