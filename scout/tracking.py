"""Object detection on a worker thread, with optical flow and Kalman tracks.

Tracks are created only from model detections. They expire if they are not
reconfirmed, even if optical flow continues to follow image features.
"""
from collections import deque
from dataclasses import dataclass
import os
import threading
import time

import cv2
import numpy as np


@dataclass
class TrackedDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
    track_id: int
    tracked: bool
    detection_age_ms: float


def _box_measurement(box):
    x1, y1, x2, y2 = box
    return np.array([[(x1 + x2) / 2], [(y1 + y2) / 2],
                     [max(2, x2 - x1)], [max(2, y2 - y1)]], dtype=np.float32)


def _state_box(state, width, height):
    cx, cy, w, h = (float(state[i, 0]) for i in range(4))
    w, h = max(2, w), max(2, h)
    x1, y1 = max(0, cx - w / 2), max(0, cy - h / 2)
    x2, y2 = min(width - 1, cx + w / 2), min(height - 1, cy + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def _flow(previous, current, boxes, frame_size):
    """Estimate median displacement in each box using forward/backward LK flow."""
    if previous is None or previous.shape != current.shape or not boxes:
        return {}
    height, width = frame_size
    sy, sx = current.shape[0] / height, current.shape[1] / width
    mask = np.zeros_like(previous)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(mask, (max(0, int(x1 * sx)), max(0, int(y1 * sy))),
                      (min(current.shape[1] - 1, int(x2 * sx)),
                       min(current.shape[0] - 1, int(y2 * sy))), 255, -1)
    points = cv2.goodFeaturesToTrack(previous, maxCorners=60, qualityLevel=.01,
                                     minDistance=4, mask=mask, blockSize=3)
    if points is None or len(points) < 3:
        return {}
    options = dict(winSize=(15, 15), maxLevel=2,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if forward is None or status is None:
        return {}
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
    if backward is None or back_status is None:
        return {}
    start, end = points.reshape(-1, 2), forward.reshape(-1, 2)
    valid = (status.reshape(-1) > 0) & (back_status.reshape(-1) > 0)
    valid &= np.linalg.norm(backward.reshape(-1, 2) - start, axis=1) < 1.5
    result = {}
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        inside = ((start[:, 0] >= x1 * sx) & (start[:, 0] <= x2 * sx) &
                  (start[:, 1] >= y1 * sy) & (start[:, 1] <= y2 * sy) & valid)
        if np.count_nonzero(inside) < 3:
            continue
        displacement = np.median(end[inside] - start[inside], axis=0)
        dx, dy = float(displacement[0] / sx), float(displacement[1] / sy)
        if abs(dx) <= width / 4 and abs(dy) <= height / 4:
            result[i] = (dx, dy)
    return result


class _Track:
    def __init__(self, detection, track_id, observed_at, now):
        self.id, self.class_id, self.confidence = track_id, detection.class_id, float(detection.confidence)
        self.last_detection, self.last_update = observed_at, now
        self.box = (float(detection.x1), float(detection.y1), float(detection.x2), float(detection.y2))
        self.filter = cv2.KalmanFilter(8, 4)
        self.filter.measurementMatrix = np.hstack((np.eye(4, dtype=np.float32), np.zeros((4, 4), dtype=np.float32)))
        self.filter.transitionMatrix = np.eye(8, dtype=np.float32)
        self.filter.processNoiseCov = np.eye(8, dtype=np.float32) * .03
        self.filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 4
        self.filter.errorCovPost = np.eye(8, dtype=np.float32) * 10
        self.filter.statePost = np.vstack((_box_measurement(self.box), np.zeros((4, 1), dtype=np.float32)))
        self.observed = True

    def predict(self, now, width, height):
        dt = min(.25, max(.001, now - self.last_update))
        transition = np.eye(8, dtype=np.float32)
        for i in range(4):
            transition[i, i + 4] = dt
        self.filter.transitionMatrix = transition
        self.last_update = now
        return _state_box(self.filter.predict(), width, height)

    def correct(self, box, width, height):
        self.box = _state_box(self.filter.correct(_box_measurement(box)), width, height)


class FlowKalmanTracker:
    def __init__(self, max_age_seconds=1.0, max_tracks=30, flow_interval=2):
        if flow_interval < 1:
            raise ValueError('Flow interval must be positive')
        self.flow_interval = flow_interval
        self.frame_number = 0
        self.max_age = max_age_seconds
        self.max_tracks = max_tracks
        self.tracks = []
        self.previous_gray = None
        self.frame_size = None
        self.next_id = 1

    def update(self, gray, frame_size, now, result=None):
        height, width = frame_size
        if self.frame_size != frame_size:
            self.tracks.clear()
            self.previous_gray = None
            self.frame_size = frame_size
        boxes = [track.box for track in self.tracks]
        self.frame_number += 1
        vectors = (_flow(self.previous_gray, gray, boxes, frame_size)
                   if self.frame_number % self.flow_interval == 0 else {})
        for index, track in enumerate(self.tracks):
            track.observed = False
            previous_box = track.box
            predicted = track.predict(now, width, height)
            if predicted is None:
                track.box = None
                continue
            track.box = predicted
            if index in vectors:
                dx, dy = vectors[index]
                shifted = (previous_box[0] + dx, previous_box[1] + dy,
                           previous_box[2] + dx, previous_box[3] + dy)
                track.correct(shifted, width, height)
        self.tracks = [track for track in self.tracks if track.box is not None and now - track.last_detection <= self.max_age]
        if result is not None:
            detections, detection_gray, captured_at = result
            # Old model results should never revive tracks at stale positions.
            if 0 <= now - captured_at <= self.max_age:
                self._merge(detections, detection_gray, gray, captured_at, now, frame_size)
        self.previous_gray = gray
        return [TrackedDetection(*(int(round(value)) for value in track.box), track.class_id,
                                 track.confidence, track.id, not track.observed,
                                 round((now - track.last_detection) * 1000, 1))
                for track in self.tracks if track.box is not None]

    def _merge(self, detections, detection_gray, gray, captured_at, now, frame_size):
        height, width = frame_size
        detections = list(detections)[:self.max_tracks]
        boxes = [(float(d.x1), float(d.y1), float(d.x2), float(d.y2)) for d in detections]
        vectors = _flow(detection_gray, gray, boxes, frame_size)
        unmatched = set(range(len(self.tracks)))
        for i, detection in enumerate(detections):
            # Without optical-flow evidence, a delayed box may no longer cover its object.
            if now - captured_at > .2 and i not in vectors:
                continue
            dx, dy = vectors.get(i, (0.0, 0.0))
            original = boxes[i]
            box = (max(0, original[0] + dx), max(0, original[1] + dy),
                   min(width - 1, original[2] + dx), min(height - 1, original[3] + dy))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            candidates = [( _iou(self.tracks[j].box, box), j) for j in unmatched
                          if self.tracks[j].class_id == detection.class_id]
            score, match = max(candidates, default=(0, None))
            if score >= .25:
                track = self.tracks[match]
                unmatched.remove(match)
                track.correct(box, width, height)
                track.confidence = float(detection.confidence)
                track.last_detection = captured_at
                track.observed = True
            elif len(self.tracks) < self.max_tracks:
                aligned = TrackedDetection(*(int(round(v)) for v in box), detection.class_id,
                                           float(detection.confidence), 0, False, 0)
                self.tracks.append(_Track(aligned, self.next_id, captured_at, now))
                self.next_id += 1


class TrackedDetector:
    """Run detection asynchronously and update tracks on each camera frame."""
    def __init__(self, detector, interval=5, max_age_seconds=1.0, flow_interval=2):
        if interval < 1 or max_age_seconds <= 0 or flow_interval < 1:
            raise ValueError('Detection interval and tracking max age must be positive')
        self.detector = detector
        self.name = detector.name + ' + flow/Kalman'
        self.precision = detector.precision
        self.input_resolution = getattr(detector, 'input_resolution', None)
        self.flow_resolution = None
        if hasattr(detector, 'class_names'):
            self.class_names = detector.class_names
        self.interval = interval
        self.max_age_seconds = max_age_seconds
        self.flow_interval = flow_interval
        self.manager = FlowKalmanTracker(max_age_seconds, flow_interval=flow_interval)
        self.generation = 0
        self.frame_number = 0
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.busy = False
        self.error = None
        self.stopping = False
        self.completed = deque(maxlen=30)
        self.last_detection_ms = None
        self.last_detection_time = None
        self.worker = threading.Thread(target=self._run, name='scout-detector', daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.stopping)
                if self.stopping:
                    return
                frame, gray, captured_at, generation = self.pending
                self.pending = None
                self.busy = True
            start = time.monotonic()
            try:
                result = self.detector.infer(frame)
                detections = result[0] if isinstance(result, tuple) else result
            except Exception as exc:
                with self.condition:
                    self.error = str(exc)
                    self.busy = False
                return
            completed_at = time.monotonic()
            with self.condition:
                if generation == self.generation:
                    self.result = (detections, gray, captured_at)
                    self.last_detection_ms = round((completed_at - start) * 1000, 1)
                    self.last_detection_time = captured_at
                    self.completed.append(completed_at)
                self.busy = False

    def infer(self, frame):
        now = time.monotonic()
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.flow_resolution = f'{gray.shape[1]}x{gray.shape[0]}'
        with self.condition:
            if self.error:
                raise RuntimeError(f'Object detector failed: {self.error}')
            result, self.result = self.result, None
            if self.frame_number % self.interval == 0 and not self.busy and self.pending is None:
                self.pending = (frame.copy(), gray, now, self.generation)
                self.condition.notify()
        self.frame_number += 1
        return self.manager.update(gray, (h, w), now, result)

    def status(self):
        with self.condition:
            frequency = ((len(self.completed) - 1) / (self.completed[-1] - self.completed[0])
                         if len(self.completed) > 1 and self.completed[-1] > self.completed[0] else 0.0)
            return {'tracking': 'Optical flow + Kalman',
                    'flow_resolution': self.flow_resolution,
                    'detector_fps': round(frequency, 1),
                    'detector_latency_ms': self.last_detection_ms,
                    'detection_age_ms': (round((time.monotonic() - self.last_detection_time) * 1000, 1)
                                         if self.last_detection_time is not None else None),
                    'detector_status': 'error' if self.error else ('running' if self.completed else 'starting')}

    def reset(self):
        """Invalidate tracks and in-flight results after a camera disconnect."""
        with self.condition:
            self.generation += 1
            self.pending = None
            self.result = None
            self.completed.clear()
            self.last_detection_ms = None
            self.last_detection_time = None
            self.frame_number = 0
        self.manager = FlowKalmanTracker(self.max_age_seconds, flow_interval=self.flow_interval)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.worker.join(timeout=5)
        if not self.worker.is_alive():
            close = getattr(self.detector, 'close', None)
            if close:
                close()


def create_tracked_detector(detector):
    return TrackedDetector(detector, int(os.getenv('SCOUT_DETECT_INTERVAL', '5')),
                           float(os.getenv('SCOUT_TRACK_MAX_AGE', '1.0')),
                           int(os.getenv('SCOUT_FLOW_INTERVAL', '2')))
