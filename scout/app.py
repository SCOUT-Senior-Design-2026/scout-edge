"""SCOUT video capture, detection and browser dashboard."""
import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from .detectors import COCO_NAMES, create_detector

ROOT = Path(__file__).parent
LYNX_SOURCE = 'rtsp://192.168.144.240:8554/payload'


class Pipeline:
    """One capture and detector shared by every browser viewer."""

    def __init__(self, source, mode='motion', engine='models/yolo11s_fp16.engine'):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.detector = create_detector(mode, engine)
        self.mode = mode
        self.stop_event = threading.Event()
        self.changed = threading.Condition()
        self.thread = None
        self.frame = None
        self.sequence = 0
        self.stats = {
            'status': 'starting', 'error': None, 'detector': self.detector.name,
            'precision': self.detector.precision, 'input_resolution': None,
            'capture_fps': 0.0, 'inference_fps': 0.0, 'display_fps': None,
            'latency_ms': None, 'inference_latency_ms': None, 'detections': [],
            'source_name': 'Gremsy Lynx' if self.source == LYNX_SOURCE else 'Configured source',
            'telemetry': {'status': 'not_configured'},
        }

    def start(self):
        self.thread = threading.Thread(target=self.run, name='scout-capture', daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread:
            self.thread.join(timeout=6)
        if not self.thread or not self.thread.is_alive():
            close = getattr(self.detector, 'close', None)
            if close:
                close()

    def snapshot(self):
        with self.changed:
            return dict(self.stats)

    def wait_frame(self, sequence):
        with self.changed:
            self.changed.wait_for(lambda: self.sequence != sequence or self.stop_event.is_set(), timeout=2)
            return self.sequence, self.frame

    def update(self, **values):
        with self.changed:
            self.stats.update(values)

    def run(self):
        # The RTSP source accepts interleaved TCP, including over SSH tunnels.
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp|stimeout;5000000')
        while not self.stop_event.is_set():
            cap = None
            try:
                self.update(status='connecting', error=None)
                backend = cv2.CAP_FFMPEG if isinstance(self.source, str) else cv2.CAP_ANY
                cap = cv2.VideoCapture(self.source, backend)
                if not cap.isOpened():
                    raise RuntimeError('Unable to open the video source')
                count, interval = 0, time.monotonic()
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        is_file = isinstance(self.source, str) and not self.source.startswith(('rtsp://', 'http://', 'https://', 'tcp://'))
                        if is_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError('Video source stopped delivering frames')
                    started = time.monotonic()
                    infer_started = time.monotonic()
                    result = self.detector.infer(frame)
                    detections = result[0] if isinstance(result, tuple) else result
                    infer_ms = (time.monotonic() - infer_started) * 1000
                    labels = getattr(self.detector, 'class_names', COCO_NAMES)
                    summary = []
                    for detection in detections[:50]:
                        label = 'motion' if self.mode == 'motion' else (labels[detection.class_id] if 0 <= detection.class_id < len(labels) else str(detection.class_id))
                        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), (79, 224, 117), 2)
                        text = label if self.mode == 'motion' else f'{label} {detection.confidence:.0%}'
                        cv2.putText(frame, text, (detection.x1, max(22, detection.y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .65, (79, 224, 117), 2)
                        summary.append({'label': label, 'confidence': None if self.mode == 'motion' else round(float(detection.confidence), 3)})
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    count += 1
                    now = time.monotonic()
                    with self.changed:
                        if now - interval >= 1:
                            rate = round(count / (now - interval), 1)
                            self.stats.update(capture_fps=rate, inference_fps=rate)
                            count, interval = 0, now
                        self.stats.update(status='running', error=None,
                            input_resolution=f'{frame.shape[1]}x{frame.shape[0]}',
                            latency_ms=round((now - started) * 1000, 1),
                            inference_latency_ms=round(infer_ms, 1), detections=summary[:20])
                        self.frame = encoded.tobytes()
                        self.sequence += 1
                        self.changed.notify_all()
            except Exception as exc:
                self.update(status='reconnecting', error=str(exc), capture_fps=0.0, inference_fps=0.0, detections=[])
                self.stop_event.wait(2)
            finally:
                if cap is not None:
                    cap.release()
        self.update(status='stopped')


def create_app(source=LYNX_SOURCE, detector_mode=None, engine_path=None):
    pipe = Pipeline(source, detector_mode or os.getenv('SCOUT_DETECTOR', 'motion'),
                    engine_path or os.getenv('SCOUT_ENGINE', 'models/yolo11s_fp16.engine'))

    preview_fps = float(os.getenv('SCOUT_PREVIEW_FPS', '15'))
    if not 1 <= preview_fps <= 60:
        raise ValueError('SCOUT_PREVIEW_FPS must be between 1 and 60')
    pipe.stats['preview_limit_fps'] = preview_fps

    @asynccontextmanager
    async def lifespan(app):
        pipe.start()
        yield
        pipe.close()

    app = FastAPI(title='SCOUT Edge Vision', lifespan=lifespan)
    app.state.pipeline = pipe

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    @app.get('/api/stats')
    async def stats():
        return pipe.snapshot()

    @app.get('/stream')
    async def stream():
        async def frames():
            sequence = -1
            while not pipe.stop_event.is_set():
                new_sequence, jpeg = await asyncio.to_thread(pipe.wait_frame, sequence)
                if new_sequence == sequence:
                    continue
                sequence = new_sequence
                if jpeg is None:
                    continue
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n'
                await asyncio.sleep(1 / preview_fps)
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

    return app
