"""SCOUT video capture, detection and browser dashboard."""
import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from .detectors import COCO_NAMES, create_detector
from .tracking import create_tracked_detector
from .lowlight import GremsyView, LowLightPolicy
from .gimbal import GimbalControl, GimbalUnavailable

ROOT = Path(__file__).parent
LYNX_SOURCE = 'rtsp://192.168.144.240:8554/payload'


class Pipeline:
    """One capture and detector shared by every browser viewer."""

    def __init__(self, source, mode='tracked', engine='models/yolo26n_1280x736_fp16.engine'):
        self.source = int(source) if str(source).isdigit() else str(source)
        base_detector = create_detector('tensorrt' if mode == 'tracked' else mode, engine)
        self.detector = create_tracked_detector(base_detector) if mode == 'tracked' else base_detector
        self.mode = mode
        self.lowlight = LowLightPolicy() if self.source == LYNX_SOURCE and os.getenv('SCOUT_LOWLIGHT', '1') == '1' else None
        self.camera_view = None
        self.lowlight_fault = None
        self.stop_event = threading.Event()
        self.changed = threading.Condition()
        self.thread = None
        self.encoder_thread = None
        self.raw_frame = None
        self.raw_sequence = 0
        self.capture_generation = 0
        self.frame = None
        self.sequence = 0
        self.stats = {
            'status': 'starting', 'error': None, 'detector': self.detector.name,
            'precision': self.detector.precision, 'input_resolution': None,
            'capture_fps': 0.0, 'inference_fps': 0.0, 'display_fps': None,
            'latency_ms': None, 'inference_latency_ms': None, 'detections': [],
            'source_name': 'Gremsy Lynx' if self.source == LYNX_SOURCE else 'Configured source',
            'telemetry': {'status': 'not_configured'},
            'tracking': None, 'detector_fps': None, 'detector_latency_ms': None,
            'detection_age_ms': None, 'detector_status': None,
            'model_input_resolution': getattr(self.detector, 'input_resolution', None),
            'flow_resolution': None,
            'camera_mode': 'visible', 'lowlight_status': 'starting' if self.lowlight else 'disabled',
            'visible_luminance': None,
            'encoded_fps': 0.0, 'encoder_error': None, 'preview_dropped': 0,
        }

    def start(self):
        self.encoder_thread = threading.Thread(target=self.encode, name='scout-encoder', daemon=True)
        self.encoder_thread.start()
        self.thread = threading.Thread(target=self.run, name='scout-capture', daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread:
            self.thread.join(timeout=6)
        if self.encoder_thread:
            self.encoder_thread.join(timeout=3)
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

    def encode(self):
        """Encode the latest processed frame without blocking capture or tracking."""
        sequence, count, interval = 0, 0, time.monotonic()
        while not self.stop_event.is_set():
            with self.changed:
                self.changed.wait_for(lambda: self.raw_sequence != sequence or self.stop_event.is_set(), timeout=2)
                if self.stop_event.is_set():
                    return
                if self.raw_sequence == sequence:
                    continue
                skipped = max(0, self.raw_sequence - sequence - 1)
                sequence, frame, generation = self.raw_sequence, self.raw_frame, self.capture_generation
                self.stats['preview_dropped'] += skipped
            if frame is None:
                continue
            try:
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                encoded = encoded.tobytes()
            except Exception as exc:
                with self.changed:
                    if generation == self.capture_generation:
                        self.frame = None
                        self.sequence += 1
                        self.stats.update(encoder_error=str(exc), encoded_fps=0.0)
                        self.changed.notify_all()
                continue
            now = time.monotonic()
            with self.changed:
                if generation != self.capture_generation:
                    continue
                count += 1
                if now - interval >= 1:
                    self.stats['encoded_fps'] = round(count / (now - interval), 1)
                    count, interval = 0, now
                self.stats['encoder_error'] = None
                self.frame = encoded
                self.sequence += 1
                self.changed.notify_all()

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
                    if self.lowlight:
                        try:
                            if self.lowlight_fault:
                                raise RuntimeError(self.lowlight_fault)
                            if self.camera_view is None:
                                self.camera_view = GremsyView()
                            desired = self.lowlight.update(frame, started)
                            if desired:
                                self.camera_view.set_mode(desired == 'thermal')
                                self.lowlight.switched(desired, started)
                                reset = getattr(self.detector, 'reset', None)
                                if reset:
                                    reset()
                                self.update(camera_mode=desired, lowlight_status='active', detections=[])
                                continue
                            self.update(lowlight_status='active', visible_luminance=self.lowlight.luminance)
                        except Exception as exc:
                            self.lowlight_fault = str(exc)
                            self.update(lowlight_status='error: ' + str(exc))
                            raise RuntimeError('Automatic camera switching failed: ' + str(exc)) from exc
                    infer_started = time.monotonic()
                    result = self.detector.infer(frame)
                    detections = result[0] if isinstance(result, tuple) else result
                    infer_ms = (time.monotonic() - infer_started) * 1000
                    labels = getattr(self.detector, 'class_names', COCO_NAMES)
                    summary = []
                    for detection in detections[:50]:
                        label = 'motion' if self.mode == 'motion' else (labels[detection.class_id] if 0 <= detection.class_id < len(labels) else str(detection.class_id))
                        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), (79, 224, 117), 2)
                        if self.mode == 'motion':
                            text = label
                        elif hasattr(detection, 'track_id'):
                            text = (f'{label} #{detection.track_id} / tracked' if detection.tracked
                                    else f'{label} #{detection.track_id} {detection.confidence:.0%}')
                        else:
                            text = f'{label} {detection.confidence:.0%}'
                        cv2.putText(frame, text, (detection.x1, max(22, detection.y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .65, (79, 224, 117), 2)
                        summary.append({'label': label, 'confidence': None if self.mode == 'motion' else round(float(detection.confidence), 3),
                                        'track_id': getattr(detection, 'track_id', None),
                                        'tracked': getattr(detection, 'tracked', False),
                                        'detection_age_ms': getattr(detection, 'detection_age_ms', None)})
                    tracking_status = self.detector.status() if self.mode == 'tracked' else {}
                    count += 1
                    now = time.monotonic()
                    with self.changed:
                        if now - interval >= 1:
                            rate = round(count / (now - interval), 1)
                            self.stats.update(capture_fps=rate,
                                              inference_fps=(tracking_status.get('detector_fps', 0.0)
                                                             if self.mode == 'tracked' else rate))
                            count, interval = 0, now
                        self.stats.update(status='running', error=None,
                            input_resolution=f'{frame.shape[1]}x{frame.shape[0]}',
                            latency_ms=round((now - started) * 1000, 1),
                            inference_latency_ms=(tracking_status.get('detector_latency_ms') if self.mode == 'tracked'
                                                  else round(infer_ms, 1)),
                            tracking_latency_ms=round(infer_ms, 1) if self.mode == 'tracked' else None,
                            detections=summary[:20], **tracking_status)
                        self.raw_frame = frame
                        self.raw_sequence += 1
                        self.changed.notify_all()
            except Exception as exc:
                reset = getattr(self.detector, 'reset', None)
                if reset:
                    reset()
                with self.changed:
                    self.capture_generation += 1
                    self.raw_frame = None
                    self.raw_sequence += 1
                    self.frame = None
                    self.sequence += 1
                    self.stats.update(status='reconnecting', error=str(exc), capture_fps=0.0,
                                      inference_fps=0.0, encoded_fps=0.0, detections=[])
                    self.changed.notify_all()
                self.stop_event.wait(2)
            finally:
                if cap is not None:
                    cap.release()
        if self.camera_view:
            try:
                if self.lowlight.mode == 'thermal':
                    self.camera_view.set_mode(False)
            finally:
                self.camera_view.close()
        self.update(status='stopped')


def create_app(source=LYNX_SOURCE, detector_mode=None, engine_path=None):
    pipe = Pipeline(source, detector_mode or os.getenv('SCOUT_DETECTOR', 'tracked'),
                    engine_path or os.getenv('SCOUT_ENGINE', 'models/yolo26n_1280x736_fp16.engine'))

    preview_fps = float(os.getenv('SCOUT_PREVIEW_FPS', '60'))
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
    app.state.gimbal = GimbalControl.from_environment()

    class GimbalMove(BaseModel):
        direction: str
        token: str

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    @app.get('/api/stats')
    async def stats():
        return pipe.snapshot()

    @app.get('/api/gimbal')
    async def gimbal_status():
        return app.state.gimbal.status()

    @app.post('/api/gimbal/move')
    async def gimbal_move(move: GimbalMove):
        try:
            await asyncio.to_thread(app.state.gimbal.move, move.direction, move.token)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GimbalUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {'status': 'sent'}

    @app.get('/stream')
    async def stream():
        async def frames():
            sequence = -1
            next_send = time.monotonic()
            while not pipe.stop_event.is_set():
                delay = next_send - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                new_sequence, jpeg = await asyncio.to_thread(pipe.wait_frame, sequence)
                if new_sequence == sequence:
                    continue
                sequence = new_sequence
                if jpeg is None:
                    continue
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n'
                next_send = max(next_send + 1 / preview_fps, time.monotonic())
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-store'})

    return app
