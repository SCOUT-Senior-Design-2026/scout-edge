import asyncio, os, time
from pathlib import Path
import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from .detectors import COCO_NAMES, create_detector

ROOT=Path(__file__).parent
class Pipeline:
    def __init__(self,source,mode='motion',engine='models/yolo11s_fp16.engine'):
        self.source=int(source) if str(source).isdigit() else source; self.cap=None; self.detector=create_detector(mode,engine); self.started=time.perf_counter(); self.frames=0
        self.stats={'capture_fps':0,'inference_fps':0,'display_fps':0,'latency_ms':0,'inference_latency_ms':0,'queue_depth':0,'dropped':0,'detector':self.detector.name,'precision':self.detector.precision,'input_resolution':'unknown','status':'ready','error':None}
    def jpeg(self):
        start=time.perf_counter()
        if self.cap is None:
            backend=cv2.CAP_FFMPEG if isinstance(self.source,str) else cv2.CAP_ANY; self.cap=cv2.VideoCapture(self.source,backend); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080); self.cap.set(cv2.CAP_PROP_FPS,60)
        ok,frame=self.cap.read()
        if not ok:
            if isinstance(self.source,str) and not self.source.startswith(('tcp://','rtsp://','http://','https://')):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,frame=self.cap.read()
            if not ok: self.stats.update(status='capture_error',error='input frame unavailable'); return None
        self.stats['input_resolution']=f'{frame.shape[1]}x{frame.shape[0]}'; infer=time.perf_counter(); result=self.detector.infer(frame); detections=result[0] if isinstance(result,tuple) else result; self.stats['inference_latency_ms']=round((time.perf_counter()-infer)*1000,1)
        for d in detections:
            label=COCO_NAMES[d.class_id] if d.class_id<len(COCO_NAMES) else str(d.class_id); cv2.rectangle(frame,(d.x1,d.y1),(d.x2,d.y2),(0,220,80),2); cv2.putText(frame,f'{label} {d.confidence:.2f}',(d.x1,max(20,d.y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,80),2)
        self.frames+=1; elapsed=time.perf_counter()-self.started
        if elapsed>=1: fps=self.frames/elapsed; self.stats.update(capture_fps=fps,inference_fps=fps,display_fps=fps); self.started=time.perf_counter(); self.frames=0
        self.stats.update(status='running',error=None,latency_ms=round((time.perf_counter()-start)*1000,1)); return cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])[1].tobytes()

async def read_frame(pipe): return await asyncio.get_event_loop().run_in_executor(None,pipe.jpeg)
def create_app(source,detector_mode=None,engine_path=None):
    app=FastAPI(title='SCOUT Edge Vision'); pipe=Pipeline(source,detector_mode or os.getenv('SCOUT_DETECTOR','motion'),engine_path or os.getenv('SCOUT_ENGINE','models/yolo11s_fp16.engine'))
    @app.get('/')
    async def index(): return FileResponse(ROOT/'static'/'index.html')
    async def frames():
        while True:
            jpeg=await read_frame(pipe)
            if jpeg: yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
    @app.get('/stream')
    async def stream(): return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame')
    @app.get('/api/stats')
    async def stats(): return pipe.stats
    return app
