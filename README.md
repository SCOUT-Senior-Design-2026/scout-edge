# SCOUT Edge Vision

Jetson Xavier edge-inference prototype for 1080p/60 video with browser-viewable overlays.

## Local run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m scout --source 0 --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`. Set `--source` to a video file for replay. The default detector is a lightweight motion detector so the transport/UI can be validated before installing a TensorRT model. The detector adapter is isolated in `Pipeline.read()` for replacement with TensorRT or DeepStream labels.

## Jetson USB deployment

With the Xavier connected by USB device mode, run:

```bash
./scripts/deploy-usb.sh xavier@192.168.55.1
```

The script copies the project to `/home/xavier/scout-edge`. On the Jetson, run this once from an interactive SSH session (sudo may request the Xavier user's password):

```bash
cd ~/scout-edge
sudo ./scripts/install-jetson.sh
```

This installs ARM-compatible system packages, creates the Python environment, and starts the user service at `http://192.168.55.1:8080`. No credentials are stored in the repository.

## Measurement

The UI reports capture FPS, inference FPS, end-to-end latency, frame drops, and detector name. This is the evidence needed to evaluate whether the Xavier sustains 60 input frames/sec; overlay smoothness alone is not sufficient.

## Xavier-local benchmark video

The people-demo clip is stored locally on the Xavier at `media/acadia-hiking-people-1080p.mp4`, so testing no longer depends on the laptop, webcam, USB video tunnel, or network jitter. It is a 1920x1080 H.264, 23.976 FPS, 33-second National Park Service clip showing people hiking; the NPS asset metadata marks it public domain. SHA-256: `b33d0313f080c7f946a6e17836b4246416a671cb0022e91f60d97205cf220a65`.

Run it locally on the Xavier:

```bash
cd /home/xavier/scout-edge
SCOUT_DETECTOR=motion ./scripts/run-local-benchmark.sh media/acadia-hiking-people-1080p.mp4
```

For YOLO after the engine is built:

```bash
SCOUT_DETECTOR=tensorrt \
SCOUT_ENGINE=models/yolo11s_fp16.engine \
./scripts/run-local-benchmark.sh media/acadia-hiking-people-1080p.mp4
```

Collect a fair 60-second measurement from another Xavier SSH session:

```bash
cd /home/xavier/scout-edge
.venv/bin/python scripts/benchmark-stats.py http://127.0.0.1:8080/api/stats 60
```

## YOLO11s TensorRT

Provision TensorRT on the Xavier:

```bash
sudo ./scripts/provision-tensorrt.sh
```

On the laptop, export the fixed 1080p ONNX model:

```bash
./scripts/setup-yolo.sh
```

Copy `models/yolo11s_1080.onnx` to the Xavier, then build and benchmark the Xavier-specific FP16 engine:

```bash
./scripts/build-engine.sh models/yolo11s_1080.onnx models/yolo11s_fp16.engine
SCOUT_DETECTOR=tensorrt SCOUT_ENGINE=models/yolo11s_fp16.engine \
  .venv/bin/python -m scout --source tcp://127.0.0.1:5002
```

The TensorRT detector uses COCO-80 labels, 1080x1080 letterboxed inference, confidence threshold 0.25, and IoU threshold 0.45. The UI reports active detector, precision, input resolution, inference latency, and measured FPS. A 60 FPS result is only accepted when sustained end-to-end display FPS reaches that target.

## Laptop sample video benchmark

To test the Xavier without a camera, start the laptop-side 1080p/60 sample stream:

```bash
./scripts/serve-sample-video.sh sample 5002
```

In another terminal, restart the Xavier app against the USB-network stream:

```bash
ssh -t xavier@192.168.55.1 \
  'cd /home/xavier/scout-edge && kill "$(cat scout.pid 2>/dev/null)" 2>/dev/null || true; \
   nohup .venv/bin/python -m scout --source tcp://127.0.0.1:5002 --host 0.0.0.0 --port 8080 > scout.log 2>&1 & echo $! > scout.pid'
```

Create the reverse tunnel before the restart command:

```bash
ssh -o ExitOnForwardFailure=yes -fN -R 5002:127.0.0.1:5002 xavier@192.168.55.1
```

The sample is generated locally, looped indefinitely, and sent over SSH; no video is uploaded to an external service. For a repeatable 60-second measurement, run:

```bash
ssh xavier@192.168.55.1 'cd /home/xavier/scout-edge && .venv/bin/python scripts/benchmark-stats.py http://127.0.0.1:8080/api/stats 60'
```

## Laptop webcam

Use the laptop webcam instead of the generated sample:

```bash
./scripts/serve-sample-video.sh webcam /dev/video0 5000
```

The remaining Jetson restart command is the same. If `/dev/video0` is not the camera, inspect available devices with `ls -l /dev/video*` and try `/dev/video1`, `/dev/video2`, or `/dev/video3`. The tested webcam exposes 1920x1080 MJPEG at 30 FPS; use the generated sample mode when testing a 60 FPS input path.

If the laptop firewall blocks direct TCP access from the Xavier, use an SSH reverse tunnel. Leave the webcam streamer running, then run:

```bash
ssh -o ExitOnForwardFailure=yes -fN \
  -R 5000:127.0.0.1:5000 xavier@192.168.55.1
ssh -t xavier@192.168.55.1 \
  'cd /home/xavier/scout-edge && kill "$(cat scout.pid 2>/dev/null)" 2>/dev/null || true; \
   nohup .venv/bin/python -m scout --source tcp://127.0.0.1:5000 \
   --host 0.0.0.0 --port 8080 > scout.log 2>&1 & echo $! > scout.pid'
```

Use the HTTP URL `http://192.168.55.1:8080`; opening the local `file:///.../index.html` bypasses the running server and will not display the stream.


## Search and rescue benchmark

SCOUT evaluates a COCO detector on search-and-rescue targets: people, vehicles,
and carried items such as backpacks. The runtime demo and benchmark harness are
separate: `scripts/evaluate-sar.py` scores labeled frames at IoU 0.5 and reports
precision/recall by class group, partial-person recall, detector latency, and
optional detector-only throughput. It does not measure end-to-end live FPS; use
`/api/stats` and `scripts/benchmark-stats.py` for that gate.

The benchmark manifest and source videos are local evaluation data and are not
committed. A manifest contains an `images` array; each item names a video path
(relative to the manifest), `time_seconds`, and `annotations` with COCO `class_id`,
`bbox` in `[x1, y1, x2, y2]` pixels, and optional `partial: true`. Mark reviewed
manifests with `evaluation_ready: true`. Example invocation:

```bash
.venv/bin/python scripts/evaluate-sar.py evaluation/sar-v1.json \
  --engine yolo26n=models/yolo26n_768_fp16.engine \
  --throughput-seconds 10 --output evaluation/sar-results.json
```

Do not treat model-generated prelabels as ground truth until they have been
independently reviewed. Keep TensorRT engines, raw recordings, and private
annotations out of Git; the project `.gitignore` excludes local media and
benchmark outputs.
