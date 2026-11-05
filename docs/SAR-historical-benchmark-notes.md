# SCOUT Edge Vision

Jetson Xavier edge-inference prototype for 1080p/60 video with browser-viewable overlays.

For moving development and remote-device operation to the onboard workflow, see
[ONBOARD.md](ONBOARD.md). It covers Tailscale on the Jetson, workstation Codex
CLI setup, credential handling, and transition checks.

## Local run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m scout --source 0 --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080`. Set `--source` to a video file for replay. The default detector is a lightweight motion detector so the transport/UI can be validated before installing a TensorRT model. The detector adapter is isolated in `scout/detectors.py` and called by `Pipeline.jpeg()` for replacement with TensorRT or DeepStream labels.

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

Three more people-focused NPS clips are available in the dashboard's Video
selector: Acadia park staff and two Grand Canyon ranger-led hikes. Their source
pages mark the assets public domain. To download these MP4 samples again, run
`scripts/download-sample-videos.sh`. The catalog and source/rights links are in
`media/samples.json`. The selected video changes live while the TensorRT engine
and WebRTC connection remain active; selection returns to the default Acadia
hiking clip after a service restart. File playback is sped up toward 60 FPS,
with the playback multiplier reduced automatically for higher-frame-rate files.
The source list is available at `GET /api/sources`; switch through the dashboard
or send `POST /api/source` with a catalogued `source_id`.

The selector also includes `Xperience stereo RGB (side-by-side, 4:51)`. Its
synchronized left and right views are combined in `media/stereo-xperience/`;
the pipeline displays the left view and uses the right view for stereo depth.

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
# Use a dedicated Python environment with ultralytics, onnx, onnxslim, and onnxruntime installed.
SCOUT_EXPORT_PYTHON=.venv-export/bin/python ./scripts/setup-yolo.sh
```

Copy `models/yolo11s_1088.onnx` to the Xavier, then build and benchmark the Xavier-specific FP16 engine:

```bash
./scripts/build-engine.sh models/yolo11s_1088.onnx models/yolo11s_fp16.engine
SCOUT_DETECTOR=tensorrt SCOUT_ENGINE=models/yolo11s_fp16.engine \
  .venv/bin/python -m scout --source tcp://127.0.0.1:5002
```

The TensorRT detector uses COCO-80 labels, 1088x1088 letterboxed inference for the 1920x1080 source, confidence threshold 0.25, and IoU threshold 0.45. The UI reports active detector, precision, input resolution, inference latency, and measured FPS. A 60 FPS result is only accepted when sustained end-to-end display FPS reaches that target.

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

## YOLO integration checks

The TensorRT adapter accepts a static, batch-one, square NCHW engine with one
raw COCO-80 output (`nms=False`). It decodes YOLO's `[1,84,N]` output, reverses
letterboxing, and applies NMS separately for each class. CUDA context activation
and inference buffers are protected when the web server uses worker threads.

Run the CPU-only adapter regression tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

On Jetson, the build and local benchmark scripts include the TensorRT binary and
CUDA library directories. For direct Python launches, set the library path too:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

If provisioning previously failed with `curand.h` missing, rerun
`sudo ./scripts/provision-tensorrt.sh`; it includes `libcurand-dev-11-4` and `cuda-nvcc-11-4` (for `crt/host_config.h`).
The older local file `models/yolo11s_1080.onnx` can be passed explicitly to
`build-engine.sh`; the benchmark uses the engine's static input dimensions.

Jetson provisioning installs PyCUDA into the SCOUT virtual environment and pins
NumPy to 1.23.5 for compatibility with TensorRT 8.5 and the PyCUDA extension.

### Validated Xavier run (2026-09-19)

Built `models/yolo11s_fp16.engine` from the existing 1088x1088 ONNX model on
TensorRT 8.5.2.2. Real hiking-video inference detected people and a dog; an
annotated frame is saved at `media/yolo-preview.jpg`. Seven adapter regression
tests and worker-thread CUDA checks passed.

A 60.02-second HTTP MJPEG run delivered 625 complete frames (**10.41 FPS**) at
1920x1080 with YOLO11s FP16. This does **not** meet the 60 FPS target. Detailed
results are in `models/yolo-validation.json`; the engine build and benchmark log
is `models/yolo-build.log`. Engine-only throughput was approximately 25
inferences/second, which excludes the full video pipeline.

## YOLO11n performance configuration

Use YOLO11n at 640x640 while retaining 1920x1080 video output. Prepare the
separate export environment on this Python 3.8 Xavier once:

```bash
python3 -m venv .venv-export
.venv-export/bin/python -m pip install 'ultralytics==8.3.40' 'torch==2.4.1' 'torchvision==0.19.1' 'numpy==1.24.4' 'onnx==1.17.0' 'onnxslim==0.1.43' 'onnxruntime==1.16.3'
```

Then export, build, and run:

```bash
SCOUT_EXPORT_PYTHON=.venv-export/bin/python ./scripts/setup-yolo.sh yolo11n 640
./scripts/build-engine.sh models/yolo11n_640.onnx models/yolo11n_640_fp16.engine
SCOUT_DETECTOR=tensorrt SCOUT_ENGINE=models/yolo11n_640_fp16.engine \
  ./scripts/run-local-benchmark.sh media/acadia-hiking-people-1080p.mp4
```

The export environment used on this Python 3.8 Xavier contains ultralytics
8.3.40, torch 2.4.1, torchvision 0.19.1, NumPy 1.24.4, ONNX 1.17.0, and
onnxslim 0.1.43, and ONNX Runtime 1.16.3. Keep these separate from Scout's TensorRT runtime environment.
Engine builds now save a timing cache alongside the engine for subsequent builds.

This device's `/etc/nvpmodel.conf` defines MAXN as mode 0. To enable and verify it:

```bash
sudo nvpmodel -m 0
sudo nvpmodel -q
```

If the command requests a reboot, complete that separately before measuring
performance. Switching the power mode does not itself verify a 60 FPS result.

### Historical YOLO11n + MAXN run (2026-09-19)

At the time of this validation, the runtime used
`models/yolo11n_640_fp16.engine` on port 8080. MAXN mode 0 and all eight CPU
cores were verified after the power-mode change.

That historical run used `scripts/start-scout.sh` with the shared WebRTC/VP8 pipeline
with YOLO11n on the local hiking video showing people. It speeds up the
23.976 FPS clip by 2.5025x and streams at 640x360. Run
`scripts/enable-autostart.sh` once to add an `@reboot` cron entry; startup waits
20 seconds for the device services and writes output to `scout-boot.log`.
The service is available at `http://100.97.217.127:8080` after boot.

