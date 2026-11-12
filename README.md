# SCOUT Edge Vision

SCOUT reads the Gremsy Lynx RTSP feed on an Orin NX and serves a browser dashboard with object boxes. YOLO26n runs on TensorRT in a worker; optical flow and a Kalman filter update model-seeded tracks between detector frames. Tracks expire when the detector does not reconfirm them. The motion-only baseline is available for diagnostics but is not the default.

## Lynx dashboard on Orin NX

The default source is `rtsp://192.168.144.240:8554/payload`. Build a model engine for the configured stream size before claiming native-resolution detection. For 1280×720, use a 1280×736 tensor; for 1920×1080, use 1920×1088. Both preserve source pixels and pad vertically. The setup scripts default to 720p; select a separately built 1080p engine with `SCOUT_ENGINE` for a 1080p stream. Optical flow runs at one-quarter width and height; its resulting boxes are mapped back to the original video coordinates.

For the default Lynx source, low-light switching uses the camera's MAVLink view control. After five seconds with median visible luminance below 22/255, it switches the main feed to thermal. Every 30 seconds it briefly checks visible light; a median of at least 35/255 returns to visible mode. Other viewers of the main stream see the same changes. Set `SCOUT_LOWLIGHT=0` to disable this behavior. The existing YOLO26 model is trained for visible imagery and has not been validated on thermal imagery; thermal detections may be unreliable. The thermal sensor is 640×512 even when its view is encoded in a 1080p stream. The thresholds and switching timing require field validation.

The Jetson needs Ethernet connectivity to the Lynx mainboard. With the engine installed, start the app as the `jetson` user:

```bash
cd ~/scout-edge
.venv/bin/python -m scout --host 0.0.0.0 --port 8080
```

Open `http://192.168.55.1:8080` from the USB-connected computer or `http://10.42.0.1:8080` from a device connected to the Jetson hotspot. The dashboard is available to network clients without an application login. Full screen expands the video in the browser; it does not configure HDMI output.

The dashboard distinguishes processed frames, encoded preview frames, and detector rate. The browser preview defaults to a maximum of 60 FPS; `SCOUT_PREVIEW_FPS` can lower it for limited bandwidth. MJPEG bandwidth and actual browser display rate depend on the scene, network, and client. Processing time starts after capture and excludes preview encoding and display; it is not camera-to-screen latency. Cube/GNSS telemetry is not yet integrated. See the [hardware target interface contract](docs/hardware-target.md) for assumptions and verification limits.

The gimbal panel contains short directional step controls. Select Enable gimbal to check availability and arm the buttons after entering the operator token; select Disable gimbal to clear it. They are disabled until the specific Lynx has completed guarded commissioning, including direction, limits, stop behavior and a device-side command timeout; a software stop packet alone cannot stop motion after a lost link. The service requires `SCOUT_GIMBAL_COMMISSIONED=1`, `SCOUT_GIMBAL_WATCHDOG_VERIFIED=1`, and a private `SCOUT_GIMBAL_TOKEN` only after those checks. A button issues a 150 ms, 2°/s rate command followed by a stop. The dashboard is HTTP, so the token and commands are not encrypted; restrict the network accordingly. No physical motion or watchdog behavior has been verified by the unit tests.

The model, exported ONNX file, and TensorRT engine are local artifacts excluded from Git. A TensorRT engine must be built for the target Jetson and its installed TensorRT version. If the default engine is missing, the app reports the error instead of silently switching to motion boxes.

The Lynx supports onboard 4K recording, but its documented Ethernet live output is HD. A 4K recording can be supplied as a source with a separately built 3840×2176 engine: the 3840×2160 picture is padded by eight pixels above and below, without resizing. This is a supported input geometry, not a claim of 60 FPS at 4K. The camera also has digital zoom; its effect on sensor sampling and this particular live stream has not been verified.

## Preparing YOLO26 and installing on Jetson

On a computer with an export environment containing Ultralytics, PyTorch, and ONNX, download the official YOLO26n weights and export them. The weight file used for this setup is the Ultralytics `v8.4.0` release and has SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.

```bash
mkdir -p models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt -o models/yolo26n.pt
SCOUT_EXPORT_PYTHON=/path/to/export-env/bin/python ./scripts/setup-yolo.sh
```

Copy `models/yolo26n_1280x736.onnx` to the same path in `~/scout-edge` on the Jetson. Build the engine on that Jetson, then run the installer as the `jetson` user:

```bash
cd ~/scout-edge
./scripts/build-engine.sh
./scripts/install-jetson.sh
```

For a 4K recording, export with `./scripts/setup-yolo.sh models/yolo26n.pt models/yolo26n_3840x2176.onnx 2176 3840`, copy that ONNX to the Jetson, and build it with `./scripts/build-engine.sh models/yolo26n_3840x2176.onnx models/yolo26n_3840x2176_fp16.engine`. Launch with `SCOUT_ENGINE=models/yolo26n_3840x2176_fp16.engine .venv/bin/python -m scout --source /path/to/recording.mp4`. Export and build resource requirements are higher than for the default live feed.

The installer requests sudo for system packages, installs `requirements-jetson.txt` in `.venv`, and starts a user service after the engine exists. JetPack's TensorRT Python bindings, CUDA runtime, and OpenCV are required. PyCUDA is optional: the app uses the CUDA runtime when PyCUDA is absent. The installer needs network access and the Ubuntu `python3-venv` package. The existing ARK UI on port 80 is separate from SCOUT on port 8080.

To use a different source or engine, pass `--source` or set `SCOUT_ENGINE` when starting the app. `SCOUT_DETECT_INTERVAL` (default 5) controls detector scheduling in processed frames, `SCOUT_FLOW_INTERVAL` (default 2) controls optical-flow updates, and `SCOUT_TRACK_MAX_AGE` (default 1.0 seconds) bounds how long an unconfirmed track remains. Kalman prediction runs on each processed frame. These settings trade detection freshness and tracking quality against throughput; they do not establish a safety guarantee.

Run the focused tests with `.venv/bin/python -m unittest discover -s tests`. For rate measurements use `/api/stats` and count frames received by the actual client. Synthetic or prerecorded tests do not establish field detection accuracy or end-to-end display latency. Earlier Xavier experiments are retained in [SAR historical benchmark notes](docs/SAR-historical-benchmark-notes.md).

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


Historical detector and live pipeline measurements from the original SCOUT SAR work are preserved in [SAR historical benchmark notes](docs/SAR-historical-benchmark-notes.md). The reviewed SAR v1 annotations and the explicitly unreviewed model prelabels are under `evaluation/`; YOLO sweep reports are under `models/`. Source clips are downloaded with `scripts/download-sample-videos.sh`; they are intentionally not committed.
