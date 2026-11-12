# SCOUT Lynx target: interface contract

This contract covers the vision dashboard and software interfaces. It does not define or authorize vehicle motion or a physical wiring design.

| ID | Requirement | Owner | Verification |
| --- | --- | --- | --- |
| V1 | Read the configured Lynx RTSP stream and retry after a source failure. | `scout.app.Pipeline`; source selected by `--source` | Pipeline unit tests and a live Lynx stream on the target Jetson. |
| V2 | Serve processed video to USB and hotspot browser clients. | FastAPI `/stream`; Jetson network configuration owns addresses. | HTTP checks and actual client frame counts; hotspot performance requires a separate client test. |
| V3 | Create object tracks only from model detections, predict between detections using optical flow and Kalman filtering, and expire unreconfirmed tracks. | `scout.tracking` | Unit tests, sample video, and live source checks. |
| V4 | Report processed, encoded, and detector rates without claiming camera-to-screen latency or browser display FPS. | `/api/stats` and dashboard | Inspect API fields and UI labels. |
| V5 | Do not present GNSS or RTK values until a Cube telemetry source is implemented. | Dashboard | `telemetry.status` reports `not_configured`; Cube telemetry is not implemented. |
| V6 | For the current 1280×720 Lynx feed, preserve source pixels in the 1280×736 YOLO26 input with vertical padding. | `letterbox_geometry` and target engine | Unit test and engine input-shape inspection. |
| V7 | Leave gimbal control disabled until device direction, bounded motion, stop and watchdog are commissioned. | `scout.gimbal` and dashboard | Unit tests verify gating and stop packets; physical motion and watchdog remain unverified. |

The current RTSP endpoint is assumed to be `rtsp://192.168.144.240:8554/payload`; deployments can supply `--source`. USB address `192.168.55.1` and hotspot address `10.42.0.1` are network configuration values, not guaranteed by SCOUT. The current full-resolution claim applies only to 1280×720 input. Optical flow uses a reduced image and returns box coordinates in the source frame. The YOLO26 engine must be built on the target Jetson; its weights and engine are not committed. Camera-to-screen latency, hotspot-client performance, Cube integration, field detection accuracy, and robot or gimbal motion remain unverified by this software contract.
