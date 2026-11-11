# SCOUT Lynx target: interface contract

This contract covers the vision dashboard and its software interfaces. It does not define or authorize vehicle motion or a physical wiring design.

| ID | Requirement | Owner | Verification |
| --- | --- | --- | --- |
| V1 | Read the Lynx mainboard's configured RTSP stream and retry after a source failure. | `scout.app.Pipeline`; source selected by `--source` | Pipeline unit tests and a live Lynx stream on the target Jetson. |
| V2 | Serve one processed stream to USB and hotspot browser clients. | FastAPI `/stream`; Jetson network configuration owns addresses. | HTTP checks on both local interfaces and concurrent USB clients. |
| V3 | Name the active detector and label motion as motion. | `scout.app.Pipeline` and dashboard | Unit test and live dashboard inspection. |
| V4 | Present processing throughput without claiming camera-to-screen latency or browser display FPS. | `/api/stats` and dashboard | Inspect API fields and UI labels. |
| V5 | Do not present GNSS or RTK values until a Cube telemetry source is implemented. | Dashboard | `telemetry.status` reports `not_configured`; Cube telemetry is not implemented. |

The current Lynx RTSP endpoint is assumed to be `rtsp://192.168.144.240:8554/payload`; deployments with a different address can supply `--source`. USB address `192.168.55.1` and hotspot address `10.42.0.1` are network configuration values, not guaranteed by SCOUT. The motion baseline does not classify people. TensorRT requires a compatible engine and runtime. Camera-to-screen latency, actual hotspot-client performance, Cube integration, and any robot or gimbal motion remain unverified by this software contract.
