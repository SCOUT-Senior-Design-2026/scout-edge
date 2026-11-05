#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
python3 -m pip install --upgrade ultralytics onnx onnxslim
python3 - <<'PY'
from ultralytics import YOLO
YOLO('yolo11s.pt').export(format='onnx', imgsz=1080, opset=17, simplify=True, dynamic=False)
PY
mv -f yolo11s.onnx models/yolo11s_1080.onnx
echo "ONNX model written to models/yolo11s_1080.onnx"
