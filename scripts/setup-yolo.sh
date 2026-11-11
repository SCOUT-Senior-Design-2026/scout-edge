#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
weights="${1:-models/yolo26n.pt}"
output="${2:-models/yolo26n_1280x736.onnx}"
height="${3:-736}"
width="${4:-1280}"
export_python="${SCOUT_EXPORT_PYTHON:-python3}"
if [[ ! -f "$weights" ]]; then
  echo "Weights not found: $weights. Download the official YOLO26 weights first." >&2
  exit 1
fi
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PWD/.yolo-config}"
mkdir -p "$YOLO_CONFIG_DIR" "$(dirname "$output")"
"$export_python" - "$weights" "$output" "$height" "$width" <<'PY'
from pathlib import Path
import shutil
import sys
from ultralytics import YOLO

weights, output = Path(sys.argv[1]), Path(sys.argv[2])
height, width = int(sys.argv[3]), int(sys.argv[4])
if height <= 0 or width <= 0 or height % 32 or width % 32:
    raise SystemExit('Model dimensions must be positive multiples of 32')
exported = Path(YOLO(str(weights)).export(format='onnx', imgsz=(height, width),
                                          opset=17, simplify=False, dynamic=False,
                                          nms=False, device='cpu'))
if exported.resolve() != output.resolve():
    shutil.copy2(exported, output)
print(f'ONNX model written to {output} ({width}x{height})')
PY
