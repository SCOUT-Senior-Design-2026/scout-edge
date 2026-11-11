#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
onnx="${1:-models/yolo26n_1280x736.onnx}"
engine="${2:-models/yolo26n_1280x736_fp16.engine}"
trtexec="${SCOUT_TRTEXEC:-$(command -v trtexec || true)}"
if [[ -z "$trtexec" && -x /usr/src/tensorrt/bin/trtexec ]]; then
  trtexec=/usr/src/tensorrt/bin/trtexec
fi
if [[ -z "$trtexec" || ! -x "$trtexec" ]]; then
  echo 'TensorRT trtexec is required on the target Jetson.' >&2
  exit 1
fi
if [[ ! -f "$onnx" ]]; then
  echo "ONNX model not found: $onnx" >&2
  exit 1
fi
"$trtexec" --onnx="$onnx" --saveEngine="$engine" --fp16 \
  --memPoolSize=workspace:1024 --builderOptimizationLevel=2 --skipInference
echo "TensorRT engine written to $engine"
