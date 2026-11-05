#!/usr/bin/env bash
set -euo pipefail
ONNX="${1:-models/yolo11s_1080.onnx}"
ENGINE="${2:-models/yolo11s_fp16.engine}"
command -v trtexec >/dev/null || { echo 'trtexec is required on a compatible Jetson build host.' >&2; exit 1; }
trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 --workspace=2048 --skipInference
trtexec --loadEngine="$ENGINE" --shapes=images:1x3x1080x1080 --iterations=100 --duration=60
trtexec --loadEngine="$ENGINE" --dumpLayerInfo --exportTimes="${ENGINE%.engine}.timings.json" --iterations=100
echo "TensorRT engine written to $ENGINE"
