#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-libnvinfer python3-libnvinfer-dev libnvinfer-bin python3-dev libboost-python-dev
python3 -m pip install --upgrade pycuda
python3 - <<'PY'
import tensorrt as trt
print('TensorRT Python', trt.__version__)
PY
command -v trtexec || true
