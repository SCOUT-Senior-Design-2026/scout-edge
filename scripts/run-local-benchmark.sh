#!/usr/bin/env bash
set -euo pipefail
VIDEO="${1:-media/sample-60fps.mp4}"
DETECTOR="${SCOUT_DETECTOR:-motion}"
ENGINE="${SCOUT_ENGINE:-models/yolo11s_fp16.engine}"
exec .venv/bin/python -m scout --source "$VIDEO" --detector "$DETECTOR" --engine "$ENGINE" --host 0.0.0.0 --port 8080
