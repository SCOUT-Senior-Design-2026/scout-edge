#!/usr/bin/env bash
set -euo pipefail

# Generate a deterministic 1080p60 test source and stream it as MPEG-TS over
# the USB link. The Xavier can consume this with OpenCV's VideoCapture.
MODE="${1:-sample}"
DEVICE="${2:-/dev/video0}"
PORT="${3:-5000}"
if [ "$MODE" = "sample" ] && [[ "${2:-}" =~ ^[0-9]+$ ]]; then PORT="$2"; fi
OUT="${TMPDIR:-/tmp}/scout-sample-1080p60.mp4"
if [ "$MODE" = "webcam" ]; then
  echo "Streaming webcam $DEVICE to tcp://192.168.55.100:$PORT"
  # The tested webcam exposes 1080p as MJPEG at 30 FPS.
  while true; do
    ffmpeg -hide_banner -f v4l2 -input_format mjpeg -framerate 30 -video_size 1920x1080 \
      -i "$DEVICE" -an -c:v libx264 -preset ultrafast -tune zerolatency \
      -pix_fmt yuv420p -f mpegts "tcp://0.0.0.0:$PORT?listen" || true
    sleep 1
  done
fi
if [ "$MODE" != "sample" ]; then
  echo "usage: $0 [sample|webcam] [device] [port]" >&2
  exit 2
fi
if [ ! -s "$OUT" ]; then
  ffmpeg -y -hide_banner -loglevel warning \
    -f lavfi -i "testsrc2=size=1920x1080:rate=60" -t 30 \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p "$OUT"
fi
echo "Listening for the Xavier at tcp://192.168.55.100:$PORT"
while true; do
  ffmpeg -re -stream_loop -1 -i "$OUT" -an -c copy -f mpegts "tcp://0.0.0.0:$PORT?listen" || true
  sleep 1
done
