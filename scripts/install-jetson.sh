#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -eq 0 ]; then echo 'Run as the Jetson user; the script requests sudo for system packages.' >&2; exit 1; fi
cd "$(dirname "$0")/.."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip python3-opencv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-jetson.txt
if [ ! -f models/yolo26n_1280x736_fp16.engine ]; then
  echo 'Build models/yolo26n_1280x736_fp16.engine on this Jetson before starting the service.' >&2
  echo 'See the Lynx dashboard instructions in README.md.' >&2
  exit 1
fi
mkdir -p "$HOME/.config/systemd/user"
cp scripts/scout-edge.service "$HOME/.config/systemd/user/scout-edge.service"
if ! (systemctl --user daemon-reload && systemctl --user enable --now scout-edge.service) 2>/dev/null; then
  # Headless SSH sessions often have no user D-Bus. Keep the service usable
  # without requiring a desktop login or system-wide unit installation.
  if [ -f scout.pid ] && kill -0 "$(cat scout.pid)" 2>/dev/null; then kill "$(cat scout.pid)"; fi
  nohup .venv/bin/python -m scout --source "${SCOUT_SOURCE:-rtsp://192.168.144.240:8554/payload}" --host 0.0.0.0 --port 8080 > scout.log 2>&1 &
  echo $! > scout.pid
  sleep 2
fi
echo "SCOUT is running at http://$(hostname -I | awk '{print $1}'):8080"
