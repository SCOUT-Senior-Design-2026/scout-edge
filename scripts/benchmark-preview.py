#!/usr/bin/env python3
"""Measure MJPEG frames received and summarize the server's live counters."""
import argparse
import json
import threading
import time
import urllib.request


def summarize(samples, key):
    values = [sample[key] for sample in samples if isinstance(sample.get(key), (int, float))]
    if not values:
        return None
    return {'mean': round(sum(values) / len(values), 2), 'min': round(min(values), 2),
            'max': round(max(values), 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', nargs='?', default='http://192.168.55.1:8080', help='SCOUT server base URL')
    parser.add_argument('--seconds', type=float, default=30, help='measurement duration')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    base = args.url.rstrip('/')
    samples, errors = [], []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                with urllib.request.urlopen(base + '/api/stats', timeout=3) as response:
                    samples.append(json.load(response))
            except Exception as exc:
                errors.append(str(exc))
            stop.wait(1)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    boundary = b'--frame\r\n'
    frames, byte_count, tail = 0, 0, b''
    start = time.monotonic()
    try:
        with urllib.request.urlopen(base + '/stream', timeout=5) as response:
            while time.monotonic() - start < args.seconds:
                block = response.read1(65536)
                if not block:
                    break
                data = tail + block
                frames += data.count(boundary)
                tail = data[-(len(boundary) - 1):]
                byte_count += len(block)
    finally:
        elapsed = time.monotonic() - start
        stop.set()
        poller.join(timeout=4)
    running = [sample for sample in samples if sample.get('status') == 'running']
    result = {'duration_seconds': round(elapsed, 2), 'received_frames': frames,
              'received_fps': round(frames / elapsed, 2) if elapsed else 0,
              'received_mbps': round(byte_count * 8 / elapsed / 1e6, 2) if elapsed else 0,
              'running_samples': len(running), 'status_errors': errors,
              'processed_fps': summarize(running, 'capture_fps'),
              'encoded_fps': summarize(running, 'encoded_fps'),
              'detector_fps': summarize(running, 'detector_fps'),
              'tracking_ms': summarize(running, 'tracking_latency_ms'),
              'max_reported_objects': max((len(sample.get('detections', [])) for sample in running), default=0)}
    print(json.dumps(result, indent=2))
    if not running or not frames:
        raise SystemExit('No running video samples were received')


if __name__ == '__main__':
    main()
