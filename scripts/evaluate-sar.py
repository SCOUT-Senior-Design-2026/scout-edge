#!/usr/bin/env python3
"""Evaluate TensorRT engines against a timestamped, hand-labeled SAR frame set."""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scout.detectors import TensorRTDetector  # noqa: E402
from scout.evaluation import frame_counts, merge_counts, metrics  # noqa: E402

GROUPS = {
    'people': {0},
    'vehicles': {1, 2, 3, 4, 5, 6, 7, 8},
    'carried_items': {24, 26, 28, 30, 31, 39, 67},
}
SAR_CLASS_IDS = set().union(*GROUPS.values())


def parse_engine(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError('Engines use NAME=PATH, e.g. 768=models/yolo26n_768_fp16.engine')
    name, path = value.split('=', 1)
    return name, path


def load_frames(manifest):
    with open(manifest, encoding='utf-8') as source:
        dataset = json.load(source)
    if dataset.get('evaluation_ready') is False:
        raise ValueError('Manifest is marked as prelabels only; independently verify its annotations before scoring')
    frames = []
    for item in dataset['images']:
        path = (Path(manifest).resolve().parent / item['video']).resolve()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError('Unable to open evaluation video: {}'.format(path))
        capture.set(cv2.CAP_PROP_POS_MSEC, float(item['time_seconds']) * 1000)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError('Unable to read {} at {} seconds'.format(path, item['time_seconds']))
        frames.append((item, image))
    return frames


def run_throughput(detector, frames, seconds):
    if seconds <= 0:
        return None
    for _, image in frames[:3]:
        detector.infer(image)
    deadline = time.perf_counter() + seconds
    count = 0
    started = time.perf_counter()
    while time.perf_counter() < deadline:
        detector.infer(frames[count % len(frames)][1])
        count += 1
    elapsed = time.perf_counter() - started
    return round(count / elapsed, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='SAR evaluation JSON manifest')
    parser.add_argument('--engine', action='append', type=parse_engine, required=True,
                        help='NAME=PATH; repeat for each static TensorRT engine')
    parser.add_argument('--confidence', action='append', type=float, default=[],
                        help='Global confidence to test (default: .60, .55, .50, .45, .40)')
    parser.add_argument('--class-thresholds', default='{}',
                        help='Legacy: JSON object of COCO class ids to confidence overrides')
    parser.add_argument('--class-profile', action='append', default=[],
                        help='Named class thresholds, e.g. SAR={"0":0.45,"24":0.50}; repeatable')
    parser.add_argument('--throughput-seconds', type=float, default=0,
                        help='Optional detector-only steady-state throughput run per setting')
    parser.add_argument('--output', help='Write the full JSON report to this file')
    args = parser.parse_args()
    confidences = args.confidence or [.60, .55, .50, .45, .40]
    legacy_thresholds = json.loads(args.class_thresholds)
    if not isinstance(legacy_thresholds, dict):
        parser.error('--class-thresholds must be a JSON object')
    profiles = [('global', {})]
    if legacy_thresholds:
        profiles.append(('custom', legacy_thresholds))
    for value in args.class_profile:
        if '=' not in value:
            parser.error('--class-profile uses NAME=JSON')
        name, encoded = value.split('=', 1)
        thresholds = json.loads(encoded)
        if not name or not isinstance(thresholds, dict):
            parser.error('--class-profile requires a name and a JSON object')
        profiles.append((name, thresholds))
    frames = load_frames(args.manifest)
    report = dict(iou_threshold=.5, image_count=len(frames),
                  confidence_sweep=confidences,
                  class_profiles=dict(profiles),
                  live_60_fps_gate='not measured by this detector-only harness; verify with the live dashboard')
    settings = []
    for engine_name, engine_path in args.engine:
        for profile_name, class_thresholds in profiles:
          for confidence in confidences:
            detector = TensorRTDetector(engine_path, confidence=confidence,
                                        class_confidences=class_thresholds)
            merged = {}
            partial_people = {}
            latency_ms = []
            for _, image in frames[:3]:
                detector.infer(image)
            for item, image in frames:
                detections, latency = detector.infer(image)
                detections = [detection for detection in detections
                              if detection.class_id in SAR_CLASS_IDS]
                latency_ms.append(latency)
                merge_counts(merged, frame_counts(item['annotations'], detections, .5))
                partial_annotations = [annotation for annotation in item['annotations']
                                       if annotation.get('partial') and annotation['class_id'] == 0]
                partial_detections = [detection for detection in detections if detection.class_id == 0]
                merge_counts(partial_people,
                             frame_counts(partial_annotations, partial_detections, .5))
            all_metrics = metrics(merged)
            partial_result = metrics(partial_people)['overall']
            partial_result = {key: partial_result[key] for key in ('tp', 'fn', 'recall')}
            grouped_metrics = {}
            for group, class_ids in GROUPS.items():
                group_counts = {cid: merged[cid] for cid in class_ids if cid in merged}
                grouped_metrics[group] = metrics(group_counts)['overall']
            measured_fps = run_throughput(detector, frames, args.throughput_seconds)
            settings.append(dict(engine=engine_name, engine_path=engine_path,
                                 engine_input_resolution=detector.input_resolution,
                                 confidence=confidence, class_profile=profile_name,
                                 class_thresholds=class_thresholds,
                                 metrics=all_metrics,
                                 groups=grouped_metrics,
                                 partial_people=partial_result,
                                 mean_detector_latency_ms=round(sum(latency_ms) / len(latency_ms), 2),
                                 detector_only_fps=measured_fps))
            detector.cuda_context.detach()
    baseline = next((item for item in settings
                     if item['confidence'] == .60 and item['class_profile'] == 'global'), None)
    if baseline:
        base_precision = baseline['metrics']['overall']['precision']
        for item in settings:
            precision = item['metrics']['overall']['precision']
            item['precision_drop_points_vs_baseline'] = (
                round((base_precision - precision) * 100, 2)
                if precision is not None and base_precision is not None else None)
            item['within_five_point_precision_budget'] = (
                precision is not None and base_precision is not None and precision >= base_precision - .05)
    report['settings'] = settings
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
