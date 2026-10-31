"""
scripts/evaluate_detector.py
Validate trained YOLO detector on test split and save metrics.
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/detect/plate_detector_yolov8n/weights/best.pt')
    parser.add_argument('--data', default='data.yaml')
    parser.add_argument('--output', default='results/detector_test_metrics.json')
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f'Weights not found: {weights}')
        return

    model = YOLO(str(weights))
    start = time.time()
    metrics = model.val(data=args.data, split='test', imgsz=640)
    elapsed = time.time() - start

    results = {
        'weights': str(weights),
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'val_time_seconds': elapsed,
    }

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
