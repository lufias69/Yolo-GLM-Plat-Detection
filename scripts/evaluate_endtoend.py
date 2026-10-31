"""
scripts/evaluate_endtoend.py
End-to-end evaluation: YOLO detection + OCR on test images.
Matches predicted plates to ground-truth plates by IoU, then compares OCR text.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

from anpr.ocr_engines import get_engine
from anpr import postprocessing


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(csv_path: Path):
    by_image = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_image.setdefault(row['image_file'], []).append({
                'plate_index': int(row['plate_index']),
                'box': [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
                'text': postprocessing.clean_text(row['plate_text']),
            })
    return by_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--csv', default='ground_truth_test.csv')
    parser.add_argument('--engine', default='easyocr')
    parser.add_argument('--output', default='results/end_to_end_metrics.json')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    weights = Path(args.weights)
    csv_path = base / args.csv
    img_dir = base / 'test' / 'images'

    gt = load_ground_truth(csv_path)
    model = YOLO(str(weights))
    if args.engine == 'paddleocr':
        engine = get_engine(args.engine)
    else:
        engine = get_engine(args.engine, gpu=False)

    preprocess = {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False}

    total_images = 0
    images_all_correct = 0
    total_gt_plates = 0
    total_detected_correct = 0
    total_text_correct = 0
    inference_times = []

    for img_file, gt_plates in tqdm(gt.items(), desc='End-to-end eval'):
        img_path = img_dir / img_file
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        t0 = time.time()
        results = model(img, verbose=False, conf=args.conf)
        dt = time.time() - t0
        inference_times.append(dt)

        preds = results[0].boxes
        if preds is None or len(preds) == 0:
            total_images += 1
            continue

        pred_boxes = preds.xyxy.cpu().numpy().astype(int)
        pred_confs = preds.conf.cpu().numpy()

        matched_truth = [False] * len(gt_plates)
        text_correct_for_image = []

        for pb, conf in zip(pred_boxes, pred_confs):
            x1, y1, x2, y2 = pb
            best_iou = 0.0
            best_idx = -1
            for i, g in enumerate(gt_plates):
                val = iou([x1, y1, x2, y2], g['box'])
                if val > best_iou:
                    best_iou = val
                    best_idx = i

            if best_iou >= args.iou_threshold and not matched_truth[best_idx]:
                matched_truth[best_idx] = True
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    text_correct_for_image.append(False)
                    continue
                pred_text, _ = engine.recognize(crop, preprocess=preprocess)
                pred_text = postprocessing.clean_text(pred_text)
                truth_text = gt_plates[best_idx]['text']
                correct = pred_text == truth_text
                text_correct_for_image.append(correct)

        total_images += 1
        total_gt_plates += len(gt_plates)
        total_detected_correct += sum(matched_truth)
        total_text_correct += sum(text_correct_for_image)

        if len(gt_plates) > 0 and all(matched_truth) and len(text_correct_for_image) == len(gt_plates) and all(text_correct_for_image):
            images_all_correct += 1

    metrics = {
        'weights': str(weights),
        'ocr_engine': args.engine,
        'n_images': total_images,
        'image_level_accuracy': images_all_correct / total_images if total_images else 0.0,
        'detection_recall': total_detected_correct / total_gt_plates if total_gt_plates else 0.0,
        'end_to_end_plate_accuracy': total_text_correct / total_gt_plates if total_gt_plates else 0.0,
        'mean_inference_seconds': float(np.mean(inference_times)) if inference_times else 0.0,
        'fps': total_images / sum(inference_times) if inference_times else 0.0,
    }

    out = base / args.output
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
