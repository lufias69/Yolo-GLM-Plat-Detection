"""
scripts/inference_single_image.py
Run ANPR inference on a single image:
  - Detect license plate(s)
  - Read text via OCR
  - Report: detected/not detected, bounding box, plate text, character count,
    format validity, OCR confidence.
Usage:
    venv/Scripts/python scripts/inference_single_image.py --image path/to/image.jpg --weights runs/detect/plate_detector_yolov8n/weights/best.pt --output output.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from anpr.ocr_engines import get_engine
from anpr import postprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--weights', default='runs/detect/plate_detector_yolov8n/weights/best.pt')
    parser.add_argument('--engine', default='easyocr', choices=['easyocr', 'paddleocr', 'tesseract'])
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--output', default='output_inference.jpg')
    parser.add_argument('--save-crops', action='store_true')
    args = parser.parse_args()

    img_path = Path(args.image)
    weights = Path(args.weights)
    if not img_path.exists():
        print(f'Image not found: {img_path}')
        return
    if not weights.exists():
        print(f'Weights not found: {weights}')
        return

    # Load models
    print(f'Loading YOLO detector: {weights}')
    detector = YOLO(str(weights))
    print(f'Loading OCR engine: {args.engine}')
    if args.engine == 'paddleocr':
        engine = get_engine(args.engine)
    else:
        engine = get_engine(args.engine, gpu=False)

    # Preprocessing config for OCR
    preprocess = {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False}

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'Failed to read image: {img_path}')
        return

    # Detect plates
    results = detector(img, verbose=False, conf=args.conf)
    boxes = results[0].boxes

    output_dir = Path('inference_outputs')
    if args.save_crops:
        output_dir.mkdir(exist_ok=True)

    detections = []
    if boxes is None or len(boxes) == 0:
        print('\nResult: NO license plate detected.')
    else:
        print(f'\nResult: {len(boxes)} license plate(s) detected.')
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for idx, (box, conf) in enumerate(zip(xyxy, confs), start=1):
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # OCR
            text, ocr_conf = engine.recognize(crop, preprocess=preprocess)
            text = postprocessing.clean_text(text)
            is_valid = postprocessing.is_valid_format(text)
            n_chars = len(text.replace(' ', ''))

            detections.append({
                'index': idx,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'plate_text': text,
                'char_count': n_chars,
                'format_valid': bool(is_valid),
                'ocr_confidence': float(ocr_conf),
            })

            # Draw bounding box and text on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{idx}: {text} ({n_chars} chars) conf={conf:.2f}'
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            if args.save_crops:
                crop_path = output_dir / f'{img_path.stem}_plate{idx}.jpg'
                cv2.imwrite(str(crop_path), crop)
                print(f'  Saved crop: {crop_path}')

    # Save annotated output
    out_path = Path(args.output)
    cv2.imwrite(str(out_path), img)
    print(f'\nAnnotated image saved to: {out_path}')

    # Print summary table
    if detections:
        print('\nDetection summary:')
        print(f'  {"#":<4} {"Plate text":<20} {"Chars":<6} {"Valid":<6} {"Det conf":<10} {"OCR conf":<10}')
        for d in detections:
            print(f'  {d["index"]:<4} {d["plate_text"]:<20} {d["char_count"]:<6} {"yes" if d["format_valid"] else "no":<6} {d["confidence"]:<10.3f} {d["ocr_confidence"]:<10.3f}')
    else:
        print('  (no detections)')


if __name__ == '__main__':
    main()
