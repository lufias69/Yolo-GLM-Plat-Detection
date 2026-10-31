"""
scripts/generate_ocr_pseudolabels.py
Generate pseudo OCR labels for test set crops using a chosen OCR engine.
Usage:
    venv/Scripts/python scripts/generate_ocr_pseudolabels.py --engine easyocr
"""
import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is on path so anpr package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix Windows cp1252 encoding for EasyOCR model download progress bar
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
from tqdm import tqdm

from anpr.ocr_engines import get_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='easyocr', choices=['easyocr', 'paddleocr', 'tesseract'])
    parser.add_argument('--csv', default='ground_truth_test.csv')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    csv_path = base / args.csv
    crops_dir = base / 'crops_test'

    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f'Initializing {args.engine}...')
    if args.engine == 'paddleocr':
        engine = get_engine(args.engine)
    else:
        engine = get_engine(args.engine, gpu=args.gpu)

    preprocess = {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False}

    for row in tqdm(rows, desc='OCR pseudo-labels'):
        crop_path = crops_dir / row['crop_file']
        img = cv2.imread(str(crop_path))
        if img is None:
            row['plate_text'] = ''
            row['ocr_confidence'] = 0.0
            row['source'] = 'manual'
            continue
        text, conf = engine.recognize(img, preprocess=preprocess)
        row['plate_text'] = text
        row['ocr_confidence'] = f'{conf:.4f}'
        row['source'] = 'pseudo'

    out_path = base / f'ground_truth_test_pseudo_{args.engine}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved pseudo labels to {out_path}')


if __name__ == '__main__':
    main()
