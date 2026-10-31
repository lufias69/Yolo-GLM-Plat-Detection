"""
scripts/evaluate_ocr.py
Evaluate OCR engine on test-set crops using a ground-truth CSV.
Reports exact-match accuracy, CER, and optional ablation over preprocessing.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

from anpr.ocr_engines import get_engine
from anpr import postprocessing


def cer(pred: str, truth: str) -> float:
    """Character Error Rate (Levenshtein distance normalized by truth length)."""
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    m, n = len(truth), len(pred)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if truth[i - 1] == pred[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[m, n] / m


def load_ground_truth(csv_path: Path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def evaluate(engine_name: str, csv_path: Path, crops_dir: Path, preprocess: dict):
    rows = load_ground_truth(csv_path)
    if engine_name == 'paddleocr':
        engine = get_engine(engine_name)
    else:
        engine = get_engine(engine_name, gpu=False)

    exact = 0
    total_cer = 0.0
    results = []
    for row in tqdm(rows, desc=f'Eval {engine_name}'):
        crop_path = crops_dir / row['crop_file']
        img = cv2.imread(str(crop_path))
        truth = postprocessing.clean_text(row['plate_text'])
        if img is None or not truth:
            continue
        pred, conf = engine.recognize(img, preprocess=preprocess)
        pred = postprocessing.clean_text(pred)
        distance = cer(pred, truth)
        total_cer += distance
        if pred == truth:
            exact += 1
        results.append({
            'crop_file': row['crop_file'],
            'truth': truth,
            'pred': pred,
            'cer': distance,
            'confidence': conf,
        })
    n = len(results)
    return {
        'engine': engine_name,
        'preprocess': preprocess,
        'n': n,
        'exact_match': exact / n if n else 0.0,
        'cer_mean': total_cer / n if n else 0.0,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ground_truth_test.csv')
    parser.add_argument('--engine', default='easyocr', choices=['easyocr', 'paddleocr', 'tesseract'])
    parser.add_argument('--output', default='results/ocr_metrics.json')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    csv_path = base / args.csv
    crops_dir = base / 'crops_test'
    out_path = base / args.output
    out_path.parent.mkdir(exist_ok=True)

    ablations = [
        {'use_clahe': False, 'use_upscale': False, 'use_binarize': False, 'use_denoise': False},
        {'use_clahe': True, 'use_upscale': False, 'use_binarize': False, 'use_denoise': False},
        {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False},
        {'use_clahe': True, 'use_upscale': True, 'use_binarize': True, 'use_denoise': False},
    ]

    summary = []
    for prep in ablations:
        res = evaluate(args.engine, csv_path, crops_dir, prep)
        summary.append({
            'engine': res['engine'],
            'preprocess': res['preprocess'],
            'n': res['n'],
            'exact_match': res['exact_match'],
            'cer_mean': res['cer_mean'],
        })
        print(f"Preprocess {prep}: exact={res['exact_match']:.3%} CER={res['cer_mean']:.3f}")

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved OCR metrics to {out_path}')


if __name__ == '__main__':
    main()
