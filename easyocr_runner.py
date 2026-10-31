import sys
sys.path.insert(0, r'C:\project\nuraeni')
import csv
import cv2
from pathlib import Path
from tqdm import tqdm
from anpr.ocr_engines import get_engine
from anpr import postprocessing

base = Path(r'C:/project/nuraeni')
csv_path = base / 'ground_truth_test.csv'
crops_dir = base / 'crops_test'

rows = []
with open(csv_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rows.append(row)

print('Initializing EasyOCR...')
engine = get_engine('easyocr', gpu=True)
preprocess = {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False}

for row in tqdm(rows, desc='OCR'):
    img = cv2.imread(str(crops_dir / row['crop_file']))
    if img is None:
        row['plate_text'] = ''
        row['ocr_confidence'] = 0.0
        row['source'] = 'manual'
        continue
    text, conf = engine.recognize(img, preprocess=preprocess)
    row['plate_text'] = text
    row['ocr_confidence'] = f'{conf:.4f}'
    row['source'] = 'pseudo'

out = base / 'ground_truth_test_pseudo_easyocr.csv'
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print('Saved', out)
