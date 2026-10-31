"""
extract_test_plates.py
Ekstrak crop plat dari test set berdasarkan YOLO label untuk anotasi ground-truth OCR.
Menangani format bbox standar YOLO, Roboflow numbered bbox, dan polygon segmentation.
Output:
  - crops_test/          : folder crop plat (jpg)
  - ground_truth_test.csv: template anotasi
"""
import csv
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
SPLIT = 'test'
IMG_DIR = BASE / SPLIT / 'images'
LBL_DIR = BASE / SPLIT / 'labels'
OUT_DIR = BASE / 'crops_test'
CSV_PATH = BASE / 'ground_truth_test.csv'

PADDING = 0.05  # 5% of plate width/height


def parse_yolo_label(path: Path):
    """Return list of [xc, yc, bw, bh] normalized boxes."""
    boxes = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        nums = [float(x) for x in parts[1:]]
        if len(parts) == 5:
            # standard YOLO bbox: class x y w h
            boxes.append(nums)
        elif len(parts) == 6:
            # Roboflow numbered bbox: index class x y w h
            boxes.append(nums[1:])
        elif len(nums) >= 8 and len(nums) % 2 == 0:
            # Polygon segmentation -> convert to normalized xywh bbox
            xs = nums[0::2]
            ys = nums[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bw = x_max - x_min
            bh = y_max - y_min
            xc = x_min + bw / 2
            yc = y_min + bh / 2
            boxes.append([xc, yc, bw, bh])
        else:
            print(f'  Unrecognized label format ({len(parts)} parts): {line}')
    return boxes


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    img_files = sorted(IMG_DIR.glob('*.jpg'))

    for img_path in img_files:
        lbl_path = LBL_DIR / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Failed to read {img_path}')
            continue
        h, w = img.shape[:2]
        boxes = parse_yolo_label(lbl_path)

        for idx, (xc, yc, bw, bh) in enumerate(boxes):
            # Convert normalized YOLO -> pixel
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            # Add padding
            pad_x = int(bw * w * PADDING)
            pad_y = int(bh * h * PADDING)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f'  Empty crop: {img_path.name} plate{idx}')
                continue

            crop_name = f'{img_path.stem}_plate{idx}.jpg'
            crop_path = OUT_DIR / crop_name
            cv2.imwrite(str(crop_path), crop)

            rows.append({
                'split': SPLIT,
                'image_file': img_path.name,
                'plate_index': idx,
                'crop_file': crop_name,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'plate_text': '',  # TO BE FILLED MANUALLY
                'source': 'manual',
            })

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'split', 'image_file', 'plate_index', 'crop_file',
            'x1', 'y1', 'x2', 'y2', 'plate_text', 'source'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Extracted {len(rows)} plates to {OUT_DIR}')
    print(f'Template saved to {CSV_PATH}')


if __name__ == '__main__':
    main()
