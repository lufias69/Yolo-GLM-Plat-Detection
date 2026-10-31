"""
scripts/normalize_labels.py
Convert all label files to standard YOLO detection format.
Handles:
  - standard YOLO bbox: class xc yc w h
  - Roboflow numbered bbox: idx class xc yc w h
  - polygon segmentation: class x1 y1 x2 y2 ... -> converted to bbox
All class ids forced to 0 (single-class plate).
"""
import argparse
import shutil
from pathlib import Path


def normalize_label_file(src: Path, dst: Path):
    boxes = []
    for line in src.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        nums = [float(x) for x in parts[1:]]
        if len(parts) == 5:
            # class xc yc w h
            xc, yc, w, h = nums
        elif len(parts) == 6:
            # idx class xc yc w h
            xc, yc, w, h = nums[1:]
        elif len(nums) >= 8 and len(nums) % 2 == 0:
            # polygon -> bbox
            xs = nums[0::2]
            ys = nums[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            xc = x_min + w / 2
            yc = y_min + h / 2
        else:
            print(f'  Skipping unrecognized line in {src}: {line}')
            continue
        # Clamp to [0, 1]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        if w > 0 and h > 0:
            boxes.append(f'0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join(boxes) + ('\n' if boxes else ''))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='C:/project/nuraeni')
    parser.add_argument('--clean-cache', action='store_true')
    args = parser.parse_args()

    base = Path(args.base)
    for split in ['train', 'valid', 'test']:
        lbl_dir = base / split / 'labels'
        for src in sorted(lbl_dir.glob('*.txt')):
            normalize_label_file(src, src)
        print(f'{split}: normalized {len(list(lbl_dir.glob("*.txt")))} label files')

    if args.clean_cache:
        for cache in base.rglob('*.cache'):
            print(f'Removing cache: {cache}')
            cache.unlink()
        # Also remove previous training run to avoid confusion
        runs_dir = base / 'runs'
        if runs_dir.exists():
            shutil.rmtree(runs_dir)
            print(f'Removed {runs_dir}')


if __name__ == '__main__':
    main()
