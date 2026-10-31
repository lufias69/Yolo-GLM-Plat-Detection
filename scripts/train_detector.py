"""
train_detector.py
Train YOLO detector for single-class plate detection.
Usage:
    venv/Scripts/python scripts/train_detector.py --model yolov8n --epochs 200 --data data.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--data', default='data.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', default='plate_detector')
    parser.add_argument('--device', default='0', help='Device to use: cpu, 0, 0,1, etc.')
    parser.add_argument('--workers', type=int, default=0, help='Data loader workers')
    parser.add_argument('--cache', action='store_true', help='Cache images in RAM')
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        name=args.name,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        exist_ok=True,
        verbose=True,
    )

    metrics = results.results_dict
    print('Training complete.')
    print(metrics)

    out = Path('results') / f'training_log_{args.name}.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'model': args.model,
            'epochs': args.epochs,
            'seed': args.seed,
            'metrics': metrics,
            'best_weights': str(Path(results.save_dir) / 'weights' / 'best.pt'),
        }, f, indent=2)
    print(f'Log saved to {out}')


if __name__ == '__main__':
    main()
