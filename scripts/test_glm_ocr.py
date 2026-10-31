"""
scripts/test_glm_ocr.py
Bandingkan GLM-OCR (via Ollama) vs PaddleOCR vs EasyOCR pada crop test yang
GT-nya berbentuk plat (>=6 karakter, agar bebas dari noise stiker tahun).
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import requests

from anpr import postprocessing

BASE = Path(__file__).resolve().parent.parent
OLLAMA = 'http://localhost:11434/api/generate'


def glm_ocr(img_path: Path, model='glm-ocr:q8_0'):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    r = requests.post(OLLAMA, json={
        'model': model,
        'prompt': 'Text Recognition:',
        'images': [b64],
        'stream': False,
        'options': {'temperature': 0},
    }, timeout=180)
    r.raise_for_status()
    return r.json()['response'].strip()


def norm(t):
    return postprocessing.normalize_plate_text(t)


def pick_plate_line(response):
    """GLM mereturn semua baris teks; pilih baris pertama yang berbentuk plat."""
    for line in response.splitlines():
        n = norm(line)
        if n and postprocessing.PLATE_REGEX.match(n) and any(c.isalpha() for c in n) \
                and any(c.isdigit() for c in n):
            return n
    return norm(response.splitlines()[0]) if response.strip() else ''


def main():
    rows = json.load(open(BASE / 'results' / 'ocr_e4_paddleocr_details.json', encoding='utf-8'))
    crops = [r for r in rows if len(r['truth']) >= 6]
    print(f'{len(crops)} crop ber-GT-plat diuji\n')

    # engines pembanding
    from anpr.ocr_engines import get_engine, ENGINES, BaseOCREngine
    from anpr import preprocessing
    PP = {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False}
    easy_engine = get_engine('easyocr', gpu=False)
    import cv2 as _cv

    class EasyPick:
        """Sama seperti Paddle3: pilih baris berbentuk plat, bukan conf tertinggi."""

        def recognize(self, image, preprocess=None):
            img = preprocessing.pipeline(image, **(preprocess or {}))
            if len(img.shape) == 2:
                img = _cv.cvtColor(img, _cv.COLOR_GRAY2BGR)
            raw = easy_engine.reader.readtext(img)
            if not raw:
                return '', 0.0
            lines = [t for (_, t, _) in raw]
            scores = [c for (_, _, c) in raw]
            picked = pick_plate_line('\n'.join(lines))
            conf = 0.0
            for t, s in zip(lines, scores):
                if norm(t) == picked:
                    conf = max(conf, s)
            return picked, conf

    easy = EasyPick()

    import cv2 as _cv

    class Paddle3(BaseOCREngine):
        name = 'paddleocr'

        def __init__(self):
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(lang='en', enable_mkldnn=False)

        def recognize(self, image, preprocess=None):
            img = preprocessing.pipeline(image, **(preprocess or {}))
            if len(img.shape) == 2:
                img = _cv.cvtColor(img, _cv.COLOR_GRAY2BGR)
            res = self.ocr.predict(img)
            if not res:
                return '', 0.0
            d = dict(res[0])
            texts = d.get('rec_texts') or []
            scores = d.get('rec_scores') or []
            joined = pick_plate_line(' '.join(texts))
            return joined, (max(scores) if scores else 0.0)

    paddle = Paddle3()

    def norm(t):
        return postprocessing.normalize_plate_text(t)

    stats = {'glm': [0, 0, 0.0], 'paddle': [0, 0, 0.0], 'easy': [0, 0, 0.0]}
    for row in crops:
        p = BASE / 'crops_test' / row['crop_file']
        truth = row['truth']
        t0 = time.time(); g = pick_plate_line(glm_ocr(p)); dt_g = time.time() - t0
        img = cv2.imread(str(p))
        pg, _ = easy.recognize(img, preprocess=PP)
        pp, _ = paddle.recognize(img, preprocess=PP)
        for key, pred in (('glm', g), ('paddle', norm(pp)), ('easy', norm(pg))):
            stats[key][0] += 1
            stats[key][1] += int(pred == truth)
        print(f"GT {truth!r:14} | glm {g!r} ({dt_g:.1f}s) | paddle {norm(pp)!r} | easy {norm(pg)!r}")
    print()
    for key, (n, ok, _) in stats.items():
        print(f'{key:7} exact {ok}/{n} = {ok/n:.1%}')


if __name__ == '__main__':
    main()
