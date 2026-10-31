"""
scripts/run_e4_engine.py
Eksperimen E4: bandingkan engine OCR (easyocr/paddleocr/tesseract) memakai
fungsi evaluate() yang sama seperti evaluate_ocr.py, pada crop test dengan
ground-truth pseudo-label. Output: results/ocr_metrics_<engine>.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = Path(__file__).resolve().parent.parent


def load_eval_script():
    spec = importlib.util.spec_from_file_location('evaluate_ocr', BASE / 'scripts' / 'evaluate_ocr.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True, choices=['easyocr', 'paddleocr', 'tesseract'])
    parser.add_argument('--csv', default='ground_truth_test_pseudo_easyocr.csv')
    args = parser.parse_args()

    if args.engine == 'tesseract':
        import pytesseract
        exe = Path.home() / 'Tools' / 'tesseract' / 'tesseract.exe'
        if exe.exists():
            pytesseract.pytesseract.tesseract_cmd = str(exe)
            import os
            os.environ['TESSDATA_PREFIX'] = str(exe.parent / 'tessdata')
        # TesseractEngine tidak menerima kwarg gpu yang dikirim evaluate()
        import anpr.ocr_engines as oe
        _orig = oe.get_engine

        def get_engine(name, **kw):
            if name == 'tesseract':
                kw.pop('gpu', None)
            return _orig(name, **kw)
        oe.get_engine = get_engine

    ev = load_eval_script()
    if args.engine in ('paddleocr', 'easyocr'):
        # Seleksi fragmen yang SAMA untuk semua engine (adil untuk E4):
        # gabungkan window fragmen berurutan yang cocok format plat
        # (huruf->angka->huruf); jika tidak ada, ambil fragmen conf tertinggi.
        import anpr.ocr_engines as oe
        import cv2
        from anpr import postprocessing

        PLATE = postprocessing.PLATE_REGEX

        def select_plate_from_fragments(texts, scores):
            frags = [(postprocessing.normalize_plate_text(t), s)
                     for t, s in zip(texts, scores) if t and t.strip()]
            if not frags:
                return '', 0.0
            best = None
            n = len(frags)
            for i in range(n):
                for j in range(i, n):
                    joined = ' '.join(f[0] for f in frags[i:j + 1])
                    joined = postprocessing.normalize_plate_text(joined)
                    if PLATE.match(joined) and any(c.isalpha() for c in joined) \
                            and any(c.isdigit() for c in joined):
                        score = min(f[1] for f in frags[i:j + 1])
                        if best is None or score > best[1]:
                            best = (joined, score)
            if best:
                return best
            t, s = max(frags, key=lambda f: f[1])
            return t, s

        if args.engine == 'paddleocr':
            class Paddle3(oe.BaseOCREngine):
                name = 'paddleocr'

                def __init__(self, lang='en'):
                    from paddleocr import PaddleOCR
                    self.ocr = PaddleOCR(lang=lang, enable_mkldnn=False)

                def recognize(self, image, preprocess=None):
                    from anpr import preprocessing
                    img = preprocessing.pipeline(image, **(preprocess or {}))
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    res = self.ocr.predict(img)
                    if not res:
                        return '', 0.0
                    d = dict(res[0])
                    return select_plate_from_fragments(d.get('rec_texts') or [],
                                                       d.get('rec_scores') or [])
            oe.ENGINES['paddleocr'] = Paddle3
        else:
            class EasyFrag(oe.EasyOCREngine):
                def recognize(self, image, preprocess=None):
                    from anpr import preprocessing
                    img = preprocessing.pipeline(image, **(preprocess or {}))
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    raw = self.reader.readtext(img)
                    texts = [t for (_, t, _) in raw]
                    scores = [c for (_, _, c) in raw]
                    return select_plate_from_fragments(texts, scores)
            oe.ENGINES['easyocr'] = EasyFrag
    csv_path = BASE / args.csv
    crops_dir = BASE / 'crops_test'
    ablations = [
        {'use_clahe': False, 'use_upscale': False, 'use_binarize': False, 'use_denoise': False},
        {'use_clahe': True, 'use_upscale': False, 'use_binarize': False, 'use_denoise': False},
        {'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False},
        {'use_clahe': True, 'use_upscale': True, 'use_binarize': True, 'use_denoise': False},
    ]
    summary = []
    for prep in ablations:
        res = ev.evaluate(args.engine, csv_path, crops_dir, prep)
        detail = res.pop('results')
        summary.append(res)
        best = max(summary, key=lambda r: r['exact_match'])
        print(f"[{args.engine}] clahe={prep['use_clahe']} up={prep['use_upscale']} "
              f"bin={prep['use_binarize']} -> exact {res['exact_match']:.2%} cer {res['cer_mean']:.4f}")
        Path(BASE / 'results' / f'ocr_e4_{args.engine}_details.json').write_text(
            json.dumps(detail, indent=1), encoding='utf-8')
    out = BASE / 'results' / f'ocr_metrics_{args.engine}.json'
    out.write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print('BEST', args.engine, json.dumps(best))


if __name__ == '__main__':
    main()
