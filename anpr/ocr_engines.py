"""
anpr/ocr_engines.py
Unified wrappers for OCR engines used in the ANPR pipeline.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from anpr import preprocessing, postprocessing


class BaseOCREngine(ABC):
    name: str = 'base'

    @abstractmethod
    def recognize(self, image: np.ndarray, preprocess: Optional[dict] = None) -> Tuple[str, float]:
        """Return (normalized_text, confidence)."""
        raise NotImplementedError

    def recognize_fragments(self, image: np.ndarray,
                            preprocess: Optional[dict] = None) -> list:
        """All text fragments as (normalized_text, confidence). Default: the
        single recognize() result as a one-element list."""
        return [self.recognize(image, preprocess)]


class TesseractEngine(BaseOCREngine):
    name = 'tesseract'

    def __init__(self, config: str = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        import pytesseract
        self._locate_binary(pytesseract)
        self.pytesseract = pytesseract
        self.config = config

    @staticmethod
    def _locate_binary(pytesseract):
        """Point pytesseract at a known local install when not on PATH."""
        if pytesseract.pytesseract.tesseract_cmd != 'tesseract':
            return
        from pathlib import Path
        candidates = [
            Path.home() / 'Tools' / 'tesseract' / 'tesseract.exe',
            Path('C:/Program Files/Tesseract-OCR/tesseract.exe'),
        ]
        for c in candidates:
            if c.exists():
                import os
                pytesseract.pytesseract.tesseract_cmd = str(c)
                os.environ.setdefault('TESSDATA_PREFIX', str(c.parent / 'tessdata'))
                return

    def recognize(self, image: np.ndarray, preprocess: Optional[dict] = None) -> Tuple[str, float]:
        img = preprocessing.pipeline(image, **(preprocess or {}))
        # Tesseract expects grayscale/RGB numpy array
        text = self.pytesseract.image_to_string(img, config=self.config).strip()
        return postprocessing.normalize_plate_text(text), 1.0


class EasyOCREngine(BaseOCREngine):
    name = 'easyocr'

    def __init__(self, languages: Optional[list] = None, gpu: bool = False):
        import easyocr
        self.reader = easyocr.Reader(languages or ['en'], gpu=gpu)

    def recognize_fragments(self, image: np.ndarray, preprocess: Optional[dict] = None) -> list:
        import cv2
        img = preprocessing.pipeline(image, **(preprocess or {}))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(img)
        return [(postprocessing.normalize_plate_text(r[1]), float(r[2])) for r in results]

    def recognize(self, image: np.ndarray, preprocess: Optional[dict] = None) -> Tuple[str, float]:
        frags = self.recognize_fragments(image, preprocess)
        if not frags:
            return '', 0.0
        # legacy behavior: single highest-confidence fragment
        return max(frags, key=lambda f: f[1])


class PaddleOCREngine(BaseOCREngine):
    name = 'paddleocr'

    def __init__(self, lang: str = 'en', enable_mkldnn: bool = True):
        from paddleocr import PaddleOCR
        kw = {'lang': lang}
        if not enable_mkldnn:
            kw['enable_mkldnn'] = False
        self.ocr = PaddleOCR(**kw)

    def recognize_fragments(self, image: np.ndarray, preprocess: Optional[dict] = None) -> list:
        import cv2
        img = preprocessing.pipeline(image, **(preprocess or {}))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result = self.ocr.predict(img)
        lines = []
        for item in result or []:
            d = dict(item)
            texts = d.get('rec_texts')          # PaddleOCR 3.x
            scores = d.get('rec_scores')
            if texts is not None:
                lines += [(t, float(s)) for t, s in zip(texts, scores or [])]
            else:                                # API lama
                t = d.get('rec_text') or d.get('text', '')
                if t:
                    lines.append((t, float(d.get('rec_score') or d.get('score') or 0.0)))
        return [(postprocessing.normalize_plate_text(t), c) for t, c in lines if t and t.strip()]

    def recognize(self, image: np.ndarray, preprocess: Optional[dict] = None) -> Tuple[str, float]:
        frags = self.recognize_fragments(image, preprocess)
        if not frags:
            return '', 0.0
        # legacy behavior: single highest-confidence fragment
        return max(frags, key=lambda f: f[1])


def plate_shape_score(text: str) -> float:
    """Heuristic 0..1: how much `text` looks like an Indonesian plate
    (letters -> digits -> optional letters). Used as a confidence proxy for
    engines that return no score (GLM-OCR)."""
    t = postprocessing.normalize_plate_text(text)
    if not t:
        return 0.0
    compact = t.replace(' ', '')
    if not (any(c.isalpha() for c in compact) and any(c.isdigit() for c in compact)):
        return 0.0
    score = 1.0 if postprocessing.PLATE_REGEX.match(compact) else 0.4
    if 6 <= len(compact) <= 10:
        score += 0.3
    elif len(compact) < 4 or len(compact) > 12:
        score -= 0.3
    return max(0.0, min(1.0, score))


class GLMOCREngine(BaseOCREngine):
    """GLM-OCR via local Ollama (model glm-ocr:q8_0). Vision-language OCR:
    reads the whole crop, returns all text lines; the plate-shaped line wins.
    No native confidence -> plate_shape_score is used as the score."""
    name = 'glm-ocr'

    def __init__(self, model: str = 'glm-ocr:q8_0', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url.rstrip('/')

    def recognize(self, image, preprocess=None):
        import base64
        import cv2
        import requests
        import tempfile
        from anpr import preprocessing
        img = preprocessing.pipeline(image, **(preprocess or {}))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            return '', 0.0
        r = requests.post(self.base_url + '/api/generate', json={
            'model': self.model,
            'prompt': 'Text Recognition:',
            'images': [base64.b64encode(buf.tobytes()).decode()],
            'stream': False,
            'options': {'temperature': 0},
        }, timeout=120)
        r.raise_for_status()
        response = r.json().get('response', '')
        best, best_score = '', 0.0
        for line in response.splitlines():
            t = postprocessing.normalize_plate_text(line)
            s = plate_shape_score(t)
            if s > best_score:
                best, best_score = t, s
        if not best and response.strip():
            best = postprocessing.normalize_plate_text(response.splitlines()[0])
        return best, best_score


ENGINES = {
    'tesseract': TesseractEngine,
    'easyocr': EasyOCREngine,
    'paddleocr': PaddleOCREngine,
    'glm-ocr': GLMOCREngine,
}


def get_engine(name: str, **kwargs) -> BaseOCREngine:
    name = name.lower()
    if name not in ENGINES:
        raise ValueError(f'Unknown OCR engine {name}. Available: {list(ENGINES.keys())}')
    return ENGINES[name](**kwargs)
