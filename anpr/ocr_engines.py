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


class TesseractEngine(BaseOCREngine):
    name = 'tesseract'

    def __init__(self, config: str = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        import pytesseract
        self.pytesseract = pytesseract
        self.config = config

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

    def recognize(self, image: np.ndarray, preprocess: Optional[dict] = None) -> Tuple[str, float]:
        import cv2
        img = preprocessing.pipeline(image, **(preprocess or {}))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        results = self.reader.readtext(img)
        if not results:
            return '', 0.0
        # results: [(bbox, text, confidence), ...]
        best = max(results, key=lambda r: r[2])
        text, conf = best[1], best[2]
        return postprocessing.normalize_plate_text(text), conf


class PaddleOCREngine(BaseOCREngine):
    name = 'paddleocr'

    def __init__(self, lang: str = 'en'):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang=lang)

    def recognize(self, image: np.ndarray, preprocess: Optional[dict] = None) -> Tuple[str, float]:
        import cv2
        import tempfile
        img = preprocessing.pipeline(image, **(preprocess or {}))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # PaddleOCR expects a file path
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)
        result = self.ocr.predict(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        if result is None:
            return '', 0.0
        # result may be a list of dicts; extract text and score
        lines = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    rec_text = item.get('rec_text') or item.get('text', '')
                    rec_score = item.get('rec_score', 0.0) or item.get('score', 0.0)
                    lines.append((rec_text, rec_score))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    lines.append(item)
        if not lines:
            return '', 0.0
        best = max(lines, key=lambda r: r[1])
        text, conf = best[0], best[1]
        return postprocessing.normalize_plate_text(text), conf


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
