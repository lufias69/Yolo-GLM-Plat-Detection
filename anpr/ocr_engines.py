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


ENGINES = {
    'tesseract': TesseractEngine,
    'easyocr': EasyOCREngine,
    'paddleocr': PaddleOCREngine,
}


def get_engine(name: str, **kwargs) -> BaseOCREngine:
    name = name.lower()
    if name not in ENGINES:
        raise ValueError(f'Unknown OCR engine {name}. Available: {list(ENGINES.keys())}')
    return ENGINES[name](**kwargs)
