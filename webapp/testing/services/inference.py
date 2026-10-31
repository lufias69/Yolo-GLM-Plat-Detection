"""
testing/services/inference.py
YOLO detection + EasyOCR on uploaded real-world photos.

Config matches the research scripts exactly (spec 7.6):
EasyOCR on CPU, YOLO stored at conf=0.01, operational threshold 0.25,
CLAHE+upscale OCR preprocessing.
"""
import time
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings


class InferenceError(Exception):
    """Raised when an image cannot be read or processed."""


class InferenceService:
    def __init__(self):
        self._model = None
        self._engine = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(settings.ANPR['WEIGHTS'])
        return self._model

    @property
    def engine(self):
        if self._engine is None:
            from anpr.ocr_engines import get_engine
            self._engine = get_engine(settings.ANPR['OCR_ENGINE'], gpu=settings.ANPR['OCR_GPU'])
        return self._engine

    def detect(self, image_bgr: np.ndarray) -> list:
        """Run YOLO at CONF_STORED; list of {'index','box','confidence'} desc by confidence."""
        results = self.model(image_bgr, verbose=False, conf=settings.ANPR['CONF_STORED'])
        boxes = results[0].boxes
        out = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            pairs = sorted(zip(xyxy, confs), key=lambda p: -float(p[1]))
            for i, (b, c) in enumerate(pairs):
                out.append({'index': i, 'box': [int(b[0]), int(b[1]), int(b[2]), int(b[3])],
                            'confidence': float(c)})
        return out

    def recognize_crop(self, image_bgr: np.ndarray, box) -> tuple:
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return '', 0.0
        from anpr.postprocessing import normalize_plate_text
        text, conf = self.engine.recognize(crop, preprocess=settings.ANPR['OCR_PREPROCESS'])
        return normalize_plate_text(text), float(conf)

    def render_annotated(self, image_bgr: np.ndarray, detections: list, operational: list) -> np.ndarray:
        out = image_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = d['box']
            is_op = d['index'] in operational
            color = (0, 200, 0) if is_op else (160, 160, 160)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"#{d['index']} {d['confidence']:.2f}"
            if is_op and d.get('ocr_text'):
                label += f" {d['ocr_text']}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        return out

    def process_file(self, path: Path) -> dict:
        image = cv2.imread(str(path))
        if image is None:
            raise InferenceError(f'Gambar tidak dapat dibaca: {path.name}')
        h, w = image.shape[:2]

        t0 = time.perf_counter()
        detections = self.detect(image)
        detection_ms = (time.perf_counter() - t0) * 1000.0

        operational = [d['index'] for d in detections
                       if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]

        t1 = time.perf_counter()
        for d in detections:
            if d['index'] in operational:
                text, conf = self.recognize_crop(image, d['box'])
                d['ocr_text'] = text
                d['ocr_confidence'] = conf
            else:
                d['ocr_text'] = ''
                d['ocr_confidence'] = 0.0
        ocr_ms = (time.perf_counter() - t1) * 1000.0

        rendered = self.render_annotated(image, detections, operational)
        import hashlib
        digest = hashlib.md5(str(path).encode()).hexdigest()[:10]
        rel = f'annotated/{int(time.time() * 1000)}_{digest}.jpg'
        out_path = Path(settings.MEDIA_ROOT) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), rendered)

        return {
            'detections': detections,
            'operational_boxes': operational,
            'latency': {
                'detection_ms': round(detection_ms, 2),
                'ocr_ms': round(ocr_ms, 2),
                'total_ms': round(detection_ms + ocr_ms, 2),
            },
            'threshold': settings.ANPR['CONF_OPERATIONAL'],
            'width': int(w),
            'height': int(h),
            'annotated_image': rel.replace('\\', '/'),
        }


_service = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service
