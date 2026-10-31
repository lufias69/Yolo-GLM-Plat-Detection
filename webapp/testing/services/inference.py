"""
testing/services/inference.py
YOLO detection + EasyOCR on uploaded real-world photos.

Config matches the research scripts exactly (spec 7.6):
EasyOCR on CPU, YOLO stored at conf=0.01, operational threshold 0.25,
CLAHE+upscale OCR preprocessing.
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings


class InferenceError(Exception):
    """Raised when an image cannot be read or processed."""


ROTATE_FLAGS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def map_box_to_original(box, rotation_deg, orig_w, orig_h):
    """Map a detection box made on the rotated frame back to original coords.

    rotation_deg is the clockwise rotation applied to the ORIGINAL to obtain
    the frame the box lives in (0/90/180/270)."""
    x1, y1, x2, y2 = box
    if rotation_deg == 0:
        return [x1, y1, x2, y2]
    if rotation_deg == 90:      # frame W'=H, H'=W
        return [y1, orig_h - x2, y2, orig_h - x1]
    if rotation_deg == 180:
        return [orig_w - x2, orig_h - y2, orig_w - x1, orig_h - y1]
    # 270 cw (=90 ccw): frame W'=H, H'=W
    return [orig_w - y2, x1, orig_w - y1, x2]


class InferenceService:
    def __init__(self):
        self._model = None
        self._engine = None
        self._load_lock = threading.Lock()
        self._models = {}

    @property
    def model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from ultralytics import YOLO
                    self._model = YOLO(settings.ANPR['WEIGHTS'])
        return self._model

    def model_for(self, path=None):
        """Cached YOLO handle per weights path (thread-safe). None/default = active settings model."""
        if not path or path == settings.ANPR['WEIGHTS']:
            return self.model
        cached = self._models.get(path)
        if cached is None:
            with self._load_lock:
                cached = self._models.get(path)
                if cached is None:
                    from ultralytics import YOLO
                    cached = YOLO(path)
                    self._models[path] = cached
        return cached

    @property
    def engine(self):
        if self._engine is None:
            with self._load_lock:
                if self._engine is None:
                    from anpr.ocr_engines import get_engine
                    self._engine = get_engine(settings.ANPR['OCR_ENGINE'], gpu=settings.ANPR['OCR_GPU'])
        return self._engine

    def detect(self, image_bgr: np.ndarray, model=None) -> list:
        """Run YOLO at CONF_STORED; list of {'index','box','confidence'} desc by confidence."""
        results = (model or self.model)(image_bgr, verbose=False, conf=settings.ANPR['CONF_STORED'])
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

    def detect_oriented(self, image_bgr: np.ndarray, model=None) -> tuple:
        """Detect on all 4 cardinal orientations (0/90/180/270 cw) and keep the
        best one. The detector also fires on upside-down plates but OCR cannot
        read those, so we pick the orientation with the highest top confidence
        rather than the first hit. Returns (frame, detections_in_frame, deg)."""
        best = None
        for deg in (0, 90, 180, 270):
            frame = image_bgr if deg == 0 else cv2.rotate(image_bgr, ROTATE_FLAGS[deg])
            dets = self.detect(frame, model=model)
            top = dets[0]['confidence'] if dets else 0.0
            operational = top >= settings.ANPR['CONF_OPERATIONAL']
            cand = (operational, top, frame, dets, deg)
            if best is None or cand[:2] > best[:2]:
                best = cand
        _, _, frame, dets, deg = best
        return frame, dets, deg


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

    def process_file(self, path: Path, weights_path: str = None) -> dict:
        image = cv2.imread(str(path))
        if image is None:
            raise InferenceError(f'Gambar tidak dapat dibaca: {path.name}')
        h, w = image.shape[:2]

        t0 = time.perf_counter()
        frame, detections, rotation = self.detect_oriented(image, model=self.model_for(weights_path))
        detection_ms = (time.perf_counter() - t0) * 1000.0

        operational = [d['index'] for d in detections
                       if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]

        t1 = time.perf_counter()
        for d in detections:
            if d['index'] in operational:
                # OCR pada frame terorientasi (plat tegak), bukan frame asli
                text, conf = self.recognize_crop(frame, d['box'])
                d['ocr_text'] = text
                d['ocr_confidence'] = conf
            else:
                d['ocr_text'] = ''
                d['ocr_confidence'] = 0.0
        ocr_ms = (time.perf_counter() - t1) * 1000.0

        if rotation:
            detections = [dict(d, box=map_box_to_original(d['box'], rotation, w, h))
                          for d in detections]

        rendered = self.render_annotated(image, detections, operational)
        import hashlib
        digest = hashlib.md5(str(path).encode()).hexdigest()[:10]
        # stable name: reprocess overwrites instead of leaking files per run
        rel = f'annotated/{digest}.jpg'
        out_path = Path(settings.MEDIA_ROOT) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), rendered)

        return {
            'detections': detections,
            'operational_boxes': operational,
            'rotation': int(rotation),
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
_service_lock = threading.Lock()


def get_service() -> InferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InferenceService()
    return _service
