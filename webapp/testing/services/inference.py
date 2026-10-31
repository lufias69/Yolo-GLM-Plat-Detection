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

import re

PLATE_SHAPE = re.compile(r'^[A-Z]{1,3}[0-9 ]{1,6}[A-Z ]{0,3}$')


def select_plate_fragment(frags) -> str:
    """EasyOCR returns every text line in the crop — plates often include the
    year sticker ("05 25"), which can score HIGHER than the plate itself.
    Pick the fragment matching Indonesian plate shape (letters first, then
    digits, optional trailing letters); fall back to the highest confidence."""
    def norm(t):
        from anpr.postprocessing import normalize_plate_text
        return normalize_plate_text(t)

    def compact(t):
        return t.replace(' ', '')

    shaped = [fr for fr in frags if PLATE_SHAPE.match(compact(norm(fr['text'])))
              and any(c.isalpha() for c in norm(fr['text']))
              and any(c.isdigit() for c in norm(fr['text']))]
    pool = shaped or frags
    if not pool:
        return ''
    best = max(pool, key=lambda fr: fr['conf'])
    return norm(best['text'])


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
                    name = settings.ANPR['OCR_ENGINE']
                    kw = {'gpu': settings.ANPR['OCR_GPU']} if name == 'easyocr' else {}
                    self._engine = get_engine(name, **kw)
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

    def detect_oriented(self, image_bgr: np.ndarray, model=None,
                        preferred_rotation=None) -> list:
        """Detect on cardinal orientations. Returns candidate list
        [(frame, detections, deg)] sorted best-first.

        The detector also fires on upside-down plates (high det-confidence,
        garbage OCR), so ranking by detection confidence is not enough — the
        caller must OCR the candidates and keep the best OCR confidence.
        preferred_rotation (0/90/180/270) tries only that orientation; used by
        reprocess when the user already confirmed a rotated result."""
        degrees = [preferred_rotation] if preferred_rotation is not None else [0, 90, 180, 270]
        candidates = []
        best_any = None
        for deg in degrees:
            frame = image_bgr if deg == 0 else cv2.rotate(image_bgr, ROTATE_FLAGS[deg])
            dets = self.detect(frame, model=model)
            operational = [d for d in dets if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]
            top = operational[0]['confidence'] if operational else 0.0
            if operational:
                candidates.append((top, frame, dets, deg))
            if best_any is None or (dets and dets[0]['confidence'] > best_any[0]):
                best_any = (dets[0]['confidence'] if dets else 0.0, frame, dets, deg)
        candidates.sort(key=lambda c: -c[0])
        if candidates:
            return [(f, d, deg) for _, f, d, deg in candidates]
        if best_any and best_any[2]:
            _, f, d, deg = best_any
            return [(f, d, deg)]
        return []


    def recognize_crop(self, image_bgr: np.ndarray, box) -> tuple:
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return '', 0.0
        from anpr import preprocessing
        from anpr.postprocessing import normalize_plate_text
        pp = settings.ANPR['OCR_PREPROCESS']
        img = preprocessing.pipeline(crop, **pp)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        engine = self.engine
        reader = getattr(engine, 'reader', None)  # EasyOCR: akses semua fragmen
        if reader is not None:
            raw = reader.readtext(img)
            frags = [{'text': t, 'conf': float(c)} for (_, t, c) in raw]
            text = select_plate_fragment(frags)
            conf = max((f['conf'] for f in frags
                        if normalize_plate_text(f['text']) == text), default=0.0)
            return text, conf
        text, conf = engine.recognize(crop, preprocess=pp)
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

    def process_file(self, path: Path, weights_path: str = None,
                     preferred_rotation=None) -> dict:
        image = cv2.imread(str(path))
        if image is None:
            raise InferenceError(f'Gambar tidak dapat dibaca: {path.name}')
        h, w = image.shape[:2]
        model = self.model_for(weights_path)

        t0 = time.perf_counter()
        candidates = self.detect_oriented(image, model=model,
                                          preferred_rotation=preferred_rotation)
        detection_ms = (time.perf_counter() - t0) * 1000.0

        # OCR box target (conf tertinggi, operasional) tiap kandidat; orientasi
        # dengan confidence OCR terbaik menang (detektor juga fires pada plat
        # terbalik -> conf deteksi bukan penentu orientasi).
        t1 = time.perf_counter()
        best = None  # (ocr_conf, frame, dets, deg, target)
        for frame, dets, deg in candidates:
            operational = [d for d in dets
                           if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]
            if not operational:
                continue
            target = operational[0]
            text, ocr_conf = self.recognize_crop(frame, target['box'])
            if best is None or ocr_conf > best[0]:
                best = (ocr_conf, frame, dets, deg, target, text)
            if ocr_conf >= 0.99:
                break  # sudah jelas terbaca, hentikan pencarian
        ocr_ms = (time.perf_counter() - t1) * 1000.0

        rotation = 0
        if best is not None:
            _, frame, dets, rotation, target, text = best
            detections = []
            for d in dets:
                d = dict(d)
                if d['index'] == target['index']:
                    d['ocr_text'] = text
                    d['ocr_confidence'] = float(best[0])
                else:
                    d['ocr_text'] = ''
                    d['ocr_confidence'] = 0.0
                d['box'] = map_box_to_original(d['box'], rotation, w, h) if rotation else d['box']
                detections.append(d)
            operational = [d['index'] for d in dets
                           if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]
            render_frame = frame
        else:
            if candidates:
                frame, dets, rotation = candidates[0]
            else:
                frame, dets, rotation = image, [], 0
            detections = [dict(d, ocr_text='', ocr_confidence=0.0,
                               box=map_box_to_original(d['box'], rotation, w, h) if rotation else d['box'])
                          for d in dets]
            operational = [d['index'] for d in dets
                           if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]
            render_frame = frame

        rendered = self.render_annotated(image, detections, operational)

        # Semua engine OCR pada crop target (tersimpan permanen di system_result)
        ocr_all = []
        ocr_all_ms = 0.0
        if best is not None:
            t2 = time.perf_counter()
            try:
                ocr_all = self.ocr_all(render_frame, box=target['box'])
            except Exception:
                ocr_all = []
            ocr_all_ms = (time.perf_counter() - t2) * 1000.0
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
            'ocr_engine': settings.ANPR['OCR_ENGINE'],
            'ocr_all': ocr_all,
            'latency': {
                'detection_ms': round(detection_ms, 2),
                'ocr_ms': round(ocr_ms, 2),
                'ocr_all_ms': round(ocr_all_ms, 2),
                'total_ms': round(detection_ms + ocr_ms + ocr_all_ms, 2),
            },
            'threshold': settings.ANPR['CONF_OPERATIONAL'],
            'width': int(w),
            'height': int(h),
            'annotated_image': rel.replace('\\', '/'),
        }


    def ocr_all(self, image_bgr: np.ndarray, box=None) -> list:
        """Run every registered OCR engine on (the crop of) the image and
        return per-engine picks + raw fragments, for side-by-side comparison.
        Stored in system_result so nothing is lost."""
        from anpr.ocr_engines import plate_shape_score
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            image_bgr = image_bgr[y1:y2, x1:x2]
        out = []
        for name in ('easyocr', 'paddleocr', 'tesseract', 'glm-ocr'):
            try:
                if name == 'paddleocr':
                    engine = self._cached_engine(name, enable_mkldnn=False)
                elif name == 'easyocr':
                    engine = self._cached_engine(name, gpu=settings.ANPR['OCR_GPU'])
                else:
                    engine = self._cached_engine(name)
                frags = engine.recognize_fragments(image_bgr, preprocess=settings.ANPR['OCR_PREPROCESS'])
                if frags:
                    text, conf = max(frags, key=lambda f: plate_shape_score(f[0]))
                    if plate_shape_score(text) == 0.0:
                        text, conf = max(frags, key=lambda f: f[1])
                else:
                    text, conf = '', 0.0
                out.append({'engine': name, 'text': text, 'conf': round(float(conf), 3),
                            'fragments': [[t, round(float(c), 3)] for t, c in frags]})
            except Exception as exc:
                out.append({'engine': name, 'text': '', 'conf': 0.0, 'fragments': [],
                            'error': str(exc)[:200]})
        return out

    def _cached_engine(self, name, **kw):
        with self._load_lock:
            key = f'engine:{name}'
            if self._models.get(key) is None:
                from anpr.ocr_engines import get_engine
                self._models[key] = get_engine(name, **kw)
            return self._models[key]


_service = None
_service_lock = threading.Lock()


def get_service() -> InferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InferenceService()
    return _service
