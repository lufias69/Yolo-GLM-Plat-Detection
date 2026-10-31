import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from django.test import SimpleTestCase, TestCase, override_settings

from anpr.postprocessing import normalize_plate_text
from testing.services import inference
from testing.services.inference import InferenceError, InferenceService

TMP_MEDIA = tempfile.mkdtemp()


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeBoxes:
    def __init__(self, boxes, confs):
        self.xyxy = FakeTensor(boxes)
        self.conf = FakeTensor(confs)

    def __len__(self):
        return len(self.xyxy._arr)


class FakeResult:
    def __init__(self, boxes, confs):
        self.boxes = FakeBoxes(boxes, confs)


class FakeYOLO:
    def __init__(self, boxes, confs):
        self._out = [FakeResult(boxes, confs)]
        self.last_conf = None

    def __call__(self, image, verbose=False, conf=0.25):
        self.last_conf = conf
        return self._out


class FakeEngine:
    def recognize(self, image, preprocess=None):
        return 'b 1234 abc', 0.91


def stub_service(boxes=None, confs=None):
    svc = InferenceService.__new__(InferenceService)
    svc._model = FakeYOLO(
        boxes if boxes is not None else [[10, 10, 210, 90], [5, 5, 30, 15]],
        confs if confs is not None else [0.5, 0.05],
    )
    svc._engine = FakeEngine()
    return svc


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class ProcessFileTest(TestCase):
    def setUp(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        fd, self.path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        cv2.imwrite(self.path, img)
        inference._service = stub_service()
        self.addCleanup(setattr, inference, '_service', None)

    def tearDown(self):
        Path(self.path).unlink(missing_ok=True)

    def test_result_schema_filtering_and_ocr(self):
        res = inference.get_service().process_file(Path(self.path))
        self.assertEqual(res['threshold'], 0.25)
        dets = res['detections']
        self.assertEqual(len(dets), 2)
        self.assertEqual([d['confidence'] for d in dets], [0.5, 0.05])
        self.assertEqual([d['index'] for d in dets], [0, 1])
        self.assertEqual(res['operational_boxes'], [0])
        self.assertEqual(dets[0]['ocr_text'], normalize_plate_text('b 1234 abc'))
        self.assertEqual(dets[1]['ocr_text'], '')
        self.assertAlmostEqual(dets[0]['ocr_confidence'], 0.91)
        self.assertGreater(res['latency']['total_ms'], 0)
        self.assertTrue((Path(TMP_MEDIA) / res['annotated_image']).exists())
        self.assertEqual((res['width'], res['height']), (320, 240))

    def test_stored_at_low_threshold(self):
        svc = inference.get_service()
        svc.process_file(Path(self.path))
        self.assertEqual(svc._model.last_conf, 0.01)

    def test_missing_file_raises_inference_error(self):
        with self.assertRaises(InferenceError):
            inference.get_service().process_file(Path(self.path + '.nope'))

    def test_corrupt_image_raises_inference_error(self):
        fd, junk = tempfile.mkstemp(suffix='.jpg')
        with os.fdopen(fd, 'wb') as f:
            f.write(b'not an image')
        try:
            with self.assertRaises(InferenceError):
                inference.get_service().process_file(Path(junk))
        finally:
            Path(junk).unlink(missing_ok=True)


class ThreadSafetyTest(unittest.TestCase):
    def test_get_service_singleton_thread_safe(self):
        original = inference.InferenceService
        created = []

        class Counting(original):
            def __init__(self):
                super().__init__()
                time.sleep(0.01)
                created.append(1)

        inference.InferenceService = Counting
        old = inference._service
        inference._service = None
        services = []

        def grab():
            services.append(inference.get_service())

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        try:
            self.assertEqual(len(created), 1)
            self.assertTrue(all(s is services[0] for s in services))
        finally:
            inference.InferenceService = original
            inference._service = old


class OrientedModel:
    """Fake: hanya mendeteksi pada gambar portrait (hasil rotasi dari landscape)."""

    def __init__(self, need_portrait=True):
        self.need_portrait = need_portrait
        self.calls = 0

    def __call__(self, image, verbose=False, conf=0.25):
        self.calls += 1
        h, w = image.shape[:2]
        hit = (h > w) if self.need_portrait else True
        boxes = [[10, 10, 100, 50]] if hit else [[0, 0, 0, 0]]
        confs = [0.9] if hit else [0.0]
        return [FakeResult(boxes if hit else [], confs if hit else [])]


class RotationFallbackTest(TestCase):
    def setUp(self):
        img = np.zeros((200, 400, 3), dtype=np.uint8)  # landscape
        fd, self.path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        cv2.imwrite(self.path, img)
        self.addCleanup(lambda: Path(self.path).unlink(missing_ok=True))

    def _svc(self, fake_model):
        svc = InferenceService.__new__(InferenceService)
        svc._model = fake_model
        svc._engine = FakeEngine()
        return svc

    @override_settings(MEDIA_ROOT=TMP_MEDIA)
    def test_rotated_photo_detected_via_fallback(self):
        inference._service = self._svc(OrientedModel())
        self.addCleanup(setattr, inference, '_service', None)
        res = inference.get_service().process_file(Path(self.path))
        self.assertEqual(res['rotation'], 90)
        self.assertEqual(res['operational_boxes'], [0])
        box = res['detections'][0]['box']
        # box dipetakan ke koordinat gambar asli (landscape 400x200)
        self.assertLessEqual(box[2], 400)
        self.assertLessEqual(box[3], 200)
        self.assertGreaterEqual(box[1], 100)  # hasil inverse-rotasi jatuh di sisi bawah-kiri
        self.assertGreater(inference.get_service()._model.calls, 1)

    @override_settings(MEDIA_ROOT=TMP_MEDIA)
    def test_upright_photo_prefers_rotation_zero(self):
        class AnyModel(OrientedModel):
            def __call__(self, image, verbose=False, conf=0.25):
                self.calls += 1
                return [FakeResult([[10, 10, 100, 50]], [0.9])]

        m = AnyModel(True)
        inference._service = self._svc(m)
        self.addCleanup(setattr, inference, '_service', None)
        res = inference.get_service().process_file(Path(self.path))
        self.assertEqual(res['rotation'], 0)  # seri conf -> orientasi awal menang
        self.assertEqual(m.calls, 4)  # keempat orientasi selalu dicoba (pilih terbaik)


class MapBoxTest(SimpleTestCase):
    def test_roundtrip_corners(self):
        from testing.services.inference import map_box_to_original
        import cv2
        W, H = 400, 200
        box = [10, 20, 110, 70]
        img = np.zeros((H, W, 3), np.uint8)
        for deg, flag in [(90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180),
                          (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            rot = cv2.rotate(img, flag)
            # titik tengah box di frame rotasi harus sama dengan tengah box terpetakan di asli
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            # mapping forward: box asli -> frame rotasi via map_box_to_original kebalikan
            ob = map_box_to_original(box, deg, W, H)  # box(di frame)->asli; untuk forward pakai deg invers
            self.assertTrue(all(v >= 0 for v in ob))
            self.assertLessEqual(ob[2], W)
            self.assertLessEqual(ob[3], H)
