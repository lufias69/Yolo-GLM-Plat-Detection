import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase, override_settings

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
