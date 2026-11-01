import json
from unittest.mock import patch
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from django.test import Client, TestCase, override_settings

from testing.models import Annotation, Photo, TestBatch
from testing.services import workers
from testing.services.inference import InferenceError

TMP_MEDIA = tempfile.mkdtemp()

RESULT = {
    'detections': [{'index': 0, 'box': [1, 1, 10, 10], 'confidence': 0.9,
                    'ocr_text': 'B 1 ABC', 'ocr_confidence': 0.8}],
    'operational_boxes': [0],
    'latency': {'detection_ms': 5.0, 'ocr_ms': 7.0, 'total_ms': 12.0},
    'threshold': 0.25, 'width': 160, 'height': 120,
    'annotated_image': 'annotated/x.jpg',
}


def make_photo(batch):
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    p = Path(TMP_MEDIA) / 'photos' / 'test.jpg'
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return Photo.objects.create(batch=batch, image='photos/test.jpg', original_name='test.jpg')


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class WorkerTest(TestCase):
    def setUp(self):
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)
        Path(TMP_MEDIA).mkdir(parents=True, exist_ok=True)
        self.batch = TestBatch.objects.create(name='B')

    def test_process_photo_done(self):
        photo = make_photo(self.batch)
        workers._process_fn = lambda path: dict(RESULT)
        self.addCleanup(setattr, workers, '_process_fn', None)
        workers._process_photo(photo.id)
        photo.refresh_from_db()
        self.assertEqual(photo.status, 'done')
        self.assertEqual(photo.result_version, 1)
        self.assertEqual((photo.width, photo.height), (160, 120))
        self.assertEqual(photo.system_result['operational_boxes'], [0])
        self.assertEqual(photo.annotated_image.name, 'annotated/x.jpg')
        self.assertIsNotNone(photo.processed_at)

    def test_process_photo_error(self):
        photo = make_photo(self.batch)

        def boom(path):
            raise InferenceError('Gambar tidak dapat dibaca')
        workers._process_fn = boom
        self.addCleanup(setattr, workers, '_process_fn', None)
        workers._process_photo(photo.id)
        photo.refresh_from_db()
        self.assertEqual(photo.status, 'error')
        self.assertIn('tidak dapat dibaca', photo.error_message)

    def test_enqueue_puts_id_on_queue(self):
        photo = make_photo(self.batch)
        # stub so the real daemon thread (if it grabs the item first) never loads models
        workers._process_fn = lambda path: dict(RESULT)
        self.addCleanup(setattr, workers, '_process_fn', None)
        workers.enqueue_photo(photo.id)
        self.assertIn(photo.id, list(workers._QUEUE.queue))

    def test_reprocess_bumps_version_and_marks_stale(self):
        photo = make_photo(self.batch)
        Annotation.objects.create(photo=photo, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0)
        photo.reprocess()
        photo.refresh_from_db()
        self.assertEqual(photo.result_version, 2)
        self.assertEqual(photo.status, 'queued')
        photo.annotation.refresh_from_db()
        self.assertTrue(photo.annotation.stale)
        workers._process_fn = lambda path: dict(RESULT)
        self.addCleanup(setattr, workers, '_process_fn', None)
        workers._process_photo(photo.id)
        photo.refresh_from_db()
        self.assertEqual(photo.result_version, 2)
        self.assertEqual(photo.status, 'done')


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class PhotoStatusViewTest(TestCase):
    def test_status_endpoint(self):
        batch = TestBatch.objects.create(name='B')
        photo = Photo.objects.create(batch=batch, original_name='x.jpg', status='processing')
        response = Client().get(f'/api/photo/{photo.id}/status/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'processing')


class WorkerSaveRobustnessTest(TestCase):
    def test_final_save_failure_marks_error_not_stuck_processing(self):
        batch = TestBatch.objects.create(name='B')
        photo = make_photo(batch)
        workers._process_fn = lambda path: dict(RESULT)
        self.addCleanup(setattr, workers, '_process_fn', None)
        real_save = Photo.save

        def flaky_save(self, *a, **k):
            if not a and 'update_fields' not in k:  # the final full save
                raise Exception('database is locked')
            return real_save(self, *a, **k)

        with patch.object(Photo, 'save', flaky_save):
            workers._process_photo(photo.id)
        fresh = Photo.objects.get(pk=photo.id)
        self.assertEqual(fresh.status, 'error')
        self.assertIn('database is locked', fresh.error_message)


class WorkerMissingFileTest(TestCase):
    @override_settings(MEDIA_ROOT=TMP_MEDIA)
    def test_missing_file_gives_real_error_not_attribute_error(self):
        batch = TestBatch.objects.create(name='B')
        photo = Photo.objects.create(batch=batch, image='photos/tiada.jpg',
                                     original_name='tiada.jpg')
        workers._process_photo(photo.id)
        photo.refresh_from_db()
        self.assertEqual(photo.status, 'error')
        self.assertIn('tidak dapat dibaca', photo.error_message)
        self.assertNotIn("has no attribute 'name'", photo.error_message)
