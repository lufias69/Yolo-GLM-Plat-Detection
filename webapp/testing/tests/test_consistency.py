from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.test import Client, TestCase
from PIL import Image

from testing.models import Annotation, ConsistencyRun, Photo, TestBatch
from testing.services import consistency

SR = {
    'detections': [{'index': 0, 'box': [10, 10, 60, 40], 'confidence': 0.9,
                    'ocr_text': 'DR 1282 AS', 'ocr_confidence': 0.8}],
    'operational_boxes': [0],
    'latency': {'detection_ms': 10.0, 'ocr_ms': 5.0, 'total_ms': 15.0},
    'threshold': 0.25,
}


class StubService:
    """detect: always one box IoU=1.0 with baseline, conf 0.9.
    OCR: alternates texts on even/odd iterations so distribution is checkable."""

    def detect(self, image):
        return [{'index': 0, 'box': [10, 10, 60, 40], 'confidence': 0.9}]

    def recognize_crop(self, image, box):
        return ('DR 1282 AS' if self.i % 4 else 'DR 1282 A5'), 0.8

    i = 0


class ExecuteTest(TestCase):
    def setUp(self):
        self.batch = TestBatch.objects.create(name='B')
        buf = BytesIO()
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(buf, format='JPEG')
        from django.core.files.base import ContentFile
        self.photo = Photo.objects.create(batch=self.batch, original_name='f.jpg',
                                          status='done', system_result=SR)
        self.photo.image.save('photos/f.jpg', ContentFile(buf.getvalue()), save=True)
        Annotation.objects.create(photo=self.photo, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0)
        self.run = ConsistencyRun.objects.create(photo=self.photo, n=20, seed=5)

    def _execute_with_counter(self, cancel_at=None):
        state = {'i': 0}

        class CountingService(StubService):
            def recognize_crop(self, image, box):
                self.i = state['i']
                out = super().recognize_crop(image, box)
                state['i'] += 1
                if cancel_at is not None and state['i'] == cancel_at:
                    self.run_ref.cancel_requested = True
                    self.run_ref.save(update_fields=['cancel_requested'])
                return out

        svc = CountingService()
        svc.run_ref = self.run
        with patch.object(consistency, 'get_service', lambda: svc):
            with patch.object(consistency, 'perturb', lambda img, rng: img):
                consistency.execute(self.run)

    def test_summary_counts(self):
        self._execute_with_counter()
        self.run.refresh_from_db()
        s = self.run.summary
        self.assertEqual(self.run.status, 'done')
        self.assertEqual(len(self.run.runs), 20)
        self.assertEqual(s['detected_x_of_n'], '20/20')
        self.assertEqual(s['same_as_baseline_x_of_n'], '15/20')
        self.assertEqual(s['same_as_gt_x_of_n'], '15/20')
        self.assertAlmostEqual(s['conf_mean'], 0.9)
        self.assertEqual(sum(s['text_distribution'].values()), 20)

    def test_incremental_progress_saved(self):
        counts = []
        orig_save = ConsistencyRun.save

        def spy(self, *a, **k):
            counts.append(len(self.runs))
            return orig_save(self, *a, **k)
        with patch.object(ConsistencyRun, 'save', spy):
            self._execute_with_counter()
        self.assertIn(10, counts)

    def test_cancel_stops_early(self):
        self._execute_with_counter(cancel_at=5)
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, 'cancelled')
        self.assertEqual(len(self.run.runs), 5)


class ConsistencyViewsTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.batch = TestBatch.objects.create(name='B')
        self.photo = Photo.objects.create(batch=self.batch, original_name='f.jpg',
                                          status='done', system_result=SR)
        self.ann = Annotation.objects.create(photo=self.photo, gt_presence='ADA',
                                             detected_ok='BENAR', ocr_ok='BENAR',
                                             target_detection_index=0)

    def test_guard_requires_complete_annotation(self):
        p = Photo.objects.create(batch=self.batch, original_name='g.jpg', status='done',
                                 system_result=SR)
        resp = self.client.post(f'/photo/{p.id}/consistency/start/', {'n': 100})
        self.assertIn(resp.status_code, (200, 302, 403))
        self.assertEqual(ConsistencyRun.objects.count(), 0)

    def test_guard_rejects_stale_annotation(self):
        self.ann.stale = True
        self.ann.save()
        resp = self.client.post(f'/photo/{self.photo.id}/consistency/start/', {'n': 100})
        self.assertIn(resp.status_code, (200, 302, 403))
        self.assertEqual(ConsistencyRun.objects.count(), 0)

    def test_start_creates_queued_run(self):
        with patch.object(consistency, 'start_run', lambda run: None):
            resp = self.client.post(f'/photo/{self.photo.id}/consistency/start/', {'n': 50})
        self.assertEqual(resp.status_code, 302)
        run = ConsistencyRun.objects.get(photo=self.photo)
        self.assertEqual(run.n, 50)
        self.assertEqual(run.status, 'queued')

    def test_cancel_sets_flag(self):
        with patch.object(consistency, 'start_run', lambda run: None):
            self.client.post(f'/photo/{self.photo.id}/consistency/start/', {'n': 10})
        run = ConsistencyRun.objects.get(photo=self.photo)
        resp = self.client.post(f'/photo/{self.photo.id}/consistency/{run.id}/cancel/')
        self.assertEqual(resp.status_code, 200)
        run.refresh_from_db()
        self.assertTrue(run.cancel_requested)

    def test_status_endpoint(self):
        run = ConsistencyRun.objects.create(photo=self.photo, n=10, status='running')
        run.runs = [{'i': 0}]
        run.save()
        resp = self.client.get(f'/photo/{self.photo.id}/consistency/{run.id}/status/')
        data = resp.json()
        self.assertEqual(data['status'], 'running')
        self.assertEqual(data['done_iterations'], 1)
