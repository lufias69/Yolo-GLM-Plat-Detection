import io
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from PIL import Image

RESULT = {
    'detections': [{'index': 0, 'box': [1, 2, 30, 20], 'confidence': 0.87,
                    'ocr_text': 'DR 1282 AS', 'ocr_confidence': 0.91}],
    'operational_boxes': [0],
    'latency': {'detection_ms': 11.0, 'ocr_ms': 45.0, 'total_ms': 56.0},
    'threshold': 0.25, 'width': 100, 'height': 80,
    'annotated_image': 'annotated/quick_x.jpg',
}


def jpg():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((80, 100, 3), dtype=np.uint8)).save(buf, format='JPEG')
    buf.seek(0)
    return SimpleUploadedFile('satu.jpg', buf.read(), content_type='image/jpeg')


class QuickTestViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_page_shows_form(self):
        resp = self.client.get('/uji-cepat/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('Uji cepat', resp.content.decode())
        self.assertIn('name="image"', resp.content.decode())

    @patch('testing.views.get_service')
    def test_post_runs_and_shows_result(self, gs):
        svc = gs.return_value
        svc.process_file.return_value = dict(RESULT)
        resp = self.client.post('/uji-cepat/', {'image': jpg()}, follow=True)
        body = resp.content.decode()
        self.assertIn('DR 1282 AS', body)
        self.assertIn('0,87', body)
        svc.process_file.assert_called_once()

    @patch('testing.views.get_service')
    def test_post_rejects_non_image(self, gs):
        resp = self.client.post('/uji-cepat/', {'image': SimpleUploadedFile('a.pdf', b'%PDF')},
                                follow=True)
        self.assertNotIn('DR 1282 AS', resp.content.decode())
        gs.assert_not_called()

    @patch('testing.views.get_service')
    def test_no_database_rows_created(self, gs):
        from testing.models import Photo
        svc = gs.return_value
        svc.process_file.return_value = dict(RESULT)
        self.client.post('/uji-cepat/', {'image': jpg()})
        self.assertEqual(Photo.objects.count(), 0)
