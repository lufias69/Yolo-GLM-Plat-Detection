from django.test import Client, TestCase

from testing.models import Annotation, Photo, TestBatch

SR2 = {
    'detections': [
        {'index': 0, 'box': [10, 10, 210, 90], 'confidence': 0.9, 'ocr_text': 'DR 1282 AS', 'ocr_confidence': 0.8},
        {'index': 1, 'box': [300, 300, 420, 360], 'confidence': 0.6, 'ocr_text': 'B 88 XY', 'ocr_confidence': 0.7},
    ],
    'operational_boxes': [0, 1],
    'latency': {'detection_ms': 10.0, 'ocr_ms': 20.0, 'total_ms': 30.0},
    'threshold': 0.25,
}

VALID_BASE = {'gt_presence': 'ADA', 'detected_ok': 'BENAR', 'ocr_ok': 'BENAR',
              'multi_plate_confirmed': 'on', 'target_detection_index': '0'}


def done_photo(batch, sr=SR2):
    return Photo.objects.create(batch=batch, original_name='f.jpg', status='done', system_result=sr)


class AnnotateViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.batch = TestBatch.objects.create(name='B')
        self.photo = done_photo(self.batch)

    def test_page_renders_warning_and_boxes(self):
        resp = self.client.get(f'/photo/{self.photo.id}/')
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn('2 deteksi', body)
        self.assertIn('DR 1282 AS', body)

    def test_multiple_detections_require_target(self):
        data = {k: v for k, v in VALID_BASE.items() if k != 'target_detection_index'}
        resp = self.client.post(f'/photo/{self.photo.id}/annotate/', data)
        self.assertContains(resp, 'pilih', status_code=200)
        self.assertFalse(Annotation.objects.exists())

    def test_multiple_detections_require_confirmation(self):
        data = {k: v for k, v in VALID_BASE.items() if k != 'multi_plate_confirmed'}
        resp = self.client.post(f'/photo/{self.photo.id}/annotate/', data)
        self.assertFalse(Annotation.objects.exists())

    def test_wrong_ocr_without_gt_text_rejected(self):
        data = dict(VALID_BASE, ocr_ok='SALAH', gt_text='')
        self.client.post(f'/photo/{self.photo.id}/annotate/', data)
        self.assertFalse(Annotation.objects.exists())

    def test_valid_save_normalizes_gt_text(self):
        data = dict(VALID_BASE, ocr_ok='SALAH', gt_text='b 1234 abc', note='blur')
        resp = self.client.post(f'/photo/{self.photo.id}/annotate/', data)
        self.assertEqual(resp.status_code, 302)
        ann = Annotation.objects.get(photo=self.photo)
        self.assertEqual(ann.gt_text, 'B 1234 ABC')
        self.assertEqual(ann.note, 'blur')
        self.assertTrue(ann.is_complete())
        self.assertFalse(ann.stale)

    def test_save_and_next_redirects_to_next_unannotated(self):
        p2 = done_photo(self.batch)
        p3 = done_photo(self.batch)
        Annotation.objects.create(photo=self.photo, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0)
        # annotate p2 with next=1 -> should redirect to p3 (first unannotated after p2)
        resp = self.client.post(f'/photo/{p2.id}/annotate/', dict(VALID_BASE, next='1'))
        self.assertRedirects(resp, f'/photo/{p3.id}/')

    def test_stale_photos_stay_in_queue(self):
        batch = TestBatch.objects.create(name='B2')
        p2 = done_photo(batch)
        Annotation.objects.create(photo=p2, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0, stale=True)
        resp = self.client.get(f'/batch/{batch.id}/next-unannotated/')
        self.assertRedirects(resp, f'/photo/{p2.id}/')

    def test_single_detection_fills_target_automatically(self):
        sr = dict(SR2, detections=[SR2['detections'][0]], operational_boxes=[0])
        photo = done_photo(self.batch, sr=sr)
        data = {'gt_presence': 'ADA', 'detected_ok': 'BENAR', 'ocr_ok': 'BENAR'}
        resp = self.client.post(f'/photo/{photo.id}/annotate/', data)
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Annotation.objects.get(photo=photo).target_detection_index, 0)
