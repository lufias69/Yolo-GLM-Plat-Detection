from django.test import Client, TestCase

from testing.models import (Annotation, EngineJudgment, Photo, TestBatch)

SR = {
    'detections': [{'index': 0, 'box': [1, 2, 30, 20], 'confidence': 0.87,
                    'ocr_text': 'B 1234 ABC', 'ocr_confidence': 0.9}],
    'operational_boxes': [0],
    'latency': {'detection_ms': 1.0, 'ocr_ms': 2.0, 'total_ms': 3.0},
    'threshold': 0.25,
    'ocr_engine': 'easyocr',
    'ocr_all': [
        {'engine': 'easyocr', 'text': 'B 1234 ABC', 'conf': 0.3, 'fragments': []},
        {'engine': 'paddleocr', 'text': 'B 1234 ABD', 'conf': 0.9, 'fragments': []},
        {'engine': 'tesseract', 'text': '', 'conf': 0.0, 'fragments': []},
        {'engine': 'glm-ocr', 'text': 'B 1234 ABC', 'conf': 1.0, 'fragments': []},
    ],
}


def engine_photo(batch):
    return Photo.objects.create(batch=batch, original_name='e.jpg', status='done',
                                system_result=SR)


class EngineJudgmentSaveTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.batch = TestBatch.objects.create(name='B')
        self.photo = engine_photo(self.batch)

    def post(self, **over):
        data = {'gt_presence': 'ADA', 'detected_ok': 'BENAR',
                'eng_ok_easyocr': 'BENAR', 'eng_ok_paddleocr': 'SALAH',
                'eng_ok_tesseract': 'SALAH', 'eng_ok_glm-ocr': 'BENAR',
                'gt_text': 'B 1234 ABC', 'note': ''}
        data.update(over)
        return self.client.post(f'/photo/{self.photo.id}/annotate/', data)

    def test_judgments_saved_and_wrong_chars_computed(self):
        resp = self.post(eng_note_paddleocr='D tertukar')
        self.assertEqual(resp.status_code, 302)
        ann = Annotation.objects.get(photo=self.photo)
        js = {j.engine: j for j in ann.engine_judgments.all()}
        self.assertEqual(len(js), 4)
        self.assertEqual(js['paddleocr'].wrong_chars, 1)   # C->D, auto
        self.assertEqual(js['paddleocr'].note, 'D tertukar')
        self.assertEqual(js['tesseract'].wrong_chars, 10)  # kosong vs GT 10 char
        self.assertEqual(ann.ocr_ok, 'BENAR')  # derived from main engine (easyocr)
        self.assertTrue(ann.is_complete())

    def test_wrong_chars_override_accepted(self):
        self.post(eng_wrong_paddleocr='2')
        j = EngineJudgment.objects.get(engine='paddleocr')
        self.assertEqual(j.wrong_chars, 2)

    def test_missing_engine_judgment_rejected(self):
        data = {k: v for k, v in {
            'gt_presence': 'ADA', 'detected_ok': 'BENAR',
            'eng_ok_easyocr': 'BENAR', 'eng_ok_paddleocr': 'SALAH',
            'eng_ok_glm-ocr': 'BENAR', 'gt_text': 'B 1234 ABC'}.items()}
        # tesseract sengaja tidak dinilai
        resp = self.client.post(f'/photo/{self.photo.id}/annotate/', data)
        self.assertFalse(Annotation.objects.exists())
        self.assertIn('tesseract', resp.content.decode())

    def test_gt_text_required_when_any_salah(self):
        resp = self.post(gt_text='')
        self.assertFalse(Annotation.objects.exists())


class EngineMetricsTest(TestCase):
    def test_per_engine_exact_and_char_accuracy(self):
        from testing.services import metrics
        batch = TestBatch.objects.create(name='M')
        photo = engine_photo(batch)
        ann = Annotation.objects.create(photo=photo, gt_presence='ADA',
                                        detected_ok='BENAR', ocr_ok='BENAR',
                                        gt_text='B 1234 ABC', target_detection_index=0)
        EngineJudgment.objects.create(annotation=ann, engine='easyocr', ok='BENAR')
        EngineJudgment.objects.create(annotation=ann, engine='paddleocr', ok='SALAH',
                                      wrong_chars=1)
        em = metrics.engine_metrics([photo])
        self.assertEqual(em['easyocr']['n'], 1)
        self.assertAlmostEqual(em['easyocr']['exact'], 1.0)
        self.assertAlmostEqual(em['paddleocr']['exact'], 0.0)
        self.assertAlmostEqual(em['paddleocr']['char_accuracy'], 0.9)  # 1 - 1/10
