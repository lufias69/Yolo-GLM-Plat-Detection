from django.test import Client, TestCase

from testing.models import Annotation, EngineJudgment, Photo, TestBatch


class OcrAnalysisViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.batch = TestBatch.objects.create(name='B')

    def test_page_renders_empty_batch(self):
        resp = self.client.get(f'/batch/{self.batch.id}/report/ocr/analysis/')
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn('Analisis Similarity OCR', body)
        self.assertIn('Belum ada penilaian', body)

    def test_page_renders_with_judgments(self):
        p = Photo.objects.create(batch=self.batch, original_name='a.jpg', status='done',
                                 system_result={'ocr_all': [{'engine': 'easyocr', 'text': 'B 1234 ABC'}]})
        ann = Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                        ocr_ok='BENAR', gt_text='B 1234 ABC')
        EngineJudgment.objects.create(annotation=ann, engine='easyocr', ok='BENAR',
                                      sim_nospasi=100, sim_spasi=100)
        resp = self.client.get(f'/batch/{self.batch.id}/report/ocr/analysis/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('easyocr', resp.content.decode())
        self.assertIn('100', resp.content.decode())

    def test_page_shows_extended_sections(self):
        p = Photo.objects.create(
            batch=self.batch, original_name='a.jpg', status='done',
            system_result={'ocr_all': [
                {'engine': 'easyocr', 'text': 'B 1234 ABC', 'conf': 0.9},
                {'engine': 'paddleocr', 'text': 'B 1284 ABC', 'conf': 0.95}]})
        ann = Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                        ocr_ok='SALAH', gt_text='B 1234 ABC')
        EngineJudgment.objects.create(annotation=ann, engine='easyocr', ok='BENAR',
                                      sim_nospasi=100, sim_spasi=100)
        EngineJudgment.objects.create(annotation=ann, engine='paddleocr', ok='SALAH',
                                      sim_nospasi=86, sim_spasi=85)
        resp = self.client.get(f'/batch/{self.batch.id}/report/ocr/analysis/')
        body = resp.content.decode()
        self.assertEqual(resp.status_code, 200)
        for marker in ['Analisis 1', 'Analisis 2', 'Analisis 3', 'Analisis 4',
                       'Analisis 5', 'Analisis 6', 'Analisis 7',
                       'exactChart', 'confChart']:
            self.assertIn(marker, body)
        # alpha heatmap harus format titik (bukan lokal) agar rgba() valid
        self.assertIn('rgba(220,53,69,1.000000)', body)
