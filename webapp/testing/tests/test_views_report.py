from django.test import Client, TestCase

from testing.models import Annotation, Photo, TestBatch


def sr(conf=0.9, ocr='DR 1282 AS', ops=None):
    return {'detections': [{'index': 0, 'box': [1, 2, 30, 20], 'confidence': conf,
                           'ocr_text': ocr, 'ocr_confidence': 0.8}],
            'operational_boxes': [0] if ops is None else ops,
            'latency': {'detection_ms': 10.0, 'ocr_ms': 5.0, 'total_ms': 15.0},
            'threshold': 0.25}


class ReportViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.batch = TestBatch.objects.create(name='B')
        p = Photo.objects.create(batch=self.batch, original_name='tp.jpg',
                                 status='done', system_result=sr())
        Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0)
        p2 = Photo.objects.create(batch=self.batch, original_name='fn.jpg', status='done',
                                  system_result=sr(ops=[]))
        Annotation.objects.create(photo=p2, gt_presence='ADA', detected_ok='GAGAL',
                                  ocr_ok='TAK_TERDETEKSI')
        p3 = Photo.objects.create(batch=self.batch, original_name='fp.jpg', status='done',
                                  system_result=sr(conf=0.5))
        Annotation.objects.create(photo=p3, gt_presence='TIDAK_ADA', detected_ok='BENAR',
                                  ocr_ok='TIDAK_RELEVAN', target_detection_index=0)
        p4 = Photo.objects.create(batch=self.batch, original_name='tn.jpg', status='done',
                                  system_result=sr(ops=[]))
        Annotation.objects.create(photo=p4, gt_presence='TIDAK_ADA', detected_ok='TIDAK_RELEVAN',
                                  ocr_ok='TIDAK_RELEVAN')
        Photo.objects.create(batch=self.batch, original_name='unrated.jpg',
                             status='done', system_result=sr())

    def test_report_shows_metrics_and_counts(self):
        resp = self.client.get(f'/batch/{self.batch.id}/report/')
        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn('Laporan', body)
        self.assertIn('tp.jpg', body)
        self.assertIn('50,00', body)  # accuracy 2/4 -> id-style decimal comma
        self.assertIn('belum dinilai', body)
        self.assertIn('data-report="roc-points"', body)

    def test_empty_batch_report_ok(self):
        empty = TestBatch.objects.create(name='E')
        resp = self.client.get(f'/batch/{empty.id}/report/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('belum dinilai', resp.content.decode())

    def test_links_to_ocr_report_and_analysis(self):
        body = self.client.get(f'/batch/{self.batch.id}/report/').content.decode()
        self.assertIn(f'/batch/{self.batch.id}/report/ocr/', body)
        self.assertIn(f'/batch/{self.batch.id}/report/ocr/analysis/', body)

    def test_ocr_card_names_main_engine(self):
        body = self.client.get(f'/batch/{self.batch.id}/report/').content.decode()
        self.assertIn('engine utama', body)

    def test_end_to_end_metric_shown(self):
        # tp.jpg: TP + ocr_ok BENAR -> e2e 1/1 = 100%
        body = self.client.get(f'/batch/{self.batch.id}/report/').content.decode()
        self.assertIn('End-to-end', body)
        self.assertIn('100,00', body)

    def test_photo_column_shows_thumbnail_preview(self):
        p = Photo.objects.get(original_name='tp.jpg')
        p.annotated_image = 'annotated/tp.jpg'
        p.save(update_fields=['annotated_image'])
        body = self.client.get(f'/batch/{self.batch.id}/report/').content.decode()
        self.assertIn('/media/annotated/tp.jpg', body)
        self.assertIn('rekap-thumb', body)
