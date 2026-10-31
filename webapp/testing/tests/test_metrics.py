import sys
from pathlib import Path

from django.test import TestCase

from testing.models import Annotation, Photo, TestBatch
from testing.services import metrics


def sr(conf=0.9, ocr='DR 1282 AS', boxes=None, ops=None):
    detections = [{'index': 0, 'box': [1, 2, 30, 20], 'confidence': conf,
                   'ocr_text': ocr, 'ocr_confidence': 0.8}]
    if boxes is not None:
        detections = boxes
    return {'detections': detections,
            'operational_boxes': [0] if ops is None else ops,
            'latency': {'detection_ms': 10.0, 'ocr_ms': 5.0, 'total_ms': 15.0},
            'threshold': 0.25}


class FixtureMixin:
    def make(self, batch, name, system, **ann_kwargs):
        p = Photo.objects.create(batch=batch, original_name=name, status='done',
                                 system_result=system)
        if ann_kwargs:
            Annotation.objects.create(photo=p, **ann_kwargs)
        return p


class DetectionMetricsTest(FixtureMixin, TestCase):
    def test_confusion_matrix_and_f1(self):
        batch = TestBatch.objects.create(name='B')
        self.make(batch, 'tp1', sr(), gt_presence='ADA', detected_ok='BENAR', ocr_ok='BENAR',
                  target_detection_index=0)
        self.make(batch, 'tp2', sr(conf=0.7), gt_presence='ADA', detected_ok='BENAR',
                  ocr_ok='BENAR', target_detection_index=0)
        self.make(batch, 'fn', sr(boxes=[], ops=[]), gt_presence='ADA', detected_ok='GAGAL',
                  ocr_ok='TAK_TERDETEKSI')
        self.make(batch, 'fp', sr(conf=0.6), gt_presence='TIDAK_ADA', detected_ok='BENAR',
                  ocr_ok='TIDAK_RELEVAN')
        self.make(batch, 'tn', sr(boxes=[], ops=[]), gt_presence='TIDAK_ADA',
                  detected_ok='TIDAK_RELEVAN', ocr_ok='TIDAK_RELEVAN')
        # noise: unannotated + TIDAK_RELEVAN with box + stale
        self.make(batch, 'un', sr())
        self.make(batch, 'irr', sr(), gt_presence='ADA', detected_ok='TIDAK_RELEVAN',
                  ocr_ok='BENAR', target_detection_index=0)
        stale = self.make(batch, 'stale', sr(), gt_presence='ADA', detected_ok='BENAR',
                          ocr_ok='BENAR', target_detection_index=0, stale=True)
        photos = list(batch.photos.all())
        m = metrics.detection_metrics(photos)
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (2, 1, 1, 1))
        self.assertAlmostEqual(m['accuracy'], 0.6)
        self.assertAlmostEqual(m['precision'], 2 / 3)
        self.assertAlmostEqual(m['recall'], 2 / 3)
        self.assertAlmostEqual(m['f1'], 2 / 3)

    def test_empty_batch_returns_none(self):
        batch = TestBatch.objects.create(name='B')
        m = metrics.detection_metrics(list(batch.photos.all()))
        self.assertIsNone(m['accuracy'])
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['tn']), (0, 0, 0, 0))


class RocTest(FixtureMixin, TestCase):
    def test_auc_between_point_five_and_one(self):
        batch = TestBatch.objects.create(name='B')
        for conf in (0.9, 0.7, 0.5):
            self.make(batch, f'p{conf}', sr(conf=conf), gt_presence='ADA', detected_ok='BENAR',
                      ocr_ok='BENAR', target_detection_index=0)
        self.make(batch, 'fp', sr(conf=0.6), gt_presence='TIDAK_ADA', detected_ok='BENAR',
                  ocr_ok='TIDAK_RELEVAN')
        self.make(batch, 'tn', sr(boxes=[], ops=[]), gt_presence='TIDAK_ADA',
                  detected_ok='TIDAK_RELEVAN', ocr_ok='TIDAK_RELEVAN')
        self.make(batch, 'fn', sr(boxes=[], ops=[]), gt_presence='ADA', detected_ok='GAGAL',
                  ocr_ok='TAK_TERDETEKSI')
        roc = metrics.roc_metrics(list(batch.photos.all()))
        self.assertIsNotNone(roc['auc'])
        self.assertGreater(roc['auc'], 0.5)
        self.assertLessEqual(roc['auc'], 1.0)
        pts = roc['points']
        fprs = [p['fpr'] for p in pts]
        self.assertEqual(fprs, sorted(fprs))

    def test_auc_none_without_negatives(self):
        batch = TestBatch.objects.create(name='B')
        self.make(batch, 'p', sr(), gt_presence='ADA', detected_ok='BENAR',
                  ocr_ok='BENAR', target_detection_index=0)
        roc = metrics.roc_metrics(list(batch.photos.all()))
        self.assertIsNone(roc['auc'])


class OcrMetricsTest(FixtureMixin, TestCase):
    def test_exact_match_char_accuracy_and_cer(self):
        batch = TestBatch.objects.create(name='B')
        gt = 'B 1234 ABC'
        self.make(batch, 'ok1', sr(ocr=gt), gt_presence='ADA', detected_ok='BENAR',
                  ocr_ok='BENAR', target_detection_index=0)
        self.make(batch, 'ok2', sr(ocr=gt), gt_presence='ADA', detected_ok='BENAR',
                  ocr_ok='BENAR', target_detection_index=0)
        self.make(batch, 'bad', sr(ocr='B 1234 ABD'), gt_presence='ADA', detected_ok='BENAR',
                  ocr_ok='SALAH', gt_text=gt, target_detection_index=0)
        o = metrics.ocr_metrics(list(batch.photos.all()))
        self.assertEqual(o['n_evaluated'], 3)
        self.assertAlmostEqual(o['exact_match'], 2 / 3)
        self.assertAlmostEqual(o['cer_mean'], (0.0 + 0.0 + 1 / 10) / 3)
        self.assertAlmostEqual(o['char_accuracy'], (1 + 1 + 9 / 10) / 3)

    def test_empty(self):
        batch = TestBatch.objects.create(name='B')
        o = metrics.ocr_metrics(list(batch.photos.all()))
        self.assertIsNone(o['exact_match'])
        self.assertEqual(o['n_evaluated'], 0)


class LatencyTest(FixtureMixin, TestCase):
    def test_mean_median_p95(self):
        batch = TestBatch.objects.create(name='B')
        for i in range(20):
            s = sr()
            s['latency']['total_ms'] = float(i + 1)
            Photo.objects.create(batch=batch, original_name=f'l{i}', status='done', system_result=s)
        lat = metrics.latency_metrics(list(batch.photos.all()))
        self.assertAlmostEqual(lat['mean_ms'], 10.5)
        self.assertAlmostEqual(lat['median_ms'], 10.5)
        self.assertAlmostEqual(lat['p95_ms'], 19.05)
        self.assertAlmostEqual(lat['fps'], 1000 / 10.5)


class BatchReportTest(TestCase):
    def test_report_counts_and_keys(self):
        batch = TestBatch.objects.create(name='B')
        Photo.objects.create(batch=batch, original_name='x', status='queued')
        rep = metrics.batch_report(batch)
        self.assertEqual(rep['counts']['total'], 1)
        self.assertEqual(rep['counts']['unannotated'], 1)
        self.assertIsNone(rep['detection']['accuracy'])
        for key in ('detection', 'roc', 'ocr', 'latency', 'counts', 'outcomes'):
            self.assertIn(key, rep)
