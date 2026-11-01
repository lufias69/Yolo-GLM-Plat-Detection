from django.test import TestCase

from testing.models import Annotation, EngineJudgment, Photo, TestBatch
from testing.services import ocr_analysis


def _make_judgment(batch, name, engine, ok='BENAR', sim_ns=100, sim_sp=100,
                   ratio_ns=1.0, ratio_sp=1.0, gt_text='B 1234 ABC'):
    p = Photo.objects.create(batch=batch, original_name=name, status='done',
                             system_result={'ocr_all': [{'engine': engine, 'text': 'B 1234 ABC'}]})
    ann, _ = Annotation.objects.get_or_create(
        photo=p, defaults={'gt_presence': 'ADA', 'detected_ok': 'BENAR',
                         'ocr_ok': 'BENAR' if ok == 'BENAR' else 'SALAH',
                         'gt_text': gt_text})
    return EngineJudgment.objects.create(
        annotation=ann, engine=engine, ok=ok,
        sim_nospasi=sim_ns, sim_spasi=sim_sp,
        ratio_nospasi=ratio_ns, ratio_spasi=ratio_sp)


class OcrAnalysisTest(TestCase):
    def test_empty_batch(self):
        batch = TestBatch.objects.create(name='E')
        result = ocr_analysis.analyze_ocr_similarity(batch)
        self.assertEqual(result['engines'], {})
        self.assertEqual(result['overall']['n_judged'], 0)
        self.assertIsNone(result['overall']['mean_sim_nospasi'])

    def test_per_engine_stats(self):
        batch = TestBatch.objects.create(name='B')
        _make_judgment(batch, 'a.jpg', 'easyocr', 'BENAR', 100, 100)
        _make_judgment(batch, 'b.jpg', 'easyocr', 'SALAH', 86, 80, 0.8571, 0.8)
        _make_judgment(batch, 'c.jpg', 'paddleocr', 'BENAR', 100, 100)
        result = ocr_analysis.analyze_ocr_similarity(batch)
        easy = result['engines']['easyocr']
        self.assertEqual(easy['n'], 2)
        self.assertEqual(easy['exact'], 1)
        self.assertEqual(easy['wrong'], 1)
        self.assertAlmostEqual(easy['mean_sim_nospasi'], 93.0)
        self.assertEqual(result['overall']['wrong_but_similar_80'], 1)

    def test_bins(self):
        batch = TestBatch.objects.create(name='B')
        _make_judgment(batch, 'a.jpg', 'easyocr', 'SALAH', 10, 10)
        _make_judgment(batch, 'b.jpg', 'easyocr', 'SALAH', 40, 40)
        _make_judgment(batch, 'c.jpg', 'easyocr', 'SALAH', 60, 60)
        _make_judgment(batch, 'd.jpg', 'easyocr', 'SALAH', 90, 90)
        _make_judgment(batch, 'e.jpg', 'easyocr', 'BENAR', 100, 100)
        result = ocr_analysis.analyze_ocr_similarity(batch)
        bins = result['engines']['easyocr']['bins_nospasi']
        self.assertEqual(bins['0–25%'], 1)
        self.assertEqual(bins['25–50%'], 1)
        self.assertEqual(bins['50–75%'], 1)
        self.assertEqual(bins['75–100%'], 1)
        self.assertEqual(bins['100%'], 1)

    def test_chart_data(self):
        batch = TestBatch.objects.create(name='B')
        _make_judgment(batch, 'a.jpg', 'easyocr', 'BENAR', 100, 100)
        _make_judgment(batch, 'b.jpg', 'paddleocr', 'SALAH', 50, 50)
        result = ocr_analysis.analyze_ocr_similarity(batch)
        chart = ocr_analysis.chart_data(result)
        self.assertEqual(chart['labels'], ['0–25%', '25–50%', '50–75%', '75–100%', '100%'])
        self.assertEqual(len(chart['datasets']), 2)
        easy = next(d for d in chart['datasets'] if d['label'] == 'easyocr')
        self.assertEqual(easy['data'], [0, 0, 0, 0, 1])

    def test_chart_data_includes_spasi_variant(self):
        batch = TestBatch.objects.create(name='B')
        _make_judgment(batch, 'a.jpg', 'easyocr', 'SALAH', 90, 30)
        result = ocr_analysis.analyze_ocr_similarity(batch)
        chart = ocr_analysis.chart_data(result)
        easy_sp = next(d for d in chart['datasets_spasi'] if d['label'] == 'easyocr')
        self.assertEqual(easy_sp['data'], [0, 1, 0, 0, 0])  # sim_spasi=30 -> bin 25–50%
