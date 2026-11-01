"""Uji perluasan analisis OCR: CER per engine, error pattern, E2E,
skenario, inferensial (Wilson CI, McNemar), dan data visualisasi."""
import re

from django.test import TestCase

from testing.models import Annotation, EngineJudgment, Photo, TestBatch
from testing.services import ocr_analysis
from testing.services.metrics import similarity


def _make_photo(batch, name, gt, entries, detected_ok='BENAR'):
    """entries: [(engine, text, ok, conf)] — sim disalin seperti forms.save_judgments."""
    p = Photo.objects.create(
        batch=batch, original_name=name, status='done',
        system_result={'ocr_all': [{'engine': e, 'text': t, 'conf': c}
                                   for e, t, _, c in entries]})
    ann = Annotation.objects.create(
        photo=p, gt_presence='ADA', detected_ok=detected_ok,
        ocr_ok='BENAR' if all(ok == 'BENAR' for _, _, ok, _ in entries) else 'SALAH',
        gt_text=gt)
    for eng, text, ok, conf in entries:
        text_nospasi = re.sub(r'\s+', '', text)
        gt_nospasi = re.sub(r'\s+', '', gt)
        EngineJudgment.objects.create(
            annotation=ann, engine=eng, ok=ok,
            sim_nospasi=similarity(text_nospasi, gt_nospasi),
            sim_spasi=similarity(text, gt))
    return p


class PerEngineMetricsTest(TestCase):
    def test_cer_and_char_accuracy_per_engine(self):
        batch = TestBatch.objects.create(name='B')
        # GT 'B1234' (5 char). easyocr salah 1 char -> CER 0.2, char acc 0.8
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9)])
        # paddle benar -> CER 0
        _make_photo(batch, 'b.jpg', 'B1234', [('paddleocr', 'B1234', 'BENAR', 0.95)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        easy = result['engines']['easyocr']
        self.assertAlmostEqual(easy['cer_mean'], 0.2)
        self.assertAlmostEqual(easy['char_accuracy'], 0.8)
        self.assertAlmostEqual(result['engines']['paddleocr']['cer_mean'], 0.0)

    def test_exact_ci_wilson(self):
        batch = TestBatch.objects.create(name='B')
        for i in range(10):
            _make_photo(batch, f'{i}.jpg', 'B1234',
                        [('easyocr', 'B1234' if i < 8 else 'B1284',
                          'BENAR' if i < 8 else 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        easy = result['engines']['easyocr']
        self.assertIsNotNone(easy['ci_low'])
        self.assertLessEqual(easy['ci_low'], 0.8)
        self.assertGreaterEqual(easy['ci_high'], 0.8)

    def test_quartiles_for_boxplot(self):
        batch = TestBatch.objects.create(name='B')
        sims = [10, 30, 50, 70, 90]
        for i, s in enumerate(sims):
            p = Photo.objects.create(batch=batch, original_name=f'{i}.jpg', status='done',
                                     system_result={'ocr_all': [{'engine': 'easyocr', 'text': 'X', 'conf': 0.5}]})
            ann = Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                            ocr_ok='SALAH', gt_text='B1234')
            EngineJudgment.objects.create(annotation=ann, engine='easyocr', ok='SALAH',
                                          sim_nospasi=s, sim_spasi=s)
        result = ocr_analysis.analyze_ocr_similarity(batch)
        q = result['engines']['easyocr']['quartiles_nospasi']
        self.assertEqual(len(q), 4)  # min, q1, median(q3 in 5 pts), max-ish
        self.assertEqual(q[0], 10)
        self.assertEqual(q[3], 90)

    def test_mean_conf_per_engine(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1234', 'BENAR', 0.9),
                                              ('easyocr2x', 'X', 'SALAH', 0.3)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        self.assertAlmostEqual(result['engines']['easyocr']['mean_conf'], 0.9)


class ErrorPatternTest(TestCase):
    def test_confusion_pairs(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9)])
        _make_photo(batch, 'b.jpg', 'Z1234', [('easyocr', '21234', 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        pairs = {(p['gt'], p['pred']): p['count'] for p in result['error_patterns']['confusions']}
        self.assertEqual(pairs.get(('3', '8')), 1)
        self.assertEqual(pairs.get(('Z', '2')), 1)

    def test_heatmap_grid_prepared(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        hm = result['error_patterns']['heatmap']
        self.assertIn('3', hm['rows'])
        self.assertIn('8', hm['cols'])
        cell = hm['cells']['3']['8']
        self.assertEqual(cell['n'], 1)
        self.assertAlmostEqual(cell['alpha'], 1.0)

    def test_position_breakdown(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B123456', [('easyocr', 'X123456', 'SALAH', 0.9)])
        _make_photo(batch, 'b.jpg', 'B123456', [('easyocr', 'B12X456', 'SALAH', 0.9)])
        _make_photo(batch, 'c.jpg', 'B123456', [('easyocr', 'B12345X', 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        pos = result['error_patterns']['positions']
        self.assertEqual(pos['depan'], 1)
        self.assertEqual(pos['tengah'], 1)
        self.assertEqual(pos['belakang'], 1)

    def test_length_vs_errors(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B123', [('easyocr', 'B128', 'SALAH', 0.9)])
        _make_photo(batch, 'b.jpg', 'B1234ABC', [('easyocr', 'B1234ABX', 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        by_len = result['error_patterns']['by_length']
        self.assertEqual(by_len[4]['n'], 1)
        self.assertEqual(by_len[8]['n'], 1)
        self.assertEqual(by_len[8]['errors'], 1)


class EndToEndTest(TestCase):
    def test_e2e_per_engine(self):
        batch = TestBatch.objects.create(name='B')
        # foto deteksi benar + easyocr benar = e2e sukses
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1234', 'BENAR', 0.9)])
        # deteksi GAGAL -> e2e gagal walau OCR text ada
        _make_photo(batch, 'b.jpg', 'B1234', [('easyocr', 'B1234', 'BENAR', 0.9)],
                    detected_ok='GAGAL')
        # deteksi benar, OCR salah
        _make_photo(batch, 'c.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        easy = result['engines']['easyocr']
        self.assertEqual(easy['e2e']['n'], 3)
        self.assertEqual(easy['e2e']['success'], 1)
        # deteksi benar tapi OCR salah, terpisah dari gagal deteksi
        self.assertEqual(easy['e2e']['detected_ok_but_wrong'], 1)
        self.assertEqual(easy['e2e']['detection_failed'], 1)


class ScenarioTest(TestCase):
    def test_hard_scenario_robustness(self):
        """'sulit' = ada engine lain yang benar pada foto yang sama."""
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9),
                                              ('paddleocr', 'B1234', 'BENAR', 0.9)])
        _make_photo(batch, 'b.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9),
                                              ('paddleocr', 'B1284', 'SALAH', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        # foto 'a' sulit (paddle benar, easy salah): easy exact di sulit = 0
        self.assertEqual(result['engines']['easyocr']['hard']['n'], 1)
        self.assertEqual(result['engines']['easyocr']['hard']['exact'], 0)
        self.assertEqual(result['engines']['paddleocr']['hard']['n'], 1)
        self.assertEqual(result['engines']['paddleocr']['hard']['exact'], 1)

    def test_overconfident(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.95)])
        _make_photo(batch, 'b.jpg', 'B1234', [('paddleocr', 'B1284', 'SALAH', 0.4)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        easy = result['engines']['easyocr']
        self.assertEqual(easy['overconfident']['n'], 1)
        self.assertEqual(easy['overconfident']['count'], 1)
        self.assertEqual(result['engines']['paddleocr']['overconfident']['count'], 0)


class InferentialTest(TestCase):
    def test_mcnemar_pairwise(self):
        batch = TestBatch.objects.create(name='B')
        # 8 foto easy benar paddle salah, 1 foto sebaliknya -> discordance 9
        for i in range(8):
            _make_photo(batch, f'{i}.jpg', 'B1234', [('easyocr', 'B1234', 'BENAR', 0.9),
                                                     ('paddleocr', 'B1284', 'SALAH', 0.9)])
        _make_photo(batch, 'z.jpg', 'B1234', [('easyocr', 'B1284', 'SALAH', 0.9),
                                              ('paddleocr', 'B1234', 'BENAR', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        comp = result['comparisons']
        self.assertTrue(comp['available'])
        pair = next(c for c in comp['pairs'] if set(c['engines']) == {'easyocr', 'paddleocr'})
        self.assertEqual(pair['b'], 8)
        self.assertEqual(pair['c'], 1)
        self.assertLess(pair['p_value'], 0.05)
        self.assertGreater(pair['p_value'], 0.0)

    def test_mcnemar_not_enough_data(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1234', 'BENAR', 0.9),
                                              ('paddleocr', 'B1234', 'BENAR', 0.9)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        pair = result['comparisons']['pairs'][0]
        self.assertIsNone(pair['p_value'])  # b + c = 0, uji tak terdefinisi


class ChartDataTest(TestCase):
    def test_extra_charts(self):
        batch = TestBatch.objects.create(name='B')
        _make_photo(batch, 'a.jpg', 'B1234', [('easyocr', 'B1234', 'BENAR', 0.9),
                                              ('paddleocr', 'B1284', 'SALAH', 0.85)])
        result = ocr_analysis.analyze_ocr_similarity(batch)
        charts = ocr_analysis.chart_data_all(result)
        self.assertIn('exact', charts)
        self.assertIn('quartiles', charts)
        self.assertIn('scatter', charts)
        self.assertEqual(charts['exact']['labels'], ['easyocr', 'paddleocr'])
        self.assertEqual(charts['quartiles']['labels'], ['easyocr', 'paddleocr'])
        # scatter: satu titik per (engine, foto) yg punya conf+sim
        self.assertEqual(len(charts['scatter']['datasets']), 2)
        easy_pts = next(d for d in charts['scatter']['datasets'] if d['label'] == 'easyocr')['data']
        self.assertEqual(easy_pts, [{'x': 0.9, 'y': 100}])
