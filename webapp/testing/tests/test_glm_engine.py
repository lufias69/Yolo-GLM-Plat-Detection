from unittest.mock import patch

from django.test import SimpleTestCase

from anpr.ocr_engines import ENGINES, GLMOCREngine, plate_shape_score


class PlateShapeScoreTest(SimpleTestCase):
    def test_plate_like_scores_high(self):
        self.assertGreater(plate_shape_score('DR 1282 AS'), 0.7)
        self.assertGreater(plate_shape_score('AB 3906 XH'), 0.7)

    def test_year_sticker_scores_low(self):
        self.assertLess(plate_shape_score('05 28'), 0.5)
        self.assertLess(plate_shape_score('26'), 0.5)

    def test_empty(self):
        self.assertEqual(plate_shape_score(''), 0.0)


class GLMEngineTest(SimpleTestCase):
    def test_registered(self):
        self.assertIn('glm-ocr', ENGINES)

    def test_picks_plate_line_over_sticker(self):
        eng = GLMOCREngine()
        calls = {}

        class Resp:
            status_code = 200

            def json(self):
                return {'response': 'DR 1282 AS\n05:28'}

            def raise_for_status(self):
                pass

        def fake_post(url, json=None, timeout=None):
            calls['payload'] = json
            return Resp()

        import numpy as np
        with patch('requests.post', fake_post):
            text, conf = eng.recognize(np.zeros((40, 100, 3), dtype=np.uint8))
        self.assertEqual(text, 'DR 1282 AS')
        self.assertGreater(conf, 0.7)
        self.assertEqual(calls['payload']['model'], 'glm-ocr:q8_0')
