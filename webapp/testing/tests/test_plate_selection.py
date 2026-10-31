from django.test import SimpleTestCase

from testing.services.inference import select_plate_fragment

# fragmen nyata dari diagnosis 6 foto test (teks ternormalisasi, conf)
def f(t, c):
    return {'text': t, 'conf': c}


class SelectPlateFragmentTest(SimpleTestCase):
    def test_prefers_plate_over_year_sticker(self):
        cases = [
            ([f('KT 1910 0I', 0.145), f('26', 0.931)], 'KT 1910 0I'),
            ([f('DR 4837 EO', 0.348), f('0525', 0.478)], 'DR 4837 EO'),
            ([f('DR 8238 BE', 0.506), f('02428', 0.624)], 'DR 8238 BE'),
            ([f('DRL 137 FM', 0.389), f('08 27', 0.774)], 'DRL 137 FM'),
            ([f('DR 1298 SK', 0.643), f('I 8', 0.101)], 'DR 1298 SK'),
            ([f('DR 1282 AS', 0.32)], 'DR 1282 AS'),
        ]
        for frags, expected in cases:
            self.assertEqual(select_plate_fragment(frags), expected)

    def test_falls_back_to_conf_when_no_plate_shape(self):
        self.assertEqual(select_plate_fragment([f('XX', 0.2), f('YY', 0.9)]), 'YY')

    def test_empty(self):
        self.assertEqual(select_plate_fragment([]), '')
