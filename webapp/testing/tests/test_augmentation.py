import numpy as np
from django.test import SimpleTestCase

from testing.services.augmentation import perturb


class PerturbTest(SimpleTestCase):
    def setUp(self):
        self.img = (np.arange(60 * 80 * 3, dtype=np.int64).reshape(60, 80, 3) // 3) % 256
        self.img = self.img.astype(np.uint8)

    def test_same_seed_same_output(self):
        a = perturb(self.img, np.random.default_rng(7))
        b = perturb(self.img, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a, b))

    def test_different_seed_different_output(self):
        a = perturb(self.img, np.random.default_rng(1))
        b = perturb(self.img, np.random.default_rng(2))
        self.assertFalse(np.array_equal(a, b))

    def test_shape_dtype_range_preserved(self):
        out = perturb(self.img, np.random.default_rng(3))
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)
