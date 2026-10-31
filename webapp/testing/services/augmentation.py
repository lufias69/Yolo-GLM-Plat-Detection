"""
testing/services/augmentation.py
Light random perturbations applied per consistency-test run (spec 7.4):
mimics real-world variation so the 100x run measures robustness, not
determinism of an identical input.
"""
import cv2
import numpy as np


def perturb(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    brightness = 1.0 + rng.uniform(-0.2, 0.2)
    contrast = 1.0 + rng.uniform(-0.2, 0.2)
    out = image_bgr.astype(np.float32)
    h, w = out.shape[:2]
    dx, dy = rng.uniform(-0.004, 0.004) * w, rng.uniform(-0.004, 0.004) * h
    out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                         borderMode=cv2.BORDER_REPLICATE)
    mean = out.mean()
    out = (out - mean) * contrast + mean
    out *= brightness
    out = out + rng.normal(0.0, 3.0, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
