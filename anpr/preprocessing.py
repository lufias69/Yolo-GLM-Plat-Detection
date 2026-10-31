"""
anpr/preprocessing.py
Image preprocessing helpers for OCR on license-plate crops.
"""
import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    gray = grayscale(image)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe_obj.apply(gray)


def otsu_threshold(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    gray = grayscale(image)
    mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(image: np.ndarray, inverse: bool = False, block_size: int = 11, c: int = 2) -> np.ndarray:
    gray = grayscale(image)
    mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(gray, 255, mode, threshold_type, block_size, c)


def upscale(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    if scale <= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def denoise(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)


def auto_invert_if_dark_background(image: np.ndarray) -> np.ndarray:
    """Invert if background is darker than foreground (light text on dark plate)."""
    gray = grayscale(image)
    # sample border pixels as proxy for background
    h, w = gray.shape
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    border_mean = border.mean()
    center_mean = gray[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()
    if border_mean < center_mean:
        return cv2.bitwise_not(gray)
    return gray


def pipeline(image: np.ndarray, use_clahe: bool = True, use_upscale: bool = True,
             use_binarize: bool = False, use_denoise: bool = False) -> np.ndarray:
    """Configurable preprocessing chain returning BGR or grayscale numpy array."""
    out = image.copy()
    if use_denoise:
        out = denoise(out)
    if use_clahe:
        out = clahe(out)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if use_upscale:
        out = upscale(out, scale=2.0)
    if use_binarize:
        out = otsu_threshold(out)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out
