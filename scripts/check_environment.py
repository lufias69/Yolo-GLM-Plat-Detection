"""
check_environment.py
Check Python version, device, and import availability for ANPR pipeline.
"""
import importlib
import platform
import sys

import torch


def check():
    print(f'Python: {sys.version}')
    print(f'Platform: {platform.system()} {platform.release()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

    libs = [
        'cv2', 'numpy', 'pandas', 'matplotlib', 'tqdm',
        'ultralytics', 'torch', 'PIL',
        'pytesseract', 'easyocr', 'paddleocr',
    ]
    print('\nLibrary status:')
    for lib in libs:
        try:
            importlib.import_module(lib)
            print(f'  {lib}: OK')
        except Exception as e:
            print(f'  {lib}: NOT INSTALLED ({e})')


if __name__ == '__main__':
    check()
