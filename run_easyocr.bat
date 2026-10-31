@echo off
setlocal
set PYTHONPATH=C:\project\nuraeni
C:\Users\lufias\miniforge3\envs\mace_md\python.exe C:\project\nuraeni\scripts\generate_ocr_pseudolabels.py --engine easyocr
pause
