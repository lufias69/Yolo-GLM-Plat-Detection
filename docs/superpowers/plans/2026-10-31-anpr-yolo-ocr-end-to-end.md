# ANPR YOLO + OCR End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, evaluate, and validate a complete ANPR pipeline (YOLO plate detection + OCR) on the `gabungan-uamje` Indonesian license-plate dataset, with plate-detection mAP@0.5 > 0.90 and end-to-end plate-text accuracy > 0.90.

**Architecture:** Two-stage pipeline — (1) Ultralytics YOLO detector trained to detect a single `plate` class; (2) OCR engine on cropped plates with optional preprocessing and Indonesian plate regex post-processing. Evaluation scripts produce per-stage and end-to-end metrics.

**Tech Stack:** Python 3.11, Ultralytics YOLOv8/YOLO11, OpenCV, PaddleOCR/EasyOCR/Tesseract, PyTorch, pandas, matplotlib.

**Spec:** `C:/project/nuraeni/perencanaan-riset-yolo-ocr.md`

## Global Constraints

- Dataset root: `C:/project/nuraeni`
- Single detection class: `plate` (class id 0)
- `data.yaml` already defines `train`, `val`, `test` paths correctly.
- Train/val/test split fixed at 80/10/10 with seed 42 (1291/161/161 images).
- All training must be reproducible with a fixed random seed.
- Detector target: mAP@0.5 >= 0.90 on `test` set.
- OCR target: character-level accuracy or exact-plate accuracy must be reported; user asks for overall > 90% accuracy.
- All deliverables live under `C:/project/nuraeni`; no external publishing without confirmation.

## Review Focus

1. **Polygon labels in source data** — some YOLO labels are polygons (11 numbers) rather than xywh boxes; cropping and label parsing must normalize them correctly or training will see warped boxes.
2. **Empty/invalid crops** — very small or out-of-frame predicted boxes must not crash OCR stage; filter by min width/height and confidence.
3. **Indonesian plate format normalization** — spaces and ambiguous characters (`O` vs `0`, `I` vs `1`) must be handled with a whitelist + regex to avoid false mismatches.
4. **Hardware variability** — target mAP/FPS numbers must come from the actual local CPU/GPU run, not copied from literature; record device info.
5. **Class imbalance / multi-plate images** — stratified split already preserves multi-plate distribution, but evaluation must report per-image vs per-box metrics.

---

## Task 1: Finalize Environment & Data Sanity Checks

**Files:**
- Create: `requirements.txt`
- Create: `scripts/check_environment.py`
- Modify: `extract_test_plates.py` (polygon parsing already added; verify behavior)

**Interfaces:**
- Consumes: `data.yaml`, `train/valid/test` folders.
- Produces: printed/JSON report of class counts, image/label pairs, polygon-vs-bbox stats.

- [ ] **Step 1: Write `requirements.txt` with exact library versions that install cleanly**

  Contents must include at least: `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `pytesseract`, `easyocr`, `paddleocr`, `pillow`.

- [ ] **Step 2: Run environment check script**

  Run: `venv/Scripts/python scripts/check_environment.py`
  Expected: reports Python version, CUDA/CPU device, and import status of each library.

- [ ] **Step 3: Verify dataset labels after polygon merge**

  Run a Python snippet to count lines in all label files, report how many are standard bbox vs polygon, and ensure class id is always 0.
  Expected: 1671 boxes total across all splits, class ids = `{0}` only.

- [ ] **Step 4: Commit**

  ```bash
  git init  # if not already a repo
  git add requirements.txt scripts/check_environment.py extract_test_plates.py
  git commit -m "chore: add requirements and environment check"
  ```

---

## Task 2: Build Ground-Truth OCR CSV for Test Set

**Files:**
- Create: `scripts/annotate_test_plates.py`
- Create: `ground_truth_test.csv` (final artifact)

**Interfaces:**
- Consumes: `test/images`, `test/labels`.
- Produces: `ground_truth_test.csv` with columns `image_file, plate_index, crop_file, x1, y1, x2, y2, plate_text, source`.

- [ ] **Step 1: Extract cropped plates from test set into `crops_test/`**

  Use corrected polygon-to-bbox logic; skip empty crops; report count.
  Expected: ~165 crops for 161 test images.

- [ ] **Step 2: Generate initial pseudo-labels with PaddleOCR (fallback EasyOCR)**

  Run OCR on each crop; store the highest-confidence text in `plate_text` with `source=pseudo`.
  Expected: `ground_truth_test.csv` with every row populated.

- [ ] **Step 3: Human review / correction note**

  The user must review and correct `ground_truth_test.csv` (manual step). Mark rows `source=manual` after correction.
  Expected: no empty `plate_text` values in final CSV.

- [ ] **Step 4: Commit**

  ```bash
  git add scripts/annotate_test_plates.py ground_truth_test.csv crops_test/
  git commit -m "feat: add test set ground-truth OCR annotations"
  ```

---

## Task 3: Train YOLO Plate Detector

**Files:**
- Create: `scripts/train_detector.py`
- Create: `detector_config.yaml`
- Modify: `data.yaml` if absolute path causes issues (switch to relative)

**Interfaces:**
- Consumes: `data.yaml`, Ultralytics.
- Produces: trained model directory `runs/detect/plate_detector/` containing `weights/best.pt` and training logs.

- [ ] **Step 1: Pick detector variants to train**

  Use `yolov8n.pt` first (fast baseline). If GPU/time allows, also train `yolov8s.pt` and `yolo11n.pt`.
  Fixed seed: `seed=42`, `imgsz=640`, `epochs=200`, `patience=20`, `batch=16` (or auto if memory limited).

- [ ] **Step 2: Run training for primary model**

  Run: `venv/Scripts/python scripts/train_detector.py --model yolov8n --epochs 200 --data data.yaml`
  Expected: training completes and saves `runs/detect/train/weights/best.pt`.

- [ ] **Step 3: Verify detector training metrics**

  Read `runs/detect/train/results.csv`; print final `metrics/mAP50(B)` and `metrics/mAP50-95(B)`.
  Target: mAP@0.5 >= 0.90 on validation set.

- [ ] **Step 4: Commit / record run info**

  Save a small `training_log.json` with model name, epochs, seed, final mAP, and device.

---

## Task 4: Evaluate YOLO Detector on Test Set

**Files:**
- Create: `scripts/evaluate_detector.py`

**Interfaces:**
- Consumes: `best.pt`, `data.yaml`.
- Produces: `results/detector_test_metrics.json` with mAP50, mAP50-95, precision, recall, FPS.

- [ ] **Step 1: Run Ultralytics validation on test split**

  Use `model.val(data='data.yaml', split='test')` and capture metrics.

- [ ] **Step 2: Export per-image predictions to CSV**

  Columns: `image_file, pred_boxes, pred_confidences` for downstream OCR use.

- [ ] **Step 3: Report test metrics and verify target**

  Expected: `mAP50` >= 0.90. If not, iterate (more epochs, larger model, or augmentation tuning).

---

## Task 5: Build OCR Preprocessing & Inference Pipeline

**Files:**
- Create: `scripts/ocr_pipeline.py`
- Create: `anpr/preprocessing.py`
- Create: `anpr/postprocessing.py`

**Interfaces:**
- Consumes: crop image (numpy array or path), OCR engine name.
- Produces: `(text, confidence)` tuple and optionally preprocessed image.

- [ ] **Step 1: Implement preprocessing helpers**

  Functions: `grayscale`, `clahe`, `otsu_threshold`, `upscale`, `binarize_by_plate_color`.

- [ ] **Step 2: Implement OCR engine wrappers**

  Wrappers for PaddleOCR, EasyOCR, Tesseract with consistent interface.

- [ ] **Step 3: Implement Indonesian plate post-processing**

  Function `normalize_plate_text(text)`:
  - Uppercase, strip non-alphanumeric except space.
  - Replace `O` with `0` and `I` with `1` only when context suggests digits.
  - Validate against regex `^[A-Z]{1,2}\s*[0-9]{1,4}\s*[A-Z]{0,3}$`.

- [ ] **Step 4: Test pipeline on a few crops manually**

  Print raw OCR output and normalized output for 5 sample crops.

---

## Task 6: Evaluate OCR on Ground-Truth Test Crops

**Files:**
- Create: `scripts/evaluate_ocr.py`

**Interfaces:**
- Consumes: `ground_truth_test.csv`, `crops_test/`, chosen OCR engine.
- Produces: `results/ocr_test_metrics.json` with exact-match accuracy and CER.

- [ ] **Step 1: Run OCR on every ground-truth crop**

  For each crop, apply best-performing preprocessing chain and record prediction.

- [ ] **Step 2: Compute exact-match and character-error metrics**

  Exact Plate Match (%) and Character Error Rate using ground-truth `plate_text`.
  Target: exact match >= 90% for user target.

- [ ] **Step 3: Compare preprocessing ablations**

  Test at least 4 variants: (a) no preprocessing, (b) CLAHE, (c) CLAHE+upscale, (d) full chain.
  Save comparison table to `results/ocr_ablation.csv`.

---

## Task 7: End-to-End ANPR Validation

**Files:**
- Create: `scripts/evaluate_endtoend.py`

**Interfaces:**
- Consumes: detector `best.pt`, OCR pipeline, `ground_truth_test.csv`, `data.yaml`.
- Produces: `results/end_to_end_metrics.json`.

- [ ] **Step 1: Run detector + OCR on each test image**

  For every predicted plate box, crop, run OCR, normalize, and compare to all ground-truth plates in that image.

- [ ] **Step 2: Compute end-to-end per-image accuracy**

  Metric: percentage of test images where every ground-truth plate is correctly detected and recognized.
  Also report per-plate accuracy (detection IoU > 0.5 AND text exact match).

- [ ] **Step 3: Measure inference speed**

  Report ms/frame for detector alone and full pipeline, plus FPS.

- [ ] **Step 4: Generate error gallery**

  Save up to 20 failure case images to `results/error_gallery/`.

---

## Task 8: Final Report & Model Packaging

**Files:**
- Create: `results/final_report.md`
- Create: `scripts/export_model.py`

**Interfaces:**
- Consumes: all metrics JSON files and model weights.
- Produces: final report and optional ONNX export.

- [ ] **Step 1: Summarize all metrics in `results/final_report.md`**

  Include dataset stats, detector mAP, OCR exact-match, end-to-end accuracy, FPS, preprocessing ablation, and device info.

- [ ] **Step 2: Export best model to ONNX**

  Run Ultralytics export: `model.export(format='onnx', imgsz=640)`.
  Save artifact path in report.

- [ ] **Step 3: Final commit and summary to user**

  Commit all new scripts, results, and report.

---

## Execution Recommendation

Because this plan involves many interdependent tasks, GPU-dependent training, and manual annotation review, I recommend **native execution with human checkpoint after Task 2** (ground-truth review) and after Task 3 (training metrics). The remaining evaluation tasks can then run automatically.

Does this plan capture what you want, and should I proceed with native execution?
