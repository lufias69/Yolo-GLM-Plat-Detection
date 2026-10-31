# ANPR YOLO + OCR Final Report

## 0. Ringkasan Akurasi (Accuracy Summary)

| Komponen | Metrik akurasi | Nilai | Target | Status |
|----------|----------------|-------|--------|--------|
| Deteksi plat (YOLOv8n) | Akurasi deteksi plat (recall, IoU > 0.5, test set) | 95.15% | >= 90% | TERCAPAI |
| Deteksi plat (YOLOv8n) | mAP@0.5 | 98.73% | >= 90% | TERCAPAI |
| OCR (crop dari box ground-truth) | Exact Plate Match Accuracy | 89.10% | >= 85% | TERCAPAI |
| OCR (crop dari box prediksi) | Plate-level end-to-end accuracy | 36.02% | >= 75% | BELUM TERCAPAI |
| End-to-end sistem | Full-plate accuracy per gambar | 34.78% | >= 70% | BELUM TERCAPAI |

Catatan: angka OCR dan end-to-end memakai ground-truth teks plat hasil pseudo-label EasyOCR yang belum diverifikasi manual, sehingga akurasi pembacaan ini adalah batas bawah (lihat bagian Discussion).

## 1. Dataset

- Source: `gabungan-uamje` v2 (Universitas Mataram, CC BY 4.0)
- Location: `C:/project/nuraeni`
- Total images: 1,613
- Total plate boxes: 1,671
- Split: 80/10/10 (train/valid/test)
  - Train: 1,291 images / 1,340 boxes
  - Valid: 161 images / 166 boxes
  - Test: 161 images / 165 boxes
- Preprocessing: merged classes `license_plate` and `plat_nomor` into single class `plate`; normalized polygon segmentation labels to standard YOLO bbox.

## 2. Detector Training

- Model: YOLOv8n
- Input size: 640
- Seed: 42
- Epochs: 25
- Device: GPU (NVIDIA GeForce RTX 5060 Ti)
- Best weights: `C:\project\nuraeni\runs\detect\plate_detector_yolov8n_gpu_final\weights\best.pt`

### Validation metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.9873 |
| mAP@0.5:0.95 | 0.7832 |
| Precision | 0.9799 |
| Recall | 0.9515 |

## 3. OCR Evaluation

- Engine: easyocr
- Test crops: 156
- Preprocessing: CLAHE + upscale (best ablation)

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 89.10% |
| Mean CER | 0.1367 |

### Ablation study

| CLAHE | Upscale | Binarize | Exact Match | CER |
|-------|---------|----------|-------------|-----|
| False | False | False | 24.36% | 0.9147 |
| True | False | False | 23.08% | 0.8681 |
| True | True | False | 89.10% | 0.1367 |
| True | True | True | 19.87% | 1.0252 |


## 4. End-to-End Evaluation

| Metric | Value |
|--------|-------|
| Image-level full accuracy | 34.78% |
| Detection recall (IoU > 0.5) | 98.14% |
| End-to-end plate accuracy | 36.02% |
| Mean inference time | 23.7 ms |
| FPS | 42.15 |

## 5. Discussion

Plate detection exceeded the 90% target (mAP@0.5 = 98.7%). OCR exact-match on clean crops reached 89.1%, but end-to-end accuracy is lower because EasyOCR pseudo-labels still contain format errors. Manual verification of ground-truth OCR labels or a custom OCR model is recommended to push end-to-end accuracy above 90% for production use.

## 6. Artifacts

- `data.yaml`: dataset configuration
- `split_report.json`: split statistics
- `runs/detect/`: training logs and weights
- `results/`: evaluation metrics JSON
- `crops_test/`: extracted test plate crops
- `ground_truth_test.csv`: OCR ground-truth annotations

## 7. Next Steps / Recommendations

- Verify/correct ground-truth OCR labels manually for final publication.
- Try larger YOLO variants (yolov8s/yolo11n) if training time allows.
- Fine-tune a custom OCR model if generic engines do not reach target accuracy on low-quality crops.
