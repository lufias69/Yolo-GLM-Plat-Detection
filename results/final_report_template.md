# ANPR YOLO + OCR Final Report

## 0. Ringkasan Akurasi (Accuracy Summary)

| Komponen | Metrik akurasi | Nilai | Target | Status |
|----------|----------------|-------|--------|--------|
| Deteksi plat (YOLOv8n) | Akurasi deteksi plat (recall, IoU > 0.5, test set) | {det_acc:.2%} | >= 90% | {det_recall_status} |
| Deteksi plat (YOLOv8n) | mAP@0.5 | {mAP50:.2%} | >= 90% | {det_status} |
| OCR (crop dari box ground-truth) | Exact Plate Match Accuracy | {ocr_exact_match:.2%} | >= 85% | {ocr_status} |
| OCR (crop dari box prediksi) | Plate-level end-to-end accuracy | {e2e_plate_acc:.2%} | >= 75% | {ocr_pred_status} |
| End-to-end sistem | Full-plate accuracy per gambar | {image_level_acc:.2%} | >= 70% | {e2e_status} |

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
- Epochs: {epochs}
- Device: CPU
- Best weights: `{best_weights}`

### Validation metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | {mAP50:.4f} |
| mAP@0.5:0.95 | {mAP50_95:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |

## 3. OCR Evaluation

- Engine: {ocr_engine}
- Test crops: {n_crops}
- Preprocessing: {preprocess_desc}

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | {ocr_exact_match:.2%} |
| Mean CER | {ocr_cer_mean:.4f} |

### Ablation study

| CLAHE | Upscale | Binarize | Exact Match | CER |
|-------|---------|----------|-------------|-----|
{ablation_rows}

## 4. End-to-End Evaluation

| Metric | Value |
|--------|-------|
| Image-level full accuracy | {image_level_acc:.2%} |
| Detection recall (IoU > 0.5) | {det_recall:.2%} |
| End-to-end plate accuracy | {e2e_plate_acc:.2%} |
| Mean inference time | {mean_inf_ms:.1f} ms |
| FPS | {fps:.2f} |

## 5. Discussion

{discussion}

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
