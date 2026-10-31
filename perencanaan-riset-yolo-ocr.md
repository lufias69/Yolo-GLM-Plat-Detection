# Perencanaan Riset (Kerangka Proposal Skripsi)

**Deteksi dan Pengenalan Nomor Plat Kendaraan (ANPR) menggunakan YOLO untuk Deteksi Objek dan Optical Character Recognition (OCR)**

Dataset: `gabungan-uamje` v2 (Universitas Mataram, CC BY 4.0) — tersimpan di `C:\project\nuraeni`

---

## 1. Usulan Judul (pilih/sesuaikan)

1. *"Sistem Automatic Number Plate Recognition (ANPR) untuk Plat Nomor Kendaraan Indonesia Menggunakan YOLOv8 dan OCR"*
2. *"Deteksi dan Pengenalan Plat Nomor Kendaraan Bermotor dengan Arsitektur Two-Stage: YOLO untuk Deteksi dan [Tesseract/EasyOCR/PaddleOCR] untuk Recognisi Karakter"*
3. *"Evaluasi Kinerja Model YOLO dan Engine OCR pada Sistem ANPLR (Automatic Number Plate License Recognition) Berbasis Citra Real-world Indonesia"*

---

## 2. Latar Belakang (ringkas)

- ANPR penting untuk ITS: ETLE (electronic traffic law enforcement), parkir, gerbang tol.
- Plat Indonesia punya karakter khas: format `Kota  Nomor Seri  Huruf belakang` (mis. `B 1234 ABC`, `DR 1570 PW`), kombinasi huruf+vokal dengan spasi, warna dasar & teks variatif (hitam, putih, kuning, merah).
- Engine OCR umum (Tesseract) dilatih dominan pada teks Latin/Eropa → akurasi pada plat Indonesia bisa turun; perlu studi preprocessing dan perbandingan engine.
- Dataset publik plat Indonesia terbatas; dataset ini (1.613 citra) bisa dipakai untuk deteksi, tapi **belum punya label teks plat** → menjadi salah satu kontribusi riset (penyusunan subset ground-truth OCR).

## 3. Rumusan Masalah

1. Seberapa efektif model keluarga YOLO (mis. YOLOv8n/s, YOLO11n/s) mendeteksi plat nomor kendaraan Indonesia pada citra dunia nyata?
2. Engine OCR mana (Tesseract, EasyOCR, PaddleOCR, atau model CRNN custom) yang menghasilkan akurasi pembacaan tertinggi untuk plat Indonesia pasca-crop dari detektor?
3. Seberapa besar pengaruh tahapan preprocessing citra (grayscale, CLAHE, thresholding, upscaling, deskew) terhadap akurasi OCR?
4. Bagaimana kinerja end-to-end sistem (akurasi dan kecepatan) untuk potensi penggunaan real-time?

## 4. Tujuan Penelitian

1. Melatih dan mengevaluasi model deteksi plat nomor berbasis YOLO pada dataset `gabungan-uamje`.
2. Membandingkan minimal 3 engine OCR pada crop plat hasil deteksi.
3. Mengukur kontribusi tiap langkah preprocessing terhadap akurasi OCR (ablation study).
4. Menyusun prototipe pipeline ANPR end-to-end dengan metrik akurasi & FPS, serta subset ground-truth teks plat untuk evaluasi OCR.

## 5. Manfaat

- **Akademis:** studi perbandingan terkontrol YOLO × OCR khusus plat Indonesia; dataset subset berlabel teks plat bisa menjadi kontribusi terbuka.
- **Praktis:** rekomendasi konfigurasi pipeline ANPR (model, engine, preprocessing) yang murah & bisa jalan di hardware terbatas.

## 6. Batasan & Ruang Lingkup

- Hanya citra plat kendaraan Indonesia (2 kelas sumber yang akan digabung — lihat §7).
- Tidak termasuk: OCR plat luar negeri, kendaraan khusus (kereta/api), integrasi kamera + infrastruktur produksi.
- Fokus pada **two-stage pipeline** (deteksi → OCR), dengan opsional perbandingan singkat terhadap pendekatan end-to-end (mis. PaddleOCR det+rec langsung) sebagai pembanding.
- Bahasa penulisan karakter: alphanumeric Latin + angka; validasi format memakai pola plat Indonesia.

## 7. Analisis Dataset Saat Ini (hasil inspeksi)

| Aspek | Kenyataan | Implikasi riset |
|---|---|---|
| Jumlah citra | 1.613 (semua `.jpg`, 640×640, ~113 MB) | Cukup untuk fine-tune YOLO; perlu hati-hati overfit |
| Split | **Hanya `train/`** (`valid/` dan `test/` dirujuk `data.yaml` tapi tidak ada di file) | **Wajib** membuat ulang split train/val/test dari nol |
| Kelas | `license_plate` (687 box) dan `plat_nomor` (984 box) — sinonim, berasal dari penggabungan 2 sumber | Gabung jadi 1 kelas `plate`; dokumentasikan sebagai keputusan preprocessing |
| Total box | 1.671; 1.563 citra 1 plat, 50 citra multi-plat (maks 5) | Cocok untuk uji deteksi multi-objek |
| Label teks plat | **Tidak ada** (hanya bounding box) | Perlu pembuatan ground-truth OCR: transkripsi manual 300–500 plat (±2–4 jam kerja) — jadikan kontribusi dataset |
| Resolusi | seragam 640×640 (pra-proses Roboflow) | Latih `imgsz=640` langsung; jika butuh detail karakter, uji crop-resolution |
| Lisensi | CC BY 4.0 | Bebas untuk riset & publikasi dengan atribusi |

## 8. Metodologi Penelitian

### 8.1 Tahap 1 — Persiapan Data
1. Audit kualitas label (visual sampling ≥100 citra; perbaiki box buruk bila perlu).
2. Gabungkan kelas → `plate` (single-class).
3. Split acak stratified **80/10/10** (train/val/test) ≈ 1.290/161/162; pastikan citra multi-plat tidak tersebar tidak merata (split per-citra, bukan per-box).
4. Ground-truth OCR: dari test set, crop semua plat (±170–200 box) + sampel acak tambahan → transkripsi manual dengan tools (CVAT/LabelStudio); simpan sebagai CSV `image_path, plate_text, bbox`.
5. Ekstraksi metadata teks (opsional, untuk analisis): distribusi prefix kota (B, D, DR, ...) dan panjang string.

### 8.2 Tahap 2 — Deteksi (YOLO)
- Model: **YOLOv8n, YOLOv8s** dan **YOLO11n, YOLO11s** (library `ultralytics`) — kontrol budget komputasi & efek skala.
- Konfigurasi dasar: `imgsz=640`, epochs 100–200 + early stop patience 20, Augmentasi bawaan Ultralytics; **tanpa** augmentasi ekstrem yang merusak proporsi plat (rotasi besar).
- Transfer learning: mulai dari bobot pra-latih COCO.
- Evaluasi: **mAP@0.5, mAP@0.5:0.95, Precision, Recall**, ukuran kecil vs sedang, dan FPS pada hardware uji.
- Validasi silang 3-run (seed berbeda) untuk stabilitas angka.

### 8.3 Tahap 3 — OCR
- Stream: crop box detektor (+ padding 5–10 px) → preprocessing → engine OCR → post-processing.
- Engine yang dibandingkan (minimal 3):
  1. **Tesseract 5** (LSTM, whitelist `[A-Z0-9 ]`)
  2. **EasyOCR** (domain-agnostic, dukung multi-bahasa)
  3. **PaddleOCR** (PP-OCRv4/v5, det+rec terlatih kuat teks Latin)
  - (Opsional/ekstensi) **CRNN/SVTR custom** dilatih pada subset berlabel teks — bila waktu dan GPU cukup.
- Preprocessing (faktoral kecil, diuji per ablation): grayscale, CLAHE, adaptive threshold/Otsu, upscale 2×, deskew (deteksi sudut garis tepi plat), biner-inversi (plat gelap↔terang).
- Post-processing: normalisasi karakter (O→0, I→1 bila lolos whitelist), regex pola plat Indonesia `^[A-Z]{1,2} ?[0-9]{1,4} ?[A-Z]{0,3}$`, skor confidence.

### 8.4 Tahap 4 — Evaluasi & Eksperimen
| Aspek | Metrik |
|---|---|
| Deteksi | mAP@0.5, mAP@0.5:0.95, P, R, F1, FPS |
| OCR teks | **Exact plate match (%)**, **Character Error Rate (CER)**, WER per-kata |
| End-to-end | Full-plate accuracy pada test set (deteksi benar ∧ OCR benar), latensi ms/frame, FPS sistem penuh |
| Kualitas sistem | Curva recall vs confidence-threshold; analisis error (occlusion, glare, blur, miring) |

Eksperimen terkunci:
- **E1:** matriks model YOLO (4 varian) → pilih detektor terbaik & tercepat.
- **E2:** matriks engine OCR × preprocessing (baseline tanpa preprocessing vs penuh).
- **E3:** ablation tiap komponen preprocessing.
- **E4:** pipeline end-to-end vs engine OCR all-in-one (PaddleOCR det+rec langsung pada citra penuh) — apakah tahap deteksi YOLO tetap bernilai.
- **E5:** analisis error kualitatif + (opsional) uji signifikansi perbedaan akurasi (McNemar/bootstrap CI).

### 8.5 Tools & Lingkungan
- Python 3.11, PyTorch, `ultralytics`, `opencv-python`, `pytesseract`, `easyocr`, `paddleocr`, `labelstudio/cvat`, `albumentations`, `matplotlib`.
- Hardware uji: [laptop/GPU kampus — sebutkan VRAM]; catat reproducibility (seed, versi, config YAML).

## 9. Jadwal Riset (±16 minggu)

| Minggu | Aktivitas | Deliverable |
|---|---|---|
| 1–2 | Studi literatur (ANPR, YOLO, OCR, plat Indonesia) | Bab 2 draft / catatan tinjauan |
| 2–3 | Audit data, gabung kelas, re-split | `data.yaml` baru + report split |
| 3–5 | Anotasi ground-truth teks plat | CSV transkripsi ±400 plat |
| 4–6 | Eksperimen E1: training & evaluasi YOLO | Tabel mAP/FPS 4 model |
| 6–8 | Eksperimen E2–E3: OCR × preprocessing | Matriks CER/EMR + ablation |
| 8–9 | Eksperimen E4: perbandingan pipeline | Laporan perbandingan |
| 9–10 | Analisis error, prototipe CLI sederhana | Galeri error + demo |
| 10–13 | Penulisan Bab 1–3 + hasil | Draft skripsi lengkap |
| 13–14 | Revisi & validasi angka (repro run) | Draft siap seminar |
| 15–16 | Finalisasi + persiapan presentasi | Proposal/laporan + slide |

## 10. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Tidak ada split val/test | Evaluasi tidak valid | Re-split manual terdokumentasi (Tahap 1) |
| Dua kelas duplikat | Model belajar 2 label sama | Gabungkan sejak awal |
| Tanpa label teks → tidak bisa ukur OCR | Kontribusi & eksperimen E2 terganggu | Transkripsi manual subset; bila mepet, pseudo-label PaddleOCR + verifikasi manusia sebagian |
| Kualitas crop rendah (res 640) | OCR turun | Upscale + uji beberapa resolusi; bila dataset sumber asli resolusi tinggi tersedia, ambil ulang export Roboflow tanpa resize |
| Distribusi citra ≠ video CCTV/ETLE | Generalisasi lemah | Batasi klaim (citra statis); dokumentasikan sebagai keterbatasan |
| GPU terbatas | 4 model × epochs lama | Prioritaskan varian `n` dulu; epochs dinamis early-stop |
| Dataset agregat berkualitas tidak seragam | Noise label | Sampling audit manual 5–10% |

## 11. Indikator Keberhasilan (target awal)

- Deteksi: mAP@0.5 ≥ 0.90 pada test set.
- OCR: Exact Plate Match ≥ 85% (engine terbaik, crop dari ground-truth box) dan ≥ 75% (crop dari box prediksi).
- End-to-end: Full-plate accuracy ≥ 70% pada test set; latensi < 300 ms/frame (GPU) / dokumentasikan pada CPU.
- Dokumen proposal lengkap: rumusan masalah, metodologi reproducible, jadwal terealisasi ≥ 80%.

## 12. Acuan Literatur Awal (starter — lengkapi dengan Google Scholar/SINTA)

1. Jocher, G. et al. — *Ultralytics YOLOv8 / YOLO11* (dokumentasi & repo resmi, 2023–2024).
2. Redmon, J. & Farhadi, A. — *YOLOv3: An Incremental Improvement*, arXiv:1804.02767.
3. Du, C. et al. — *PP-OCRv3: Front-End Modifications Toward Practical OCR Engine*, arXiv:2108.00067 (dan seri PP-OCRv4/v5).
4. Smith, R. — *An Overview of the Tesseract OCR Engine*, ICDAR 2007.
5. Young, A. et al. — *EasyOCR: A versatile OCR tool*, (repo JaidedAI).
6. Suvorov, R. et al. — *What Is Wrong with Neural Text Recognition? (CRNN)*, ICDAR Workshop 2022.
7. surveys ANPR: Li, X. et al. *ANPR: a survey of techniques and systems*, IEEE T-ITS 2022; An, P. et al. — *Towards Real-World ANPR*, arXiv:2104.08992.
8. Riset lokal: cari kata kunci `"automatic number plate recognition" Indonesia Tesseract/SVM/YOLO` di SINTA/Garuda — mis. studi ANPR plat Indo dengan SVM/CNN & Tesseract (beberapa dari Univ. Mataram/lain).
9. Korpus & format plat: PP No. 55/2012 (regulasi TNKB) untuk justifikasi pola regex validasi.

## 13. Langkah Praktis Berikutnya (aksi nyata)

1. [ ] Konfirmasi judul & batasan dengan dosen pembimbing.
2. [ ] Saya bisa langsung eksekusi Tahap 1 di project ini: gabung kelas, buat split 80/10/10 + `data.yaml` koreksi.
3. [ ] Setup environment: `pip install ultralytics opencv-python pytesseract easyocr paddleocr` (atau file requirements).
4. [ ] Mulai transkripsi ground-truth (bisa dibantu semi-otomatis dengan PaddleOCR).
