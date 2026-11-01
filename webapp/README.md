# ANPR Testing Webapp (Django)

Aplikasi web lokal untuk menguji model ANPR (detektor YOLOv8n + EasyOCR) terhadap
foto dunia nyata per batch, dengan penilaian manual, laporan metrik, dan uji
konsistensi 100×.

## Menjalankan

Instalasi dari nol (environment, dependensi, bobot model): lihat [`INSTALL.md`](INSTALL.md).
Daftar kebutuhan perangkat lunak: [`docs/requirements-webapp.md`](../docs/requirements-webapp.md).

Menggunakan environment riset (bukan folder `venv/` — venv tidak berisi torch):

```bat
C:\Users\lufias\miniforge3\envs\mace_md\python.exe webapp\manage.py migrate
C:\Users\lufias\miniforge3\envs\mace_md\python.exe webapp\manage.py runserver 127.0.0.1:8000
```

Buka http://127.0.0.1:8000/ . Test: `... webapp\manage.py test testing`.

## Alur pemakaian

1. Buat batch → unggah foto (drag & drop boleh banyak).
2. Foto diproses otomatis (deteksi + OCR) di thread background; halaman auto-refresh.
3. Klik "Nilai": tandai ada/tidaknya plat, keberhasilan deteksi, dan **verdict OCR per engine**
   (easyocr/paddleocr/tesseract/glm-ocr — masing-masing Benar/Salah; jumlah karakter salah
   dihitung otomatis dari "teks plat seharusnya" dan bisa dioverride, plus catatan per engine).
   Bila >1 deteksi, pilih box target lebih dulu.
4. "Laporan batch": accuracy/precision/recall/F1, confusion matrix, AUC-ROC,
   exact match, char accuracy, CER, latensi; export CSV/Excel.
   Laporan juga memuat **tabel perbandingan engine OCR** (exact match & char accuracy per
   engine dari penilaian manual); export menambah kolom nilai_/slh_karakter_/catatan_<engine>.
5. "Laporan OCR" (`/batch/<id>/report/ocr/`) menampilkan GT vs bacaan semua engine per foto,
   dengan similarity & ratio per engine (tersimpan di `EngineJudgment.sim_nospasi/sim_spasi/
   ratio_nospasi/ratio_spasi` — dihitung saat menyimpan form penilaian). Export CSV/Excel
   laporan OCR tersedia di halaman ini.
6. **"Analisis similarity OCR"** (`/batch/<id>/report/ocr/analysis/`) — jawaban "kalau OCR
   salah, seberapa salah?". Bagian bernomor sesuai kerangka analisis:
   - **Analisis 1** — exact match %, CER, char accuracy, mean similarity −spasi vs +spasi
     per engine; distribusi similarity (bin 0–25/…/100%) untuk kedua varian.
   - **Analisis 2** — pola kesalahan: heatmap konusi karakter GT→OCR (backtrace Levenshtein,
     '−' = karakter hilang), posisi salah (depan/tengah/belakang), panjang plat vs jumlah
     karakter salah, daftar "SALAH tapi similarity ≥ 80%".
   - **Analisis 4** — end-to-end per engine (deteksi benar DAN OCR benar), dipisah dari
     "deteksi benar tapi OCR salah" dan "gagal deteksi".
   - **Analisis 5** — ketahanan di "foto sulit" (foto yang ada engine lain yang benar) dan
     tingkat over-confidence (conf OCR ≥ 0,8 tapi dinilai SALAH).
   - **Analisis 6** — inferensial: confidence interval 95% (Wilson) exact match per engine;
     uji McNemar exact berpasangan antar engine per foto (p-value + penanda signifikan).
   - **Analisis 7** — visualisasi Chart.js: bar exact %, box plot similarity, heatmap,
     scatter confidence vs similarity, distribusi similarity dengan toggle −spasi/+spasi;
     angka tampil di atas batang.
   - **Analisis 3** (pengaruh kondisi citra: sudut/cahaya/blur/oklusi) belum tersedia —
     metadata tersebut tidak direkam oleh pipeline.
7. "Uji konsistensi…" (hanya untuk foto yang sudah dinilai DAN minimal terdeteksi
   sekali di threshold operasional — baseline box dibutuhkan): N inferensi (default
   100) dengan augmentasi acak ringan per run; hasilnya: "terdeteksi X dari N run",
   "teks sama dengan baseline Y dari N", "teks sama dengan GT Z dari N" + sebaran.

## Uji cepat (`/uji-cepat/`)

Satu gambar → inference langsung dengan model aktif; hasil (kotak + OCR + latensi)
tampil di halaman dan **tidak** masuk batch/penilaian/laporan. Untuk cek cepat foto
sebelum pengujian seri.

## Model YOLO ganda (halaman /models/)

Upload file `.pt` (bobot Ultralytics YOLO hasil training), aktifkan satu model.
Foto baru memakai model aktif; tiap foto mencatat model aslinya (tampil di laporan &
export, kolom `model`). Uji konsistensi memakai model yang sama dengan saat foto
diproses. Model `best.pt` bawaan riset terdaftar otomatis sebagai model aktif awal.
`ANPR['WEIGHTS']` di settings hanya menjadi sumber model default saat pertama kali jalan.

## Konfigurasi (`webapp/anpr_web/settings.py`, dict `ANPR`)

- `WEIGHTS`: path .pt (default hasil training riset).
- `CONF_OPERATIONAL` (0.25): threshold yang menentukan "sistem mendeteksi" & display.
- `CONF_STORED` (0.01): deteksi disimpan pada ambang ini supaya kurva ROC dapat direproduksi.
- `OCR_ENGINE`/`OCR_GPU`/`OCR_PREPROCESS`: identik dengan script riset
  (EasyOCR CPU, CLAHE + upscale).
- `OCR_ENGINE: 'glm-ocr'` memakai GLM-OCR via Ollama lokal (`glm-ocr:q8_0`,
  server harus hidup). Karena VLM tidak memberi confidence, pemilih orientasi
  memakai skor "bentuk plat" (`anpr.ocr_engines.plate_shape_score`).

## Rotasi otomatis

Setiap foto diuji pada 4 orientasi (0/90/180/270); orientasi dengan confidence
terbaik dipakai (deteksi + OCR pada frame terorientasi, box dipetakan balik ke
koordinat foto asli). Foto plat miring/terbalik tetap terbaca. Orientasi terpilih
disimpan di `system_result.rotation` dan tampil sebagai badge di halaman foto &
uji cepat. Biaya: 4× pass deteksi per foto (OCR tetap 1×).

## Definisi metrik

- Satuan analisis = foto (1 foto diasumsikan 1 motor).
- TP = GT "ADA" & terdeteksi & dinilai BENAR; FN = GT "ADA" & (tak terdeteksi atau dinilai GAGAL);
  FP = GT "TANPA" & terdeteksi; TN = GT "TANPA" & tidak terdeteksi.
- AUC-ROC: sweep threshold confidence atas box target (positif=TP; negatif=FP/TN/FN(score 0)).
- OCR (subset foto TP yang dinilai OCR): exact match; CER = Levenshtein/len(GT)
  (fungsi `cer()` dari `scripts/evaluate_ocr.py`); char accuracy = 1 − distance/len(GT).
- Teks GT dan keluaran OCR dinormalisasi dengan `anpr.postprocessing.normalize_plate_text`.
- Foto belum dinilai / penilaian STALE tidak masuk perhitungan; jumlahnya selalu tampil.

## Batasan

- Ground truth = penilaian tunggal Anda; akurasi OCR end-to-end adalah batas bawah
  selama GT belum diverifikasi saksama.
- Tanpa autentikasi — jalankan hanya di mesin lokal, jangan expose ke jaringan publik.
- AUC-ROC butuh foto ber-GT ADA **dan** TANPA plat; bila salah satu absen → N/A.
