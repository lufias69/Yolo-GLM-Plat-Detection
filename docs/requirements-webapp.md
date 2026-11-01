# Dokumen Kebutuhan (Requirements) — ANPR Testing Webapp

| | |
|---|---|
| Proyek | Webapp pengujian ANPR plat Indonesia (YOLO + multi-engine OCR) |
| Versi dokumen | 1.0 — 2026-11-01 |
| Status | Implementasi berjalan (commit `f886dda`), dokumen merefleksikan sistem saat ini |
| Tautan | Cara pasang: [`webapp/INSTALL.md`](../webapp/INSTALL.md) · Panduan pemakaian: [`webapp/README.md`](../webapp/README.md) |

## 1. Tujuan & ruang lingkup

Aplikasi web lokal untuk **menguji dan melaporkan akurasi pipeline ANPR**
(detektor YOLO + 4 engine OCR: easyocr, paddleocr, tesseract, glm-ocr) terhadap
foto plat nomor Indonesia dunia nyata, dengan penilaian manual (ground truth)
per foto, laporan metrik otomatis, analisis perbandingan engine, dan uji
konsistensi inferensi berulang.

**Di luar lingkup:** autentikasi multi-user, deployment publik, training model,
pengumpulan metadata kondisi citra (sudut/cahaya/blur/oklusi).

**Pengguna:** peneliti/skripsi tunggal di mesin lokal (Windows/Linux).

## 2. Kebutuhan fungsional

### FR-1 Manajemen batch & foto
- FR-1.1 Buat, lihat, dan tutup batch (`aktif`/`selesai`) dengan nama & deskripsi.
- FR-1.2 Unggah banyak foto sekaligus (drag & drop); format gambar umum; file
  tervalidasi (magic byte), ukuran & dimensi tersimpan.
- FR-1.3 Setiap foto diproses otomatis oleh worker background: rotasi 4 orientasi
  (0/90/180/270) → deteksi YOLO → OCR semua engine pada crop target; hasil
  (box, confidence, teks, fragment, latensi, rotasi terpilih) disimpan di
  `Photo.system_result` (JSON) beserta gambar beranotasi.
- FR-1.4 Status foto: menunggu / diproses / selesai / gagal (dengan pesan error);
  dapat diproses ulang (menandai anotasi lama `stale`).
- FR-1.5 Foto boleh ditandai tidak relevan (bukan motor/plat).

### FR-2 Penilaian manual (ground truth)
- FR-2.1 Per foto: ada/tidaknya plat (ADA/TIDAK_ADA), keberhasilan deteksi
  (BENAR/GAGAL/TIDAK_RELEVAN), dan pemilihan box target bila >1 deteksi.
- FR-2.2 Verdict OCR **per engine** (Benar/Salah) + jumlah karakter salah
  (otomatis dari Levenshtein vs GT, dapat dioverride) + catatan per engine.
- FR-2.3 Teks GT wajib bila ada verdict SALAH; validasi di form dan `full_clean`
  model (tidak bisa dilewati lewat pembuatan langsung).
- FR-2.4 Saat menyimpan, similarity & rasio (−spasi dan +spasi) GT vs tiap engine
  dihitung dan disimpan permanen di `EngineJudgment`.

### FR-3 Laporan batch
- FR-3.1 Metrik deteksi: accuracy/precision/recall/F1 + confusion matrix TP/FP/FN/TN.
- FR-3.2 AUC-ROC dari sweep confidence (butuh foto GT ADA dan TANPA plat).
- FR-3.3 Metrik OCR (engine utama): exact match, CER, char accuracy; latensi
  (mean/median/p95/FPS); jumlah foto belum dinilai / stale selalu tampil.
- FR-3.4 Tabel perbandingan engine OCR dari penilaian manual.

### FR-4 Laporan & analisis OCR per engine
- FR-4.1 Halaman laporan OCR: GT vs bacaan semua engine per foto, similarity/ratio.
- FR-4.2 Halaman **analisis similarity** (`/batch/<id>/report/ocr/analysis/`)
  dengan bagian bernomor sesuai kerangka analisis riset:
  1. Perbandingan akurasi antar engine: exact %, CER, char accuracy, mean
     similarity −spasi vs +spasi, distribusi bin similarity (−/+spasi).
  2. Error pattern: heatmap konusi karakter GT→OCR (backtrace Levenshtein,
     termasuk karakter hilang), posisi salah (depan/tengah/belakang),
     panjang plat vs jumlah karakter salah, daftar "SALAH tapi similarity ≥ 80%".
  3. Pengaruh kondisi citra — **ditangguhkan**: metadata tidak direkam (dipasang
     keterangan eksplisit di halaman).
  4. End-to-end per engine: sukses (deteksi benar DAN OCR benar) vs "deteksi
     benar tapi OCR salah" vs "gagal deteksi".
  5. Engine terbaik per skenario: exact % pada foto sulit (ada engine lain yang
     benar pada foto sama) dan tingkat over-confidence (conf ≥ 0,8 tapi SALAH).
  6. Statistik inferensial: CI 95 % Wilson exact match per engine; uji McNemar
     exact berpasangan antar engine per foto dengan p-value + penanda signifikan.
  7. Visualisasi: bar exact %, box plot similarity, heatmap, scatter
     confidence vs similarity, distribusi similarity dengan toggle −/+spasi,
     label angka di atas batang.

### FR-5 Export
- FR-5.1 Export laporan batch dan laporan OCR ke CSV (`;` + BOM UTF-8) dan XLSX,
  lengkap dengan kolom per engine (bacaan/nilai/karakter salah/catatan/sim/ratio).

### FR-6 Manajemen model YOLO
- FR-6.1 Halaman /models/: upload `.pt` (validasi magic-byte), aktifkan tepat
  satu, hapus dengan guard (foto lama tetap menunjuk model aslinya via
  `Photo.model`).
- FR-6.2 Bobot riset default terdaftar otomatis (`YoloModel.ensure_default`).

### FR-7 Uji konsistensi
- FR-7.1 Hanya untuk foto selesai + sudah dinilai + minimal terdeteksi sekali di
  threshold operasional (baseline box bermakna).
- FR-7.2 N inferensi (default 100, seed tetap) dengan augmentasi acak ringan per
  run; ringkasan: terdeteksi X/N, teks sama baseline Y/N, teks sama GT Z/N +
  sebaran; dapat dibatalkan; status queued/running/done/error/cancelled.

### FR-8 Uji cepat
- FR-8.1 `/uji-cepat/`: satu gambar → inference model aktif → tampilkan kotak +
  OCR + latensi; **tidak** masuk batch/penilaian/laporan.

## 3. Kebutuhan non-fungsional

### NFR-1 Kinerja
- Inference per foto: 4× pass deteksi + 1× pass OCR multi-engine; berjalan di
  thread background, UI tidak memblokir (auto-refresh).
- Latensi tercatat per foto dan diagregasi laporan (p95, FPS).

### NFR-2 Keandalan
- Worker tahan file tidak terbaca (pesan error nyata, status `error`, tidak
  menghentikan antrean); `save()` robust.
- SQLite WAL + busy timeout; singleton worker thread-safe.
- Reproses tidak membocorkan file anotasi (nama stabil berbasis hash).

### NFR-3 Integritas data
- Penilaian `stale` (hasil lama setelah reproses) tidak pernah masuk metrik;
  nilai hilang → `None`/N/A, bukan 0 yang menyesatkan.
- GT & output OCR dinormalisasi (`anpr.postprocessing.normalize_plate_text`).

### NFR-4 Keamanan
- Tanpa autentikasi — **hanya untuk mesin lokal**; dilarang expose ke jaringan
  publik (didokumentasikan di README & INSTALL).
- Validasi upload (magic byte), escaping template (confirm XSS-safe).

### NFR-5 Portabilitas & ketergantungan
- Python 3.11/3.12; Windows/Linux. Stack: Django ≥5.2, openpyxl, ultralytics,
  torch, opencv, numpy, pillow, easyocr, paddleocr+paddlepaddle, pytesseract
  (binary eksternal), requests (glm-ocr via Ollama). Daftar pin:
  `webapp/requirements.txt`.
- Aset frontend (Bootstrap, Chart.js) dilokalkan — tanpa CDN, jalan offline.

### NFR-6 Testabilitas
- Suite Django `manage.py test testing`: **111 test** (model, form, service
  metrik/analisis/export/inference, view, konsistensi) — harus hijau sebelum
  commit; logika analisis baru dibuat TDD (test RED→GREEN).

### NFR-7 Dokumentasi
- `webapp/README.md`: alur pemakaian, konfigurasi, definisi metrik, batasan.
- `webapp/INSTALL.md`: cara install step-by-step + troubleshooting.
- `docs/`: dokumen perencanaan & catatan fitur riset.

## 4. Batasan & asumsi

- 1 foto diasumsikan 1 motor (satuan analisis = foto).
- Ground truth = penilaian tunggal pengguna; akurasi end-to-end adalah batas
  bawah selama GT belum diverifikasi ganda.
- Engine glm-ocr (VLM) tidak memberi confidence numerik → pemilih orientasi
  memakai skor bentuk plat.
- Analisis "pengaruh kondisi citra" (FR-4.2 poin 3) butuh fitur rekaman metadata
  baru — di luar versi ini.

## 5. Kriteria penerimaan (ringkas)

1. `pip install -r webapp/requirements.txt && manage.py migrate && runserver`
   berhasil dari clone bersih (lihat INSTALL.md §2–5).
2. Alur upload → proses → nilai → laporan → analisis → export berjalan end-to-end.
3. `manage.py test testing` hijau 111/111.
4. Halaman analisis menampilkan bagian Analisis 1–7 (3 dengan keterangan
   tidak tersedia) beserta grafik berlabel angka.
