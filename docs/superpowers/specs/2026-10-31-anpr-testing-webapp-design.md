# Desain Aplikasi Web Pengujian ANPR (Django)

Tanggal: 2026-10-31
Status: disetujui pemilik proyek (2026-10-31; keputusan 1A/2A dari rekomendasi builder diterima via "lanjutkan")

## 1. Intent & Why

Model ANPR (detektor YOLOv8n + OCR EasyOCR) sudah dilatih pada dataset
`gabungan-uamje` dan sudah dievaluasi pada test set dataset. Pemilik proyek
(periset, bekerja sendiri) kini perlu menguji model terhadap **foto dunia
nyata** yang ia ambil sendiri, secara terstruktur per batch, dengan penilaian
manual sebagai ground truth, dan laporan metrik siap-sidang.

Aplikasi ini adalah alat pengujian riset, bukan produk produksi. Keberhasilan
diukur dari: pengujian 100+ foto per batch terasa cepat, penilaian manual
nyaman, dan laporan metrik akhir bisa langsung dipakai di bab hasil skripsi.

## 2. Goals

1. Kelola batch pengujian dan upload foto real satu per satu (atau drag beberapa
   sekaligus), tiap foto langsung diproses (deteksi + OCR) saat diunggah.
2. Tampilkan hasil per foto: foto dengan kotak deteksi + confidence, teks hasil
   OCR, dan peringatan bila terdeteksi lebih dari 1 plat (asumsi riset: 1 foto =
   1 motor).
3. Kumpulkan penilaian manual per foto dari user (satu-satunya penilai):
   - Label ground truth kehadiran: "di foto ini ada plat / tidak ada plat".
   - Verdict deteksi: sistem benar / gagal mendeteksi.
   - Verdict OCR: karakter terbaca benar / salah; bila salah, user mengetik teks
     plat seharusnya (ini menjadi ground truth OCR).
   - Catatan bebas (teks).
4. Hitung dan tampilkan laporan per batch: accuracy, precision, recall, F1,
   confusion matrix, kurva AUC-ROC (deteksi), exact match & CER (OCR), latensi
   rata-rata per foto, dan rekap per foto. Export CSV dan Excel (.xlsx).
5. Uji konsistensi: pilih satu foto dari batch yang sudah dinilai → jalankan
   100× inferensi dengan augmentasi acak ringan per run → laporkan tingkat
   konsistensi (deteksi, teks OCR, sebaran confidence) dengan grafik.
6. Semua berjalan lokal di mesin user memakai GPU yang sama dengan training
   (RTX 5060 Ti), memakai `best.pt` yang sudah ada dan engine OCR EasyOCR dengan
   preprocessing CLAHE + upscale (ablation terbaik pada riset).

## 3. Non-Goals

- Multi-user, login, role, dan audit trail (penilai hanya pemilik proyek).
- Pengujian video / real-time streaming; hanya file gambar.
- Training ulang model, tuning threshold model, atau perubahan pipeline
  deteksi/OCR di luar yang dipakai riset saat ini.
- Deploy ke internet; aplikasi dijalankan lokal (`runserver` / gunicorn nanti).
- Manajemen dataset COCO/YOLO lama — ini hanya untuk foto real baru.

## 4. Anti-Goals (kegagalan meski teknis jalan)

- Laporan metrik yang tidak bisa direproduksi dari data tersimpan (setiap angka
  harus bisa ditelusuri ke foto + penilaian yang menyimpannya).
- Alur penilaian yang memaksa user membuka dua jendela per foto (upload lalu
  nilai terpisah) — proses dan penilaian harus satu layar per foto.
- Metrik yang diam-diam mengasumsikan ground truth yang belum diisi (foto tanpa
  penilaian lengkap tidak masuk perhitungan, dan laporan menampilkan jumlah foto
  yang belum dinilai).

## 5. Constraints

- Bahasa/teknologi: **Django** (diminta user). Python 3.x pada venv existing
  `C:/project/nuraeni/venv`; tambah `django`, `openpyxl` (export Excel);
  `ultralytics`, `easyocr`, `opencv`, `torch`, `pillow` sudah terpasang.
- Windows 11, path proyek `C:/project/nuraeni`. Aplikasi dibuat sebagai subfolder
  `webapp/` di proyek ini agar bisa memakai `runs/detect/.../weights/best.pt`
  dan modul `anpr/` (preprocessing, ocr_engines) yang sudah ada.
- Database: SQLite (cukup untuk satu peneliti; mudah di-backup).
- Foto disimpan di filesystem (`webapp/media/`), meta + penilaian di DB.
- OCR EasyOCR lambat pada load pertama; model YOLO + EasyOCR di-load sekali per
  proses (singleton), bukan per request.
- 100× inferensi konsistensi harus berjalan di belakang layar (background job
  sederhana berbasis thread + status polling), tidak memblokir UI.

## 6. Keputusan "how" yang sudah diambil user

- Django sebagai framework; fungsi utama: pengujian model terhadap data real.
- Penilai tunggal = pemilik proyek; tanpa autentikasi multi-user.
- 1 foto diasumsikan 1 motor; bila sistem mendeteksi >1 plat, tampilkan
  peringatan, user tetap menilai terhadap satu objek target.
- Uji konsistensi dipilih dari foto yang **sudah punya ground truth** dari batch
  sebelumnya; jalan 100× dengan variasi/augmentasi acak ringan; laporan dalam
  bentuk kalimat sederhana + grafik (keputusan teknis didelegasikan ke builder).
- Laporan harus memuat metrik pengujian lengkap (accuracy, precision, recall,
  F1, AUC-ROC, dan metrik pendukung lain); detail definisi didelegasikan ke
  builder dengan prinsip di §7.

## 7. Struktur Aplikasi

### 7.1 Model data (Django app `testing`)

- **TestBatch**: nama, deskripsi, tanggal, catatan; status (aktif/selesai).
- **Photo**: FK batch, file gambar, thumbnail, ukuran asli, metadata (nama file
  asli, waktu upload), status pemrosesan (queued/processing/done/error), hasil
  mentah sistem disimpan sebagai JSON (daftar deteksi: box, confidence; teks
  OCR per deteksi; latensi ms deteksi + OCR), gambar ber-kotak hasil render.
- **Annotation** (1-1 dengan Photo): 
  - `gt_presence`: ADA_PLAT / TANPA_PLAT (ground truth kehadiran oleh user).
  - `detected_ok`: BENAR / GAGAL / TIDAK_RELEVAN (penilaian user atas hasil
    deteksi sistem pada objek target).
  - `ocr_ok`: BENAR / SALAH / TAK_TERDETEKSI / TANPA_PLAT (penilaian OCR).
  - `gt_text`: teks plat seharusnya (wajib diisi bila `ocr_ok` = SALAH).
  - `note`: catatan bebas.
  - `multi_plate_warning_ack`: flag user sudah mengonfirmasi kasus >1 deteksi.
- **ConsistencyRun**: FK Photo (harus punya Annotation lengkap), N (default
  100), status (queued/running/done/error), hasil JSON per iterasi (detected,
  confidence, teks OCR, IoU box terhadap run baseline), ringkasan (jumlah terdeteksi,
  jumlah teks == baseline, jumlah teks == ground truth, mean/std confidence,
  tabel sebaran teks).

### 7.2 Halaman (URL sederhana, tanpa login)

1. **Dashboard** `/` — daftar batch (jumlah foto, % sudah dinilai, tombol
   laporan), form buat batch baru.
2. **Detail batch** `/batch/<id>/` — grid foto dengan status (belum diproses /
   menunggu penilaian / sudah dinilai), upload satu/banyak foto (drag & drop),
   tombol "Laporan batch", daftar foto yang bisa diklik untuk menilai.
3. **Penilaian foto** `/photo/<id>/` — satu layar: foto besar dengan kotak
   deteksi + confidence + teks OCR di bawah setiap box, badge peringatan bila
   >1 deteksi, panel penilaian (radio ground truth kehadiran, verdict deteksi,
   verdict OCR + input teks seharusnya, catatan), tombol simpan & lanjut ke foto
   berikutnya yang belum dinilai.
4. **Laporan batch** `/batch/<id>/report/` — kartu metrik + tabel + grafik
   (confusion matrix, kurva ROC, histogram confidence, sebaran hasil OCR), tombol
   export CSV/Excel, daftar per foto (termasuk penanda foto belum dinilai).
5. **Konsistensi** `/photo/<id>/consistency/` — form mulai (N default 100),
   progress bar selama job jalan, hasil: kalimat ringkasan, tabel sebaran teks
   OCR, grafik confidence, daftar 100 run (bisa di-expand). Daftar "foto siap
   uji konsistensi" (yang sudah ada ground truth) di dashboard batch.

### 7.3 Definisi metrik laporan (konvensi builder)

Satuan analisis utama = **foto** (karena 1 foto = 1 motor):

- Deteksi (dihitung atas foto yang `gt_presence`-nya terisi):
  - TP = foto ADA_PLAT & sistem mendeteksi ≥1 box (pada objek target,
    sesuai verdict `detected_ok`); FN = ADA_PLAT tapi sistem tidak
    mendeteksi/mendeteksi salah objek; FP = TANPA_PLAT tapi sistem mendeteksi;
    TN = TANPA_PLAT & sistem tidak mendeteksi.
  - Accuracy, Precision, Recall, F1 dari confusion matrix di atas.
  - **AUC-ROC**: kurva mengubah threshold confidence pada box terbaik tiap foto
    (score = confidence; positif = foto `gt_presence=ADA_PLAT` yang
    `detected_ok=BENAR`; negatif = TANPA_PLAT atau `detected_ok=GAGAL`).
    AUC dihitung dengan trapezoid (numpy) — ditampilkan bersama grafiknya.
  - mAP@0.5 pada foto real (dapat dihitung dari box vs verdict bila user
    menandai posisi salah; jika tidak cukup data, tampilkan "N/A" bukan angka
    menyesatkan).
- OCR (atas foto dengan `gt_text`/verdict OCR terisi):
  - Exact match = (jumlah `ocr_ok=BENAR`) / (jumlah foto dengan target plat
    terdeteksi & dinilai).
  - Character/Symbol accuracy per karakter terhadap `gt_text`.
  - CER rata-rata (Levenshtein / panjang GT) — reuse kode `scripts/` riset bila
    ada; kalau belum, implementasi standar.
- Kinerja: latensi rata-rata/median/p95 (deteksi, OCR, total), FPS ekuivalen.
- Rekap: jumlah foto, terdeteksi, belum dinilai; tabel CSV/Excel memuat semua
  kolom mentah (file, gt_presence, verdict, gt_text, hasil sistem, catatan).

### 7.4 Konsistensi (definisi operasional)

Tiap run dari N (default 100) menjalankan augmentasi acak ringan pada salinan
foto — kecerahan ±, kontras ±, pergeseran kecil, noise Gaussian kecil (menggunakan
transform torchvision/opencv; seed per run disimpan agar reproducible) — lalu
inferensi YOLO + OCR dengan preprocessing yang sama seperti di pipeline batch.
Baseline = hasil pemrosesan foto saat upload. Ringkasan yang dilaporkan:

- "Terdeteksi pada X dari N run" dan mean/std confidence.
- "Teks OCR sama persis dengan baseline pada Y dari N run".
- "Teks OCR sama persis dengan ground truth pada Z dari N run".
- Tabel sebaran semua teks unik yang muncul + grafik histogram confidence.

### 7.5 Teknologi pendukung

- Gambar ber-kotak: render dengan OpenCV/Pillow saat processing, simpan sebagai
  file turunan; halaman menampilkan hasil render.
- Grafik laporan: **Chart.js** via template (tanpa build tooling). Sederimbang:
  bila builder memilih server-side matplotlib ke PNG, boleh — yang penting
  muncul di halaman laporan dan ter-embed di export.
- Background job: thread daemon + status field di DB + polling fetch() sederhana.
  Celery/Redis sengaja tidak dipakai (kompleksitas Windows, kebutuhan lokal).
- Processing upload: juga memakai thread worker agar request upload tidak
  timeout; foto bisa di-batch-upload dan diproses berurutan.

## 7.6 Keputusan hasil review desain (disetujui user)

- **Konfigurasi inferensi = identik dengan script riset** (`evaluate_endtoend.py`,
  `inference_single_image.py`): EasyOCR di CPU, YOLO `conf=0.25`, `imgsz` default,
  preprocessing OCR CLAHE + upscale. Semua angka laporan & konsistensi
  dibandingkan terhadap hasil evaluasi lama dengan basis yang sama.
  Konsekuensi diterima: OCR ±1-2 detik/foto; job konsistensi 100× berjalan
  background dengan progress bar dan bisa dibatalkan.
- **Simpan mentahan deteksi di threshold sangat rendah (0,01)**: saat processing,
  YOLO dijalankan sekali dengan `conf=0.01`; semua box + confidence disimpan.
  Threshold operasional 0,25 hanya dipakai untuk display, penentuan "sistem
  mendeteksi" (TP/FN), dan verdict — sehingga kurva ROC tetap bisa direproduksi
  untuk threshold di bawah 0,25 sesuai anti-goal §4.
- **Pemilihan deteksi target**: saat >1 box di atas threshold 0,25, user WAJIB
  mengklik satu box target sebelum menilai; indeks box target disimpan di
  `Annotation.target_detection_index` dan menjadi acuan verdict deteksi, verdict
  OCR, teks baseline, serta baseline IoU pada ConsistencyRun. Untuk 1 deteksi,
  field diisi 0 otomatis.
- **mAP@0.5 tidak dihitung di aplikasi** (tidak ada ground truth kotak per foto);
  laporan memakai metrik per-foto §7.3. Tidak ada editor kotak GT.
- **Normalisasi teks**: `gt_text` yang diketik user dan teks OCR melewati
  `anpr.postprocessing.normalize_plate_text` yang sama seperti `scripts/` riset,
  sehingga CER/exact match dapat dibandingkan langsung dengan hasil evaluasi lama.
- **Reproses foto**: hasil sistem menyimpan `result_version`; bila foto
  diproses ulang, annotation lama ditandai `stale=True` (tidak ikut dihitung di
  laporan sampai user mengonfirmasi ulang).

## 8. Tinggal builder putuskan (decided by builder, marked as such)

- Nama Django app & layout folder internal; asumsi di atas: `webapp/` dengan app
  `testing`.
- Batas "augmentasi ringan" konkret (mis. brightness ±0.2, geser ±2%, noise σ±3).
- Presisi angka metrik & tampilan visual detail; gaya UI bersih, mudah dibaca,
  boleh pakai Django template + CSS sederhana / Bootstrap tanpa customization.
- Format kolom export CSV/Excel mengikuti tabel rekap §7.3.

## 9. Risiko & Mitigasi

- EasyOCR first-load lambat → load singleton saat startup + halaman "siap/sibuk".
- 100× OCR per foto bisa menit-scale → jalankan sebagai job background dengan
  progress; user bisa meninggalkan halaman.
- Ground truth user adalah satu-satunya sumber kebenaran → wajib tampilkan jumlah
  foto belum dinilai di laporan supaya angka tidak menyesatkan.
- AUC-ROC tidak terdefinisi bila semua foto "ADA_PLAT" (tanpa negatif) → tampilkan
  N/A dengan penjelasan, jangan angka palsu.
