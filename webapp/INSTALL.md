# Cara Install — ANPR Testing Webapp

Panduan memasang aplikasi web pengujian ANPR (`webapp/`) dari nol di Windows
(linux serupa). Prasyarat keras: **Python 3.11+** dan ruang disk ±5 GB
(bobot model + dependensi ML).

## 1. Prasyarat

| Komponen | Versi teruji | Catatan |
|---|---|---|
| Python | 3.11 / 3.12 | 64-bit |
| GPU (opsional) | NVIDIA CUDA 12.x | tanpa GPU tetap jalan (CPU, lebih lambat) |
| Tesseract (opsional) | 5.x | hanya bila memakai engine `tesseract`; tambahkan ke PATH |
| Ollama (opsional) | — | hanya bila memakai engine `glm-ocr` (`ollama serve` + model `glm-ocr:q8_0`) |
| Browser | Chrome/Firefox/Edge modern | UI memakai Bootstrap + Chart.js lokal (tanpa CDN) |

## 2. Siapkan environment

Dari folder project (`C:\project\nuraeni`):

```bat
:: buat virtual environment
python -m venv venv

:: aktifkan
venv\Scripts\activate

:: (opsional, GPU) install torch CUDA sebelum ultralytics — sesuaikan versi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. Install dependensi

```bat
pip install -r webapp\requirements.txt
```

Alternatif (satu environment dengan skrip riset, seperti setup pengembangan):

```bat
pip install -r requirements.txt
```

`requirements.txt` root berisi semua di atas + pandas/matplotlib/jupyter untuk
skrip riset. Webapp **tidak** butuh folder `venv/` berisi torch bila hanya
menjalankan halaman non-inference, tetapi `runserver` yang memproses foto
memerlukan ultralytics + satu OCR engine.

## 4. Letakkan bobot model YOLO

Default settings (`webapp/anpr_web/settings.py`, `ANPR['WEIGHTS']`) menunjuk ke:

```
runs\detect\plate_detector_yolov8n_gpu_final\weights\best.pt
```

Bila bobot riset tidak ada, taruh file `.pt` hasil training di path tersebut,
atau ubah `ANPR['WEIGHTS']`. Setelah webapp jalan, model tambahan bisa
di-upload lewat halaman **/models/** (upload .pt → aktifkan; foto baru memakai
model aktif).

## 5. Migrasi database & jalankan

```bat
cd webapp
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Buka **http://127.0.0.1:8000/**. Database SQLite dibuat otomatis di
`webapp/db.sqlite3` (sudah ada bila clone repo ini). Media (foto upload &
hasil anotasi) tersimpan di `webapp/media/`.

Ingin data awal contoh? Buat batch dari UI, unggah foto plat, worker background
langsung memproses (deteksi 4 rotasi + 4 engine OCR).

## 6. Verifikasi instalasi

```bat
cd webapp
python manage.py test testing
```

Seluruh suite (111 test) harus hijau. Cek manual:

1. `/` — daftar batch tampil.
2. Buat batch → unggah 1 foto → status berubah `Selesai` (inference pertama bisa
   lama: EasyOCR/PaddleOCR mengunduh bobot pertamanya saat dipakai pertama kali).
3. Klik *Nilai* → isi GT → simpan.
4. `/batch/<id>/report/ocr/analysis/` — grafik analisis muncul.

## 7. Konfigurasi engine OCR

`webapp/anpr_web/settings.py` → dict `ANPR`:

- `OCR_ENGINE`: `'easyocr'` (default) | `'paddleocr'` | `'tesseract'` | `'glm-ocr'`
- `OCR_GPU`: `True` bila EasyOCR boleh pakai CUDA
- `CONF_OPERATIONAL` (0.25): ambang "sistem mendeteksi"
- Engine lain tetap diekspor/dibandingkan lewat `ocr_all` (semua engine jalan
  per foto tersimpan); `tesseract`/`glm-ocr` butuh dependensi eksternal di §1.

## 8. Troubleshooting

| Gejala | Penyebab / solusi |
|---|---|
| `ModuleNotFoundError: torch` saat proses foto | environment tanpa torch — install torch (§2) atau pakai env riset `mace_md` |
| `GLM-OCR connection refused` | Ollama mati — jalankan `ollama serve`, tarik model `glm-ocr:q8_0` |
| Tesseract hasil kosong | binary belum di PATH — install Tesseract 5 + language pack `eng` |
| Inference pertama sangat lambat | EasyOCR/PaddleOCR mengunduh bobot model sekali di awal |
| Foto status `Gagal` permanen | buka detail foto, lihat `error_message`; klik *Proses ulang* setelah perbaikan |
| Port 8000 bentrok | `python manage.py runserver 127.0.0.1:8001` |

## 9. Keamanan (penting)

Aplikasi **tanpa autentikasi** — dirancang untuk mesin lokal. Jangan
mengekspose `runserver` / port ini ke jaringan publik. Untuk pemakaian multi-user
di LAN, taruh di belakang reverse proxy dengan autentikasi sendiri.
