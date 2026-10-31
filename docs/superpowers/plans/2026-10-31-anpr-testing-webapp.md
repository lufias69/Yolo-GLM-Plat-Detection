# ANPR Testing Webapp (Django) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aplikasi web Django lokal untuk menguji model ANPR (YOLOv8n + EasyOCR) terhadap foto real per batch, dengan penilaian manual, laporan metrik lengkap (accuracy/precision/recall/F1/AUC-ROC/exact-match/CER/latensi), dan uji konsistensi 100× dengan augmentasi acak.

**Architecture:** Django project `webapp/` (single app `testing`, SQLite, tanpa auth) yang menambah modul `anpr/` dan weights riset yang sudah ada. Upload foto memicu job thread (deteksi conf=0.01 disimpan, threshold operasional 0.25 untuk display/verdict, OCR per crop target). Metrik dihitung murni dari tabel Photo+Annotation dan bisa diekspor CSV/Excel. Konsistensi = job thread N=100 inferensi dengan augmentasi acak, status via polling.

**Tech Stack:** Python 3.12 (venv existing), Django (baru), openpyxl (baru), ultralytics, easyocr, opencv-python, numpy, Pillow, Chart.js (CDN, tanpa build step), Bootstrap 5 (CDN).

**Spec:** `docs/superpowers/specs/2026-10-31-anpr-testing-webapp-design.md`

## Global Constraints

- Windows 11; semua perintah python memakai `C:/project/nuraeni/venv/Scripts/python.exe`.
- Folder kerja aplikasi: `C:/project/nuraeni/webapp/`; reuse `C:/project/nuraeni/anpr/` via `sys.path.insert` (pola yang sama dengan `scripts/`).
- Konfigurasi inferensi WAJIB identik dengan riset: EasyOCR `gpu=False`, engine preprocessing `{'use_clahe': True, 'use_upscale': True, 'use_binarize': False, 'use_denoise': False}`, YOLO disimpan pada `conf=0.01`, threshold operasional `0.25`, weights default `runs/detect/plate_detector_yolov8n_gpu_final/weights/best.pt` (bisa diganti via settings).
- Teks: OCR output dan `gt_text`user dinormalisasi dengan `anpr.postprocessing.normalize_plate_text` sebelum dibandingkan; CER memakai `cer()` dari `scripts/evaluate_ocr.py` (import, jangan salin ulang).
- Tanpa login/auth. UI berbahasa Indonesia.
- Tidak ada metrik yang dihitung dari foto belum dinilai; laporan selalu menampilkan jumlah foto belum dinilai. mAP@0.5 tidak dihitung (keputusan §7.6).
- Database SQLite `webapp/db.sqlite3`; file foto di `webapp/media/`. `.gitignore` mengecualikan `venv/`, `runs/`, `train*/`, `valid/`, `test/`, `crops_test/`, `media/`, `db.sqlite3`, `__pycache__/`.
- Proyek ini bukan git repo: Task 1 melakukan `git init` di `C:/project/nuraeni`. Setiap commit diakhiri baris `Co-Authored-By: Claude Code <noreply@anthropic.com>`.

## Review Focus

Input/kondisi yang spesifikasikan diam tapi paling mungkin menjatuhkan aplikasi:

1. **File upload bukan gambar / gambar rusak** (mis. .pdf diganti ekstensi .jpg) → processing harus gagal rapi: `Photo.status='error'`, `error_message` terisi, halaman batch tetap tampil, tidak ada trace ke user. → Task 3.
2. **`ocr_ok=SALAH` tapi `gt_text` kosong** → form penilaian harus menolak simpan (invalidasi di model + form), karena ground truth OCR adalah alasan field itu ada. → Task 5.
3. **Batch dengan 0 foto dinilai / semua `detected_ok=TIDAK_RELEVAN`** → semua metrik `None`/`N/A`, bukan `ZeroDivisionError` dan bukan 0.0 menipu; AUC `None` bila tidak ada kelas negatif. → Task 6.
4. **Foto dengan >1 deteksi operasional lalu user simpan tanpa memilih box target** → ditolak dengan pesan jelas; hanya 1 deteksi → `target_detection_index=0` diisi otomatis. → Task 5.
5. **Diproses ulang setelah dinilai** → annotation lama ditandai `stale=True` dan keluar dari perhitungan metrik/konsistensi-baseline sampai user konfirmasi. → Task 3 + 6.

---

### Task 1: Scaffold Django + git init

**Files:**
- Create: git repo di `C:/project/nuraeni` (init + `.gitignore`)
- Create: `webapp/manage.py`, `webapp/anpr_web/` (`settings.py`, `urls.py`, `wsgi.py`, `asgi.py`), `webapp/testing/` (`__init__.py`, `apps.py`, `views.py`, `urls.py`, `migrations/__init__.py`), template `webapp/testing/templates/testing/base.html` + `testing/home.html`, `webapp/testing/tests/__init__.py`
- Modify: `requirements.txt` (tambah `django`, `openpyxl`)

**Interfaces:**
- Produces: project Django berjalan; `python webapp/manage.py test testing` executable; URL `/` → view `home`; `settings.ANPR = {'WEIGHTS': str(base/'runs/detect/plate_detector_yolov8n_gpu_final/weights/best.pt'), 'CONF_STORED': 0.01, 'CONF_OPERATIONAL': 0.25, 'OCR_PREPROCESS': {...}, 'MEDIA_ROOT': ...}` (dict konfigurasi dipakai Task 3+).

- [ ] **Step 1: git init + .gitignore.** `git init` di `C:/project/nuraeni`, tulis `.gitignore` sesuai Global Constraints, commit kosong `chore: init repo` (+ attribution line). Semua commit task berikutnya memakai konvensi `feat:`/`test:` + attribution.
- [ ] **Step 2: Install deps.** `venv/Scripts/python.exe -m pip install django openpyxl`; tambahkan baris `django` + versi dan `openpyxl` ke `requirements.txt`.
- [ ] **Step 3: Scaffold.** `django-admin startproject anpr_web webapp` (lalu jalankan via venv python), `startapp testing webapp/testing`. Settings: `INSTALLED_APPS += ['testing']`, `LANGUAGE_CODE='id'`, `TEMPLATES DIRS [webapp/testing/templates]`, SQLite default, `MEDIA_ROOT=BASE_DIR/'media'`, `MEDIA_URL='/media/'`, `ALLOWED_HOSTS=['*']` (lokal), `DEBUG=True`, tambahkan `settings.ANPR` dict (isi sesuai Global Constraints; akar proyek riset = `BASE_DIR.parent` karena `BASE_DIR` = folder `webapp/`). Root urls: include `testing.urls` + `static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)` di DEBUG. `testing/urls.py`: `path('', views.home, name='home')`. `home` render `testing/home.html` (extends `base.html`, Bootstrap+Chart.js CDN di base, judul "ANPR Testing").
- [ ] **Step 4: Migration + smoke test.** Tulis test: `Client().get('/')` → 200 dan `'ANPR' in response.content`. Run `manage.py migrate` dan `manage.py test testing` → PASS.
- [ ] **Step 5: Commit** `feat: django scaffold webapp`.

### Task 2: Models + admin

**Files:**
- Create: `webapp/testing/models.py`, `webapp/testing/admin.py`, `webapp/testing/tests/test_models.py`, migration
- Modify: settings (tidak ada)

**Interfaces:**
- Produces:
  - `TestBatch(name: str, description: str = '', status: str = 'aktif'|'selesai', created_at)`, `.photo_count()`, `.annotated_count()`.
  - `Photo(batch FK, image FileField(upload_to='photos/%Y%m/'), annotated_image FileField(null=True, upload_to='annotated/%Y%m/'), status: 'queued'|'processing'|'done'|'error', error_message: str = '', system_result: JSONField(default=dict), result_version: int = 1, original_name: str, width: int = 0, height: int = 0, uploaded_at, processed_at null)`. `system_result` skema final (diproduksi Task 3, dikonsumsi Task 5/6/9): `{"detections":[{"index":int,"box":[x1,y1,x2,y2],"confidence":float,"ocr_text":str,"ocr_confidence":float}], "operational_boxes":[int,...], "latency":{"detection_ms":float,"ocr_ms":float,"total_ms":float}, "threshold":0.25}`.
  - `Annotation(photo OneToOne related_name='annotation', gt_presence: 'ADA'|'TIDAK_ADA'|'' , detected_ok: 'BENAR'|'GAGAL'|'TIDAK_RELEVAN'|'', target_detection_index: int|null, ocr_ok: 'BENAR'|'SALAH'|'TAK_TERDETEKSI'|'TIDAK_RELEVAN'|'', gt_text: str = '', note: str = '', multi_plate_confirmed: bool = False, stale: bool = False, annotated_at)`. `Annotation.is_complete()` → bool ( presence+detected_ok+ocr_ok terisi, dan bila `ocr_ok=='SALAH'` maka `gt_text` non-kosong ). `Annotation.clean()` raise ValidationError bila `ocr_ok=='SALAH'` tanpa `gt_text` (Review Focus #2).
  - `ConsistencyRun(photo FK related_name='consistency_runs', n: int = 100, seed: int = 42, status: 'queued'|'running'|'done'|'error'|'cancelled', error_message='', summary: JSONField(default=dict), runs: JSONField(default=list), created_at, finished_at null)`. `.is_cancellable`/helper set `cancel_requested: bool = False` (BooleanField).

- [ ] **Step 1: Failing tests** di `test_models.py`: buat batch+photo → `annotated_count()==0`; buat Annotation lengkap → `is_complete() is True`; `Annotation(ocr_ok='SALAH', gt_text='')` → `full_clean()` raise ValidationError; `Annotation(ocr_ok='SALAH', gt_text='B 1234 ABC')` → valid.
- [ ] **Step 2: Run** `manage.py test testing.tests.test_models` → FAIL.
- [ ] **Step 3: Implement models + admin** (register keempat model, list_display seperlunya), `makemigrations testing`, jalankan migrate.
- [ ] **Step 4: Run test** → PASS. **Step 5: Commit** `feat: testing models`.

### Task 3: Inference service + worker thread

**Files:**
- Create: `webapp/testing/services/__init__.py`, `webapp/testing/services/inference.py`, `webapp/testing/services/workers.py`, `webapp/testing/views.py` (tambah endpoint status), `webapp/testing/urls.py` (tambah route), `webapp/testing/tests/test_inference.py`, `test_workers.py`
- Modify: `webapp/testing/models.py` (method `Photo.reprocess()` — set status queued, `result_version += 1`, tandai `annotation.stale = True` bila ada; Review Focus #5)

**Interfaces:**
- Consumes: `settings.ANPR`, `anpr.preprocessing.pipeline`, `anpr.ocr_engines.get_engine`, `anpr.postprocessing.normalize_plate_text`, `ultralytics.YOLO`.
- Produces:
  - `inference.get_service() -> InferenceService` (singleton modul-level).
  - `InferenceService.detect(image_bgr: np.ndarray) -> list[dict]` — YOLO pada `conf=CONF_STORED`, urutan desc-confidence, tiap item `{'index','box','confidence'}`.
  - `InferenceService.recognize_crop(image_bgr, box) -> tuple[str, float]` — crop → `get_engine('easyocr', gpu=False).recognize(crop, preprocess=settings.ANPR['OCR_PREPROCESS'])`.
  - `InferenceService.process_file(path: Path) -> dict` — baca gambar (gagal → raise `InferenceError(message)`), detect, filter box `confidence >= CONF_OPERATIONAL` sebagai `operational_boxes`, OCR hanya box operasional, time `detection_ms`/`ocr_ms`/`total_ms` (perf counter), normalisasi teks via `normalize_plate_text`, dan `render_annotated(image_bgr, detections) -> np.ndarray` (kotak hijau=operasional/abu=di-bawah-threshold, label `#i conf OCR:TEXT`) disimpan `annotated/<stem>_<version>.jpg`. Return skema `system_result` Task 2 + key `annotated_image` (rel path).
  - `workers.enqueue_photo(photo_id)` — antre ke satu `queue.Queue` yang dikonsumsi daemon thread tunggal (lazy-start); worker: `status='processing'` → `process_file` → simpan `system_result` + `annotated_image` + width/height + `processed_at`, `status='done'`; `InferenceError`/except → `status='error'`, `error_message` (Review Focus #1). `workers.start_worker()` dipanggil dari `apps.py` `ready()` HANYA saat `MANAGE_RUNTESTS` tidak di env dan bukan command `test` (guard `sys.argv`), agar test tidak memuat model.
  - API: `GET /api/photo/<id>/status/` → JSON `{'status','error_message'}` (view `photo_status`).

- [ ] **Step 1: Failing unit tests** (tanpa model/OCR asli — inject fake): class `FakeYOLO` (panggilan → objek hasil palsu dengan `.boxes.xyxy`, `.boxes.conf`) dan `FakeEngine` (`recognize` → `('b 1234 abc', 0.9)`). Test `process_file` pada gambar numpy temp: dua box (conf 0.5 dan 0.05) → `detections` berisi 2 (urut desc), `operational_boxes == [0]`, `ocr_text` ternormalisasi `'B1234ABC'`? **Catatan eksekutor:** `normalize_plate_text('b 1234 abc')` = `'B 1234 ABC'` (spasi dipertahankan `clean_text`); assert nilai sebenarnya dengan memanggil fungsi itu di test, jangan hardkarang. Assert: key lengkap sesuai skema, `latency.total_ms > 0`, annotated file dibuat. Test `InferenceError` untuk path tidak ada / file rusak (tulis bytes sampah `.jpg`) (Review Focus #1).
- [ ] **Step 2: Worker tests:** `enqueue_photo` pada Photo gambar valid (service di-monkeypatch dengan stub → dict hasil minimal) → poll sampai `status=='done'` (timeout 10s), `result_version==1`; lalu panggil `photo.reprocess()` → `result_version==2` dan `annotation.stale is True` (Review Focus #5); service stub raise `InferenceError` → `status=='error'` + pesan terisi.
- [ ] **Step 3: Run** → FAIL. **Step 4: Implement** `inference.py`, `workers.py`, `photo_status` view+url, `Photo.reprocess()`. `InferenceService` load lazy: atribut `_model`/`_engine` `None` sampai dipakai (test men-set stub sebelum akses). **Step 5: Run** → PASS.
- [ ] **Step 6: Sanity nyata (sekali):** script kecil ad-hoc memuat service asli + `inference_outputs`/test image dari folder `test/images`, pastikan `status='done'` dengan kotak OCR terisi; JANGAN commit artefak media hasil sanity. **Step 7: Commit** `feat: inference service and processing worker`.

### Task 4: Batch & upload UI

**Files:**
- Create: `webapp/testing/views.py` (tambah), `testing/urls.py` (tambah), templates `testing/batch_list.html`, `batch_detail.html`, partial `_photo_card.html`, tests `test_views_batch.py`
- Modify: `home.html` → redirect/landing berisi daftar batch

**Interfaces:**
- Consumes: `TestBatch`, `Photo`, `workers.enqueue_photo`.
- Produces:
  - `GET /` → daftar batch + form buat batch (POST `name`, `description` → redirect detail).
  - `GET /batch/<id>/` → grid foto: thumbnail/placeholder, badge status (`queued`/`processing`/`done`/`error`), badge penilaian (belum dinilai / dinilai / STALE), peringatan `>1 deteksi operasional`, link `/photo/<id>/` bila done.
  - `POST /batch/<id>/upload/` → field `files` (multi, drag&drop di JS kecil + `<input type=file multiple accept=image/*>`); validasi ekstensi `jpg/jpeg/png/webp/bmp`, simpan `Photo(original_name=nama asli, status='queued')`, `enqueue_photo(id)` tiap file.
  - `POST /photo/<id>/reprocess/` → `photo.reprocess(); enqueue_photo(id)`.
  - JS polling di `batch_detail.html`: tiap 2s fetch `/api/photo/<id>/status/` untuk foto status queued/processing, lalu `location.reload()` bila semua selesai.

- [ ] **Step 1: Failing view tests** (Django test client; worker di-monkeypatch agar `enqueue_photo` no-op): POST buat batch → 302, batch ada di DB; POST upload dengan 2 `SimpleUploadedFile` valid → 2 `Photo(status='queued')`, `original_name` benar; upload file `.txt` → ditolak (form error, photo tidak dibuat); `GET /batch/<id>/` → 200, memuat nama foto; POST reprocess → `result_version` naik + `enqueue_photo` terpanggil.
- [ ] **Step 2: FAIL → Step 3: Implement views/urls/templates → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: batch management and upload UI`.

### Task 5: Halaman penilaian (annotation)

**Files:**
- Create: `webapp/testing/forms.py` (`AnnotationForm`), views `photo_detail`/`annotate`/`next_unannotated` di `views.py`, url `/photo/<id>/`, `/photo/<id>/annotate/` (POST), `/batch/<id>/next-unannotated/`, template `testing/photo_detail.html`, test `test_views_annotate.py`

**Interfaces:**
- Consumes: `Photo.system_result` (skema Task 2/3), `Annotation`, `settings.ANPR['CONF_OPERATIONAL']`, `normalize_plate_text` (preview teks GT yang disimpan sudah ternormalisasi — simpan hasil normalisasi).
- Produces:
  - Halaman menampilkan `annotated_image` besar + daftar deteksi: radio/klik per box operasional → `target_detection_index`; badge kuning "Ditemukan N deteksi — 1 foto diasumsikan 1 motor, pilih box target" bila `len(operational_boxes) > 1` (user mencentang `multi_plate_confirmed` saat simpan, Review Focus #4).
  - Form: `gt_presence` (ADA/TIDAK_ADA), `detected_ok` (BENAR/GAGAL/TIDAK_RELEVAN), `ocr_ok` (BENAR/SALAH/TAK_TERDETEKSI/TIDAK_RELEVAN), `gt_text`, `note`. Tombol "Simpan" dan "Simpan & lanjut" (redirect `next-unannotated`; bila tidak ada → kembali ke batch).
  - Aturan validasi form (Review Focus #2 + #4): `ocr_ok == 'SALAH'` → `gt_text` wajib; `len(operational_boxes) > 1` → wajib pilih `target_detection_index` + `multi_plate_confirmed`; `gt_presence == 'TIDAK_ADA'` → field deteksi/OCR boleh `TIDAK_RELEVAN`. `is_complete()` di-set saat simpan; simpan → `Annotation.update_or_create`, `stale=False`.
  - Banner oranye bila `annotation.stale` di foto yang sudah dinilai + tombol proses ulang (reuse Task 4).
- [ ] **Step 1: Failing tests** — buat photo `status='done'` dengan `system_result` stub 2 box operasional: POST tanpa pilih target → form error, Annotation tidak dibuat; POST lengkap + target 0 + `ocr_ok='SALAH'` gt_text kosong → error; POST valid (gt_text `'b 1234 abc'`) → Annotation dibuat, `gt_text` tersimpan `'B 1234 ABC'`, `is_complete()` True, redirect ke next-unannotated; next-unannotated melewati foto yang annotation-nya `stale=True`? **Keputusan builder: foto stale TETAP muncul di antrian "lanjut" agar bisa dikonfirmasi ulang** — assert sesuai pilihan ini di test.
- [ ] **Step 2: FAIL → Step 3: Implement (form + views + template; JS sederhana highlight box target) → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: photo annotation flow`.

### Task 6: Service metrik (pure functions)

**Files:**
- Create: `webapp/testing/services/metrics.py`, `webapp/testing/tests/test_metrics.py`

**Interfaces:**
- Consumes: `Photo`+`Annotation` queryset, `system_result` skema Task 2, `from scripts.evaluate_ocr import cer` (via `sys.path.insert(0, BASE_DIR.parent)` di test; di service terima fungsi `cer` sebagai argumen/injeksi atau import lazy dengan path dari `settings.BASE_DIR.parent`), `normalize_plate_text`.
- Produces (semua angka `None` bila tidak terdefinisi, bukan 0):
  - `photo_outcome(photo) -> str` — klasifikasi per foto dinilai: `'TP'` (gt ADA, ada box operasional, `detected_ok=BENAR`), `'FN'` (gt ADA, tidak ada box operasional ATAU `detected_ok=GAGAL`), `'FP'` (gt TIDAK_ADA + ada box operasional), `'TN'` (gt TIDAK_ADA + tidak ada box operasional — termasuk bila user menandai `detected_ok='TIDAK_RELEVAN'` karena memang tidak ada yang benar untuk dinilai); `''`/None untuk foto belum dinilai atau `detected_ok='TIDAK_RELEVAN'` padahal ada box operasional. Aturan TP/FN/FP/TN ini dikunci eksplisit oleh test di bawah.
  - `detection_metrics(photos) -> dict` — `{'tp','fp','tn','fn','accuracy','precision','recall','f1'}` dari hanya foto `is_complete() and not stale`; `photo_score_label(photo) -> tuple[float, bool] | None` — score = conf box target (`target_detection_index`; bila tidak terdeteksi/absen → box operasional terbaik; gt TIDAK_ADA → conf box operasional terbaik), label positif = `photo_outcome in ('TP',)`? **Keputusan builder:** positif ROC = foto gt ADA dengan outcome TP (score conf box target), negatif = outcome FP (score conf) dan outcome FN (score 0.0) dan outcome TN (score conf box operasional, atau 0.0 bila tidak ada). Fotos ber-outcome `''` / TIDAK_RELEVAN tidak masuk.
  - `roc_auc(points: list[tuple[float,bool]]) -> dict | None` — sweep threshold unik desc, TPR/FPR per threshold, titik awal (0,0)/(1,1), AUC trapezoid numpy; `None` bila tanpa positif atau tanpa negatif (Review Focus #3). Return `{'auc': float, 'points': [{'fpr','tpr','threshold'},...]}`.
  - `ocr_metrics(photos) -> dict` — subset: outcome TP, `ocr_ok in ('BENAR','SALAH')`: `exact_match` = ratio BENAR; `cer_mean` = mean `cer(normalize(pred_text), normalize(gt_text))` dengan pred = `ocr_text` box target, gt = `gt_text` (BENAR ⇒ identik ⇒ CER 0); `char_accuracy` = mean per foto `max(0.0, 1 - distance/len(gt))` (distance via fungsi yang sama dengan cer: hitung Levenshtein mentah — ekspor helper `levenshtein_distance(pred, truth) -> int` dari metrics, cer riset = distance/len(gt)); `n_evaluated`.
  - `latency_metrics(photos_done) -> dict` — mean/median/p95 `latency.total_ms` (semua foto `status='done'`, tak perlu dinilai) + `fps = 1000/mean_ms`.
  - `batch_report(batch) -> dict` — gabung semuanya + `counts`: `{'total','annotated','complete','stale','unannotated'}`, `auc` dari `roc_auc`; semua kosong → dict bernilai `None` dengan `counts` tetap terisi.

- [ ] **Step 1: Failing tests** dengan fixture photo/annotation in-memory (SQLite TestCase):
  - Confusion: 2 TP, 1 FN (gt ADA, tanpa deteksi), 1 FP (gt TIDAK_ADA + deteksi), 1 TN (gt TIDAK_ADA, tanpa deteksi, `detected_ok='TIDAK_RELEVAN'`) → assert eksplisit `tp=2 fp=1 fn=1 tn=1, accuracy=0.6, precision=2/3, recall=0.5, f1=0.5`.
  - Foto `TIDAK_RELEVAN` dan foto belum dinilai tidak mengubah angka di atas.
  - ROC: 3 positif conf (0.9,0.7,0.5), 3 negatif (FP conf 0.6; TN conf 0.1; FN score 0.0) → `0.5 < auc <= 1.0`, titik kurva monotonic; semua-positif → `auc is None` (Review Focus #3).
  - OCR: gt `'B 1234 ABC'`, pred BENAR identik → cer 0; pred `'B 1234 ABD'` SALAH → cer 1/9 ±1e-9, char_accuracy 8/9.
  - Latency p95 dari 20 nilai known.
  - `batch_report` pada batch kosong → tidak exception, semua metrik None.
- [ ] **Step 2: FAIL → Step 3: Implement `metrics.py` → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: metrics service`.

### Task 7: Halaman laporan batch

**Files:**
- Create: `webapp/testing/views.py` (`batch_report_view`), url `/batch/<id>/report/`, template `testing/report.html`, test `test_views_report.py`

**Interfaces:**
- Consumes: `metrics.batch_report(batch)`.
- Produces: halaman laporan: baris peringatan "X dari Y foto belum dinilai — metrik hanya atas foto dinilai"; kartu metrik deteksi (acc/prec/rec/F1 + confusion matrix 2×2) ; kartu ROC (`auc` atau "N/A (butuh foto ADA dan TANPA PLAT keduanya)"), canvas Chart.js line FPR/TPR + tooltip threshold (data di-embed JSON di template); kartu OCR (exact match, char accuracy, CER) ; kartu latensi (mean/median/p95, FPS); tabel per-foto (file, gt_presence, outcome TP/FN/FP/TN, conf, teks OCR, gt_text, catatan, badge stale); link export (Task 8). Nilai `None` dirender "N/A".
- [ ] **Step 1: Failing test** — batch dengan fixture Task 6: GET 200; body memuat `95.15`? tidak — pakai angka fixture Task 6 (acc 60,0% dst.), assert string metrik, nama foto, dan `data-report="roc"`; batch kosong → 200 + "belum dinilai" tanpa crash.
- [ ] **Step 2: FAIL → Step 3: Implement → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: batch report page`.

### Task 8: Export CSV + Excel

**Files:**
- Create: `webapp/testing/services/export.py`, views `batch_export_csv`/`batch_export_xlsx` + urls, test `test_export.py`

**Interfaces:**
- Consumes: `Photo`, `Annotation`, `metrics.batch_report`.
- Produces:
  - `rows(batch) -> list[list[str]]` — header: `file, gt_presence, detected_ok, target_index, ocr_ok, gt_text, catatan, sistem:deteksi_ada, sistem:conf_target, sistem:ocr_text, outcome, status_penilaian (lengkap/belum/stale), latensi_ms` — satu baris per foto (termasuk yang belum dinilai, kolom kosong; Review Focus: semua baris mentah).
  - `to_csv(batch) -> str` (delimiter `;`, BOM utf-8 untuk Excel ID), `to_xlsx(batch) -> bytes` — openpyxl: sheet 1 "Ringkasan" (key/value hasil `batch_report`, None → "N/A"), sheet 2 "Per Foto" (rows).
  - Download: `GET /batch/<id>/report/export.csv` dan `.../export.xlsx` (content-disposition attachment).
- [ ] **Step 1: Failing test** — fixture sama Task 7: CSV punya header + jumlah baris = jumlah foto; gt_text/ocr_text cocok; nilai None di Ringkasan → `'N/A'`; xlsx dibuka `openpyxl.load_workbook(BytesIO(...))` → 2 sheet, nilai sel Ringkasan accuracy cocok.
- [ ] **Step 2: FAIL → Step 3: Implement → Step 4: PASS.**
- [ ] **Step 5: Commit** `feat: csv and xlsx export`.

### Task 9: Uji konsistensi (augmentasi + job + halaman)

**Files:**
- Create: `webapp/testing/services/augmentation.py`, `webapp/testing/services/consistency.py`, view `photo_consistency` (+ start/cancel/status), url `/photo/<id>/consistency/` (+ POST start, POST cancel, GET status JSON), templates `consistency.html` + hasil, test `test_augmentation.py`, `test_consistency.py`

**Interfaces:**
- Consumes: `inference.get_service()`, `Photo.system_result` baseline box target (`target_detection_index`), `Annotation` (syarat `is_complete()`), `iou()` (salin helper kecil dari `scripts/evaluate_endtoend.py` ke `consistency.py` — logika 8 baris, fine), `metrics` untuk teks GT.
- Produces:
  - `augmentation.perturb(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray` — brightness `1±0.2` (uniform), contrast `1±0.2`, geser `0.4%` tiap sumbu (warp/translate), noise Gaussian `rng.normal(0, 3)`; clamp 0..255 uint8.
  - `consistency.baseline_of(photo) -> dict` — `{'box': [...], 'confidence': float, 'text': str}` dari `system_result` + annotation target; `text_gt` = `gt_text` (bisa '').
  - `consistency.execute(run: ConsistencyRun)` — loop `i in range(n)`: reload foto, `perturb(rng=default_rng(run.seed + i))`, `detect`, box dengan IoU terbesar vs baseline (tak ada → `detected=False`), bila detected: OCR crop box itu → `normalize_plate_text`; append `{'i', 'detected', 'confidence', 'iou', 'ocr_text', 'same_as_baseline', 'same_as_gt'}` ke `run.runs` (simpan bertahap tiap 10 iterasi — progres), cek `cancel_requested` tiap iterasi → status `'cancelled'` simpan parsial; selesai → `summary`: `{'detected_x_of_n', 'conf_mean', 'conf_std', 'same_as_baseline_x_of_n', 'same_as_gt_x_of_n', 'text_distribution': {text: count}, 'n', 'seed'}`. `iou_baseline` atas box detected yang tidak match (IoU<0.5) tetap dicatat.
  - `consistency.start_run(run)` → daemon thread (module-level, sama pola Task 3).
  - Halaman: form start (N default 100, seed default 42, read-only info baseline), hanya muncul tombol start bila `photo.annotation.is_complete()` dan tidak stale dan `status='done'` (case foto tanpa GT → halaman tampil pesan penolakan jelas). Progress via polling `GET .../consistency/status/` → `{'status','done_iterations','of','summary?'}`; hasil: kalimat ringkasan (§7.4 spec: "Terdeteksi X dari N run", "teks sama dengan baseline Y dari N", "teks sama dengan ground truth Z dari N"), Chart.js histogram confidence, tabel sebaran teks, accordion daftar run; tombol Batalkan → POST cancel.
  - Dashboard batch: link "Uji konsistensi" per foto dinilai (sumber pemilihan foto §7.2.5).
- [ ] **Step 1: Failing tests:**
  - `perturb`: deterministik untuk seed sama, beda untuk seed beda, dtype/shape/rentang valid (uint8, 0..255, tak mengubah ukuran).
  - `execute` dengan stub service (detect → 1 box tetap conf 0.9, OCR → teks bervariasi deterministik per i) n=20, seed fix: `summary['detected_x_of_n']==20`, same_as_* sesuai tabel stub, `text_distribution` menjumlah ke n, simpan progres (setelah i=10, `len(run.runs)==10`), cancel_requested di-set oleh stub iterasi ke 5 → status `'cancelled'`, runs 5.
  - View guard: foto tanpa annotation → POST start → 403/form error, run tidak dibuat; annotation `stale=True` → ditolak.
- [ ] **Step 2: FAIL → Step 3: Implement → Step 4: PASS.**
- [ ] **Step 5: Sanity nyata sekali:** 1 foto dari Task 3 sanity, `n=5` dengan service asli → done; catat durasi per-iterasi di laporan eksekusi (untuk estimasi user), jangan commit artefak. **Step 6: Commit** `feat: consistency test runs`.

### Task 10: Finishing & verifikasi end-to-end

**Files:**
- Create: `webapp/README.md`
- Modify: tidak ada (kecuali fix temuan)

**Interfaces:** —

- [ ] **Step 1:** `manage.py test testing` — seluruh suite PASS; laporkan jumlah test.
- [ ] **Step 2:** `manage.py runserver 0.0.0.0:8000` (background) → alur manual end-to-end via browser/MCP browser bila tersedia: buat batch → upload 2-3 foto real (dari `test/images` — foto test dataset bukan "real", tapi memvalidasi pipeline; minimal 1 yang memuat plat) → tunggu processing → nilai → laporan muncul angka + grafik → export CSV/Excel terbuka valid → konsistensi `n=5` jalan → batalkan. Perbaiki temuan; fix di-commmit bila ada.
- [ ] **Step 3: README.md** — cara menjalankan (activate venv, migrate, runserver, alamat), cara ganti weights/threshold di settings, definisi tiap metrik (§7.3 desain diringkas), batasan (ground truth = penilaian tunggal user; pseudo-label tidak dipakai; tanpa auth — jangan expose ke jaringan publik).
- [ ] **Step 4: Commit** `docs: webapp README` → minta whole-branch review (superpowers:requesting-code-review).

## Open defaults (keputusan builder, tandai saat implementasi)

- Ambang IoU match baseline konsistensi: 0.5.
- Polling interval UI: 2s (processing), 3s (konsistensi).
- `Photo.image` menerima hingga 20 MB; validasi di form upload.
- Layout tabel/angka desimal dua tempat, persen 2 desimal di laporan & export.
