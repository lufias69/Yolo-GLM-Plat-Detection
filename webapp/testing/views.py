import json

from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Annotation, ConsistencyRun, Photo, TestBatch, YoloModel
from .forms import AnnotationForm
from .services import consistency, export, metrics, ocr_analysis, workers
from .services.metrics import similarity_ratio
from .services.inference import get_service  # used by quick_test

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}


def home(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        if not name:
            messages.error(request, 'Nama batch wajib diisi.')
            return redirect('testing:home')
        batch = TestBatch.objects.create(
            name=name, description=request.POST.get('description', '').strip())
        return redirect('testing:batch_detail', batch_id=batch.id)
    batches = TestBatch.objects.all()
    stats = {
        'batches': batches.count(),
        'photos': Photo.objects.count(),
        'done': Photo.objects.filter(status='done').count(),
        'rated': Annotation.objects.count(),
    }
    return render(request, 'testing/home.html', {'batches': batches, 'stats': stats})


def batch_detail(request, batch_id):
    batch = get_object_or_404(TestBatch, pk=batch_id)
    photos = batch.photos.select_related('annotation').prefetch_related('annotation__engine_judgments').all()
    pending_ids = [p.id for p in photos if p.status in ('queued', 'processing')]
    return render(request, 'testing/batch_detail.html',
                  {'batch': batch, 'photos': photos, 'pending_ids': pending_ids})


def _next_unannotated(batch, after_id=None):
    photos = batch.photos.select_related('annotation').order_by('uploaded_at', 'id').filter(status='done')
    for p in photos:
        if after_id is not None and p.id <= after_id:
            continue
        ann = getattr(p, 'annotation', None)
        if ann is None or not ann.is_complete() or ann.stale:
            return p
    return None


def next_unannotated(request, batch_id):
    batch = get_object_or_404(TestBatch, pk=batch_id)
    nxt = _next_unannotated(batch)
    if nxt is None:
        return redirect('testing:batch_detail', batch_id=batch.id)
    return redirect('testing:photo_detail', photo_id=nxt.id)


def _annotated_stats(batch):
    photos = batch.photos.select_related('annotation').filter(status='done')
    total = 0
    done = 0
    for p in photos:
        total += 1
        ann = getattr(p, 'annotation', None)
        if ann is not None and ann.is_complete() and not ann.stale:
            done += 1
    return done, total


def _engine_rows(form):
    results = form.engine_results()
    rows = []
    for eng in form.engines:
        field = form[f'eng_ok_{eng}']
        rows.append({
            'engine': eng,
            'text': results.get(eng, ''),
            'ok_field': field,
            'wrong': form[f'eng_wrong_{eng}'].value(),
            'note': form[f'eng_note_{eng}'].value(),
            'errors': '; '.join(field.errors),
        })
    return rows


def photo_detail(request, photo_id):
    photo = get_object_or_404(Photo.objects.select_related('annotation', 'batch'), pk=photo_id)
    if photo.status != 'done':
        return redirect('testing:batch_detail', batch_id=photo.batch_id)
    ann = getattr(photo, 'annotation', None)
    form = AnnotationForm(instance=ann, photo=photo)
    done, total = _annotated_stats(photo.batch)
    return render(request, 'testing/photo_detail.html', {
        'photo': photo, 'form': form, 'batch': photo.batch,
        'progress_done': done, 'progress_total': total,
        'detections': photo.system_result.get('detections', []),
        'operational': photo.operational_boxes,
        'engine_rows': _engine_rows(form),
    })


@require_POST
def annotate(request, photo_id):
    photo = get_object_or_404(Photo.objects.select_related('batch'), pk=photo_id)
    form = AnnotationForm(request.POST, photo=photo)
    if form.is_valid():
        cd = {k: v for k, v in form.cleaned_data.items()
              if k in ('gt_presence', 'detected_ok', 'target_detection_index',
                       'gt_text', 'note', 'multi_plate_confirmed')}
        cd['ocr_ok'] = form.derived_ocr_ok
        ann, _ = Annotation.objects.update_or_create(
            photo=photo, defaults={**cd, 'stale': False})
        form.save_judgments(ann)
        done, total = _annotated_stats(photo.batch)
        nxt = _next_unannotated(photo.batch)
        if nxt is not None:
            messages.success(request, f'Penilaian tersimpan. Progres {done}/{total} foto — lanjut ke foto berikutnya.')
            return redirect('testing:photo_detail', photo_id=nxt.id)
        messages.success(request, f'Penilaian tersimpan. Selesai semua {total}/{total} foto.')
        return redirect('testing:batch_detail', batch_id=photo.batch_id)
    return render(request, 'testing/photo_detail.html', {
        'photo': photo, 'form': form, 'batch': photo.batch,
        'detections': photo.system_result.get('detections', []),
        'operational': photo.operational_boxes,
        'engine_rows': _engine_rows(form),
    })


@require_POST
def upload_photos(request, batch_id):
    batch = get_object_or_404(TestBatch, pk=batch_id)
    files = request.FILES.getlist('files')
    if not files:
        messages.error(request, 'Tidak ada file yang dipilih.')
        return redirect('testing:batch_detail', batch_id=batch.id)
    accepted = 0
    for f in files:
        ext = f.name.rsplit('.', 1)[-1].lower() if '.' in f.name else ''
        if ext not in ALLOWED_EXTENSIONS:
            messages.error(request, f'File {f.name} bukan gambar yang didukung '
                                    f'(jpg/jpeg/png/webp/bmp).')
            continue
        photo = Photo.objects.create(batch=batch, image=f, original_name=f.name,
                                     model=YoloModel.ensure_default())
        workers.enqueue_photo(photo.id)
        accepted += 1
    if accepted:
        messages.success(request, f'{accepted} foto masuk antrean pemrosesan.')
    return redirect('testing:batch_detail', batch_id=batch.id)


@require_POST
def reprocess_photo(request, photo_id):
    photo = get_object_or_404(Photo, pk=photo_id)
    photo.reprocess()
    workers.enqueue_photo(photo.id)
    messages.info(request, f'{photo.original_name} diproses ulang.')
    return redirect('testing:batch_detail', batch_id=photo.batch_id)


def photo_status(request, photo_id):
    photo = get_object_or_404(Photo, pk=photo_id)
    return JsonResponse({'status': photo.status, 'error_message': photo.error_message})


def batch_ocr_report(request, batch_id):
    """Rekap GT vs hasil OCR semua engine: similarity, benar/salah, jumlah salah,
    plus daftar foto yang belum dinilai / tanpa similarity."""
    batch = get_object_or_404(TestBatch, pk=batch_id)
    engines = []
    for p in batch.photos.filter(status='done'):
        for e in p.system_result.get('ocr_all') or []:
            if e.get('engine') not in engines:
                engines.append(e['engine'])
    rows = []
    for p in batch.photos.select_related('annotation').prefetch_related(
            'annotation__engine_judgments').all():
        if p.status != 'done':
            continue
        ann = getattr(p, 'annotation', None)
        import re
        texts = {e['engine']: e.get('text', '') for e in p.system_result.get('ocr_all') or []}
        judgments = {j.engine: j for j in ann.engine_judgments.all()} if ann else {}
        gt = ann.gt_text if ann else ''
        cells = []
        for e in engines:
            j = judgments.get(e)
            text = texts.get(e, '')
            text_ns = re.sub(r'\s+', '', text)
            gt_ns = re.sub(r'\s+', '', gt)
            sim_ns = j.sim_nospasi if j else None
            sim_sp = j.sim_spasi if j else None
            ratio_ns = j.ratio_nospasi if j else None
            ratio_sp = j.ratio_spasi if j else None
            if ratio_ns is None and sim_ns is not None:
                ratio_ns = similarity_ratio(text_ns, gt_ns) if gt_ns else (1.0 if sim_ns == 100 else sim_ns / 100.0)
            if ratio_sp is None and sim_sp is not None:
                ratio_sp = similarity_ratio(text, gt) if gt else (1.0 if sim_sp == 100 else sim_sp / 100.0)
            cells.append({'engine': e, 'text': text,
                          'ok': j.ok if j else '', 'wrong': j.wrong_chars if j else None,
                          'sim_nospasi': sim_ns,
                          'sim_spasi': sim_sp,
                          'ratio_nospasi': ratio_ns,
                          'ratio_spasi': ratio_sp})
        rows.append({
            'photo': p, 'ann': ann, 'cells': cells,
            'unrated': ann is None or not ann.is_complete(),
        })
    unrated = [r for r in rows if r['unrated']]
    return render(request, 'testing/ocr_report.html', {
        'batch': batch, 'engines': engines, 'rows': rows, 'unrated': unrated,
    })


def batch_ocr_analysis_view(request, batch_id):
    """Halaman analisis similarity OCR: seberapa salah hasil yang salah."""
    batch = get_object_or_404(TestBatch, pk=batch_id)
    analysis = ocr_analysis.analyze_ocr_similarity(batch)
    return render(request, 'testing/ocr_analysis.html', {
        'batch': batch,
        'analysis': analysis,
        'chart_data_json': json.dumps(ocr_analysis.chart_data(analysis)),
        'charts_json': json.dumps(ocr_analysis.chart_data_all(analysis)),
    })


def batch_ocr_export_csv(request, batch_id):
    from django.http import HttpResponse
    batch = get_object_or_404(TestBatch, pk=batch_id)
    resp = HttpResponse(export.ocr_to_csv(batch), content_type='text/csv; charset=utf-8')
    resp['Content-Disposition'] = f'attachment; filename="batch_{batch_id}_ocr.csv"'
    return resp


def batch_ocr_export_xlsx(request, batch_id):
    from django.http import HttpResponse
    batch = get_object_or_404(TestBatch, pk=batch_id)
    resp = HttpResponse(export.ocr_to_xlsx(batch),
                        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    resp['Content-Disposition'] = f'attachment; filename="batch_{batch_id}_ocr.xlsx"'
    return resp


def batch_report_view(request, batch_id):
    batch = get_object_or_404(TestBatch, pk=batch_id)
    rep = metrics.batch_report(batch)
    rows = []
    for p in batch.photos.select_related('annotation').all():
        ann = getattr(p, 'annotation', None)
        det = metrics.target_detection(p) if p.status == 'done' else None
        rows.append({
            'photo': p, 'ann': ann, 'outcome': rep['outcomes'].get(p.id, ''),
            'conf': det['confidence'] if det else None,
            'ocr_text': det['ocr_text'] if det else '',
            'latency_ms': p.system_result.get('latency', {}).get('total_ms'),
        })
    return render(request, 'testing/report.html', {
        'batch': batch, 'rep': rep, 'rows': rows,
        'main_engine': settings.ANPR['OCR_ENGINE'],
        'roc_available': rep['roc']['auc'] is not None,
        'roc_points_json': json.dumps(rep['roc']['points']),
    })


def batch_export_csv(request, batch_id):
    from django.http import HttpResponse
    batch = get_object_or_404(TestBatch, pk=batch_id)
    resp = HttpResponse(export.to_csv(batch), content_type='text/csv; charset=utf-8')
    resp['Content-Disposition'] = f'attachment; filename="batch_{batch_id}.csv"'
    return resp


def batch_export_xlsx(request, batch_id):
    from django.http import HttpResponse
    batch = get_object_or_404(TestBatch, pk=batch_id)
    resp = HttpResponse(export.to_xlsx(batch),
                        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    resp['Content-Disposition'] = f'attachment; filename="batch_{batch_id}.xlsx"'
    return resp


def _consistency_eligible(photo):
    """Konsistensi butuh baseline box yang bermakna: foto selesai, penilaian
    lengkap tidak stale, DAN sistem pernah mendeteksi pada threshold operasional."""
    ann = getattr(photo, 'annotation', None)
    return (photo.status == 'done' and ann is not None and ann.is_complete()
            and not ann.stale and bool(photo.operational_boxes))


def photo_consistency(request, photo_id):
    photo = get_object_or_404(Photo.objects.select_related('annotation', 'batch'), pk=photo_id)
    runs = photo.consistency_runs.all()[:10]
    return render(request, 'testing/consistency.html', {
        'photo': photo, 'batch': photo.batch, 'runs': runs,
        'eligible': _consistency_eligible(photo),
        'baseline': consistency.baseline_of(photo),
    })


@require_POST
def consistency_start(request, photo_id):
    photo = get_object_or_404(Photo.objects.select_related('annotation'), pk=photo_id)
    if not _consistency_eligible(photo):
        messages.error(request, 'Uji konsistensi butuh foto selesai diproses, penilaian lengkap (tidak stale), dan minimal satu deteksi sebagai baseline.')
        return redirect('testing:photo_consistency', photo_id=photo.id)
    try:
        n = max(1, min(500, int(request.POST.get('n', 100))))
    except ValueError:
        n = 100
    try:
        seed = int(request.POST.get('seed', 42))
    except ValueError:
        seed = 42
    run = ConsistencyRun.objects.create(photo=photo, n=n, seed=seed)
    consistency.start_run(run)
    messages.success(request, f'Uji konsistensi n={n} dimulai (seed {seed}).')
    return redirect('testing:photo_consistency', photo_id=photo.id)


@require_POST
def consistency_cancel(request, photo_id, run_id):
    run = get_object_or_404(ConsistencyRun, pk=run_id, photo_id=photo_id)
    if run.status in ('queued', 'running'):
        run.cancel_requested = True
        run.save(update_fields=['cancel_requested'])
    return JsonResponse({'status': run.status})


def consistency_status(request, photo_id, run_id):
    run = get_object_or_404(ConsistencyRun, pk=run_id, photo_id=photo_id)
    return JsonResponse({
        'status': run.status,
        'done_iterations': len(run.runs),
        'of': run.n,
        'summary': run.summary if run.status == 'done' else None,
    })


def _looks_like_pt_weights(f):
    """.pt = zip archive (modern torch.save) or pickle protocol stream (legacy)."""
    head = f.read(4)
    f.seek(0)
    return head[:2] == b'PK' or head[:1] == b'\x80'


def models_page(request):
    models = YoloModel.objects.all()
    YoloModel.ensure_default()
    return render(request, 'testing/models.html', {'models': YoloModel.objects.all()})


@require_POST
def model_upload(request):
    name = request.POST.get('name', '').strip()
    f = request.FILES.get('weights')
    if not name:
        messages.error(request, 'Nama model wajib diisi.')
    elif not f or not f.name.lower().endswith('.pt'):
        messages.error(request, 'File harus berformat .pt (bobot Ultralytics YOLO).')
    elif not _looks_like_pt_weights(f):
        messages.error(request, 'Isi file bukan bobot PyTorch yang valid (.pt = arsip zip '
                                'atau stream pickle).')
    elif YoloModel.objects.filter(name=name).exists():
        messages.error(request, f'Nama "{name}" sudah dipakai.')
    else:
        YoloModel.objects.create(name=name, weights=f)
        messages.success(request, f'Model "{name}" terdaftar. Klik "Jadikan aktif" untuk memakainya.')
    return redirect('testing:models')


@require_POST
def model_activate(request, model_id):
    m = get_object_or_404(YoloModel, pk=model_id)
    YoloModel.objects.filter(is_active=True).exclude(pk=m.pk).update(is_active=False)
    m.is_active = True
    m.save(update_fields=['is_active'])
    messages.success(request, f'Model aktif: {m.name}. Foto baru akan diproses dengan model ini.')
    return redirect('testing:models')


@require_POST
def model_rename(request, model_id):
    m = get_object_or_404(YoloModel, pk=model_id)
    name = request.POST.get('name', '').strip()
    if not name:
        messages.error(request, 'Nama model tidak boleh kosong.')
    elif YoloModel.objects.filter(name=name).exclude(pk=m.pk).exists():
        messages.error(request, f'Nama "{name}" sudah dipakai model lain.')
    else:
        lama = m.name
        m.name = name
        m.save(update_fields=['name'])
        messages.success(request, f'Nama model diubah: "{lama}" → "{name}".')
    return redirect('testing:models')


@require_POST
def model_delete(request, model_id):
    m = get_object_or_404(YoloModel, pk=model_id)
    if m.is_active:
        messages.error(request, 'Model aktif tidak bisa dihapus — aktifkan model lain dulu.')
    elif m.photos.exists():
        messages.error(request, f'Model "{m.name}" sudah dipakai {m.photos.count()} foto — tidak bisa dihapus (riwayat harus tertelusuri).')
    else:
        m.delete()
        messages.success(request, f'Model "{m.name}" dihapus.')
    return redirect('testing:models')


QUICKTEST_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}


def quick_test(request):
    """Uji cepat satu gambar terhadap model aktif. Tidak disimpan ke DB/batch,
    tidak masuk penilaian & laporan. Gambar hasil inference di-embed (base64),
    file sementara dibersihkan."""
    import base64
    import os
    import tempfile
    from pathlib import Path

    from .services.inference import InferenceError

    result = None
    error = None
    if request.method == 'POST':
        f = request.FILES.get('image')
        ext = f.name.rsplit('.', 1)[-1].lower() if f and '.' in f.name else ''
        if not f or ext not in QUICKTEST_EXTENSIONS:
            error = 'Unggah satu gambar (jpg/jpeg/png/webp/bmp).'
        else:
            fd, tmp_path = tempfile.mkstemp(suffix=f'.{ext}')
            os.close(fd)
            try:
                with open(tmp_path, 'wb') as out:
                    for chunk in f.chunks():
                        out.write(chunk)
                model = YoloModel.objects.filter(is_active=True).first() or YoloModel.ensure_default()
                weights = model.effective_path()
                try:
                    result = get_service().process_file(Path(tmp_path), weights_path=weights)
                    result['model_name'] = model.name
                    ann_path = os.path.join(str(settings.MEDIA_ROOT), result['annotated_image'])
                    if os.path.exists(ann_path):
                        with open(ann_path, 'rb') as af:
                            result['annotated_b64'] = base64.b64encode(af.read()).decode()
                        os.remove(ann_path)
                    # hasil 4 engine sudah dihitung & (untuk batch) tersimpan di
                    # system_result['ocr_all'] — quicktest memakainya langsung.
                except InferenceError as exc:
                    error = str(exc)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    return render(request, 'testing/quicktest.html', {'result': result, 'error': error})
