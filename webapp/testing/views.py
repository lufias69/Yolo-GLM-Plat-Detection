import json

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Annotation, Photo, TestBatch
from .forms import AnnotationForm
from .services import metrics, workers

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
    return render(request, 'testing/home.html', {'batches': batches})


def batch_detail(request, batch_id):
    batch = get_object_or_404(TestBatch, pk=batch_id)
    photos = batch.photos.select_related('annotation').all()
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


def photo_detail(request, photo_id):
    photo = get_object_or_404(Photo.objects.select_related('annotation', 'batch'), pk=photo_id)
    if photo.status != 'done':
        return redirect('testing:batch_detail', batch_id=photo.batch_id)
    ann = getattr(photo, 'annotation', None)
    form = AnnotationForm(instance=ann, photo=photo)
    return render(request, 'testing/photo_detail.html', {
        'photo': photo, 'form': form, 'batch': photo.batch,
        'detections': photo.system_result.get('detections', []),
        'operational': photo.operational_boxes,
    })


@require_POST
def annotate(request, photo_id):
    photo = get_object_or_404(Photo.objects.select_related('batch'), pk=photo_id)
    form = AnnotationForm(request.POST, photo=photo)
    if form.is_valid():
        ann, _ = Annotation.objects.update_or_create(
            photo=photo, defaults={**form.cleaned_data, 'stale': False})
        if request.POST.get('next'):
            nxt = _next_unannotated(photo.batch, after_id=photo.id)
            if nxt is not None:
                return redirect('testing:photo_detail', photo_id=nxt.id)
            return redirect('testing:batch_detail', batch_id=photo.batch_id)
        messages.success(request, 'Penilaian tersimpan.')
        return redirect('testing:photo_detail', photo_id=photo.id)
    return render(request, 'testing/photo_detail.html', {
        'photo': photo, 'form': form, 'batch': photo.batch,
        'detections': photo.system_result.get('detections', []),
        'operational': photo.operational_boxes,
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
        photo = Photo.objects.create(batch=batch, image=f, original_name=f.name)
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
        'roc_available': rep['roc']['auc'] is not None,
        'roc_points_json': json.dumps(rep['roc']['points']),
    })
