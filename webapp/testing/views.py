from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .models import Photo, TestBatch
from .services import workers

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


def photo_detail(request, photo_id):
    # Placeholder; lengkap di Task 5 (halaman penilaian).
    photo = get_object_or_404(Photo, pk=photo_id)
    return render(request, 'testing/batch_detail.html',
                  {'batch': photo.batch, 'photos': photo.batch.photos.select_related('annotation'),
                   'pending_ids': []})


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
