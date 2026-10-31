"""
testing/services/workers.py
Background processing queue (single daemon thread).
"""
import queue
import threading

_QUEUE = None
_THREAD = None
_LOCK = threading.Lock()

# Test-only injection point: set to callable(path)->dict to avoid loading models.
_process_fn = None


def _worker_loop():
    while True:
        photo_id = _QUEUE.get()
        try:
            _process_photo(photo_id)
        except Exception:  # never kill the thread
            import traceback
            traceback.print_exc()
        finally:
            _QUEUE.task_done()


def _process_photo(photo_id):
    from testing.models import Photo
    from testing.services.inference import InferenceError, get_service

    photo = Photo.objects.filter(pk=photo_id).first()
    if photo is None:
        return
    photo.status = 'processing'
    photo.error_message = ''
    photo.save(update_fields=['status', 'error_message'])

    path = photo.image.path
    try:
        fn = _process_fn if _process_fn is not None else get_service().process_file
        result = fn(path)
    except InferenceError as exc:
        photo.status = 'error'
        photo.error_message = str(exc)
        photo.save(update_fields=['status', 'error_message'])
        return
    except Exception as exc:
        photo.status = 'error'
        photo.error_message = f'Kesalahan tak terduga: {exc}'
        photo.save(update_fields=['status', 'error_message'])
        return

    annotated = result.pop('annotated_image', '')
    photo.width = result.pop('width', photo.width)
    photo.height = result.pop('height', photo.height)
    photo.system_result = result
    photo.status = 'done'
    from django.utils import timezone
    photo.processed_at = timezone.now()
    if annotated:
        # the service already wrote the file under MEDIA_ROOT; just record the path
        photo.annotated_image.name = annotated
    try:
        photo.save()
    except Exception as exc:
        # never strand the photo in 'processing' if the final write loses a lock race
        photo.status = 'error'
        photo.error_message = f'Gagal menyimpan hasil: {exc}'
        try:
            photo.save(update_fields=['status', 'error_message'])
        except Exception:
            pass


def _ensure_worker():
    global _QUEUE, _THREAD
    with _LOCK:
        if _QUEUE is None:
            _QUEUE = queue.Queue()
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_worker_loop, daemon=True, name='anpr-worker')
            _THREAD.start()


def enqueue_photo(photo_id):
    _ensure_worker()
    _QUEUE.put(photo_id)


def start_worker():
    _ensure_worker()
