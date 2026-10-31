"""
testing/services/consistency.py
100x consistency runs with light random perturbation per run (spec 7.4).
"""
import threading

from django.utils import timezone

from .augmentation import perturb
from .inference import get_service

IOU_MATCH_THRESHOLD = 0.5


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def baseline_of(photo) -> dict:
    """{'box','confidence','text'} from system_result + annotation target."""
    from .metrics import target_detection
    det = target_detection(photo)
    if det is None:
        return {'box': None, 'confidence': None, 'text': ''}
    return {'box': det['box'], 'confidence': det['confidence'], 'text': det['ocr_text']}


def execute(run):
    """Run N perturbed inferences; store per-iteration results and summary."""
    import cv2
    import numpy as np

    photo = run.photo
    run.status = 'running'
    run.save(update_fields=['status'])
    try:
        image = cv2.imread(photo.image.path)
        if image is None:
            raise RuntimeError('Foto tidak dapat dibaca ulang')
    except Exception as exc:
        run.status = 'error'
        run.error_message = str(exc)
        run.finished_at = timezone.now()
        run.save()
        return

    baseline = baseline_of(photo)
    gt_text = getattr(photo.annotation, 'gt_text', '') or baseline['text']
    service = get_service()
    weights = photo.model.effective_path() if photo.model_id else None
    yolo = service.model_for(weights)
    results = []
    try:
        for i in range(run.n):
            run.refresh_from_db()
            if run.cancel_requested:
                run.runs = results
                run.status = 'cancelled'
                run.finished_at = timezone.now()
                run.save()
                return
            rng = np.random.default_rng(run.seed + i)
            variant = perturb(image, rng)
            dets = service.detect(variant, model=yolo)
            from django.conf import settings
            dets = [d for d in dets
                    if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]
            item = {'i': i, 'detected': False, 'confidence': None, 'iou': 0.0,
                    'ocr_text': '', 'same_as_baseline': False, 'same_as_gt': False}
            if dets:
                if baseline['box'] is not None:
                    best = max(dets, key=lambda d: iou(d['box'], baseline['box']))
                    best_iou = iou(best['box'], baseline['box'])
                else:
                    best = dets[0]
                    best_iou = 1.0
                if best_iou >= IOU_MATCH_THRESHOLD:
                    from anpr.postprocessing import normalize_plate_text
                    text, _ = service.recognize_crop(variant, best['box'])
                    item.update({
                        'detected': True,
                        'confidence': best['confidence'],
                        'iou': round(best_iou, 4),
                        'ocr_text': text,
                        'same_as_baseline': text == baseline['text'],
                        'same_as_gt': bool(gt_text) and text == gt_text,
                    })
            results.append(item)
            if len(results) % 10 == 0:
                run.runs = results
                run.save(update_fields=['runs'])
        run.runs = results
        run.summary = summarize(results, run.n)
        run.status = 'done'
    except Exception as exc:
        run.runs = results
        run.status = 'error'
        run.error_message = str(exc)
    run.finished_at = timezone.now()
    run.save()
    return


def summarize(results, n) -> dict:
    import statistics

    confs = [r['confidence'] for r in results if r['detected'] and r['confidence'] is not None]
    dist = {}
    for r in results:
        key = r['ocr_text'] if r['detected'] else '(tidak terdeteksi)'
        dist[key] = dist.get(key, 0) + 1
    return {
        'n': n,
        'detected_x_of_n': f"{sum(1 for r in results if r['detected'])}/{n}",
        'conf_mean': statistics.fmean(confs) if confs else None,
        'conf_std': statistics.pstdev(confs) if len(confs) > 1 else 0.0,
        'same_as_baseline_x_of_n': f"{sum(1 for r in results if r['same_as_baseline'])}/{n}",
        'same_as_gt_x_of_n': f"{sum(1 for r in results if r['same_as_gt'])}/{n}",
        'text_distribution': dist,
    }


def start_run(run):
    t = threading.Thread(target=_run_in_thread, args=(run.pk,), daemon=True,
                         name=f'anpr-consistency-{run.pk}')
    t.start()


def _run_in_thread(run_pk):
    from django.db import close_old_connections
    close_old_connections()
    from testing.models import ConsistencyRun
    run = ConsistencyRun.objects.get(pk=run_pk)
    execute(run)
