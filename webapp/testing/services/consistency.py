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
            # orientasi sama dengan baseline (foto mungkin butuh rotasi)
            candidates = service.detect_oriented(variant, model=yolo,
                                                 preferred_rotation=photo.system_result.get('rotation', 0))
            from anpr.postprocessing import normalize_plate_text  # noqa: F401
            from django.conf import settings
            from .inference import map_box_to_original
            orig_w, orig_h = image.shape[1], image.shape[0]
            best = None  # (ocr_conf, frame_d, ob, text)
            for frame, dets, rotation in candidates:
                operational = [d for d in dets
                               if d['confidence'] >= settings.ANPR['CONF_OPERATIONAL']]
                if not operational:
                    continue
                target = operational[0]
                ob = map_box_to_original(target['box'], rotation, orig_w, orig_h)
                if baseline['box'] is not None:
                    val = iou(ob, baseline['box'])
                    if val < IOU_MATCH_THRESHOLD:
                        continue
                else:
                    val = 1.0
                text, ocr_conf = service.recognize_crop(frame, target['box'])
                if best is None or ocr_conf > best[0]:
                    best = (ocr_conf, target, ob, text, val, rotation)
            item = {'i': i, 'detected': False, 'confidence': None, 'iou': 0.0,
                    'ocr_text': '', 'same_as_baseline': False, 'same_as_gt': False,
                    'rotation': photo.system_result.get('rotation', 0)}
            if best is not None:
                ocr_conf, target, ob, text, val, _ = best
                item.update({
                    'detected': True,
                    'confidence': target['confidence'],
                    'iou': round(val, 4),
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
