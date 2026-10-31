"""
testing/services/metrics.py
Pure metric functions over Photo(+Annotation) rows. Analysis unit = photo
(1 photo = 1 motor). Un-rated or stale annotations never contribute; missing
data yields None (never a misleading 0.0).
"""
import statistics

import numpy as np

_CER = None


def _cer_fn():
    """Reuse the research CER implementation (scripts/evaluate_ocr.py) verbatim."""
    global _CER
    if _CER is None:
        import importlib.util
        from django.conf import settings
        path = settings.RESEARCH_ROOT / 'scripts' / 'evaluate_ocr.py'
        spec = importlib.util.spec_from_file_location('research_evaluate_ocr', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CER = mod.cer
    return _CER


def levenshtein_distance(pred: str, truth: str) -> int:
    if len(truth) == 0:
        return len(pred)
    m, n = len(truth), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (truth[i - 1] != pred[j - 1]))
            prev = cur
    return dp[n]


def _ann(p):
    ann = getattr(p, 'annotation', None)
    if ann is None or ann.stale or not ann.is_complete():
        return None
    return ann


def _target_detection(p):
    """Detection dict for the target box, or None."""
    ops = p.operational_boxes
    if not ops:
        return None
    ann = _ann(p)
    idx = ann.target_detection_index if ann else None
    for d in p.system_result.get('detections', []):
        if d['index'] == idx:
            return d
    best = max((d for d in p.system_result.get('detections', [])
                if d['index'] in ops), key=lambda d: d['confidence'], default=None)
    return best


def photo_outcome(p) -> str:
    """'TP'/'FN'/'FP'/'TN' for annotated photos; '' otherwise."""
    ann = _ann(p)
    if ann is None:
        return ''
    detected = bool(p.operational_boxes)
    if ann.gt_presence == 'ADA':
        if not detected or ann.detected_ok == 'GAGAL':
            return 'FN'
        return 'TP' if ann.detected_ok == 'BENAR' else ''
    if ann.gt_presence == 'TIDAK_ADA':
        return 'FP' if detected else 'TN'
    return ''


def detection_metrics(photos) -> dict:
    tp = fp = fn = tn = 0
    for p in photos:
        outcome = photo_outcome(p)
        if outcome == 'TP':
            tp += 1
        elif outcome == 'FP':
            fp += 1
        elif outcome == 'FN':
            fn += 1
        elif outcome == 'TN':
            tn += 1
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else (0.0 if (fp or tp) else None)
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    elif precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = None
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def _roc_points(photos):
    """(score, label) pairs: positive = TP photos, negative = FP/TN/FN photos."""
    pts = []
    for p in photos:
        outcome = photo_outcome(p)
        if outcome == 'TP':
            det = _target_detection(p)
            pts.append((det['confidence'] if det else 0.0, True))
        elif outcome in ('FP', 'TN'):
            dets = [d for d in p.system_result.get('detections', [])
                    if d['index'] in p.operational_boxes]
            conf = max((d['confidence'] for d in dets), default=0.0)
            pts.append((conf, False))
        elif outcome == 'FN':
            pts.append((0.0, False))
    return pts


def roc_auc(points) -> dict:
    """AUC-ROC via threshold sweep + trapezoid. None if a class is missing."""
    pos = [s for s, y in points if y]
    neg = [s for s, y in points if not y]
    if not pos or not neg:
        return {'auc': None, 'points': []}
    thresholds = sorted({s for s, _ in points}, reverse=True)
    tpr_fpr = [(0.0, 0.0, 1.0)]
    for t in thresholds:
        tpr = sum(1 for s in pos if s >= t) / len(pos)
        fpr = sum(1 for s in neg if s >= t) / len(neg)
        tpr_fpr.append((fpr, tpr, t))
    tpr_fpr.append((1.0, 1.0, min(thresholds)))
    auc = 0.0
    for i in range(1, len(tpr_fpr)):
        x0, y0 = tpr_fpr[i - 1][0], tpr_fpr[i - 1][1]
        x1, y1 = tpr_fpr[i][0], tpr_fpr[i][1]
        auc += (x1 - x0) * (y0 + y1) / 2
    return {'auc': float(auc),
            'points': [{'fpr': fpr, 'tpr': tpr, 'threshold': t}
                       for fpr, tpr, t in tpr_fpr]}


def roc_metrics(photos) -> dict:
    return roc_auc(_roc_points(photos))


def ocr_metrics(photos) -> dict:
    cer_fn = _cer_fn()
    cers = []
    exact = 0
    char_acc = []
    n = 0
    for p in photos:
        ann = _ann(p)
        if ann is None or ann.ocr_ok not in ('BENAR', 'SALAH'):
            continue
        det = _target_detection(p)
        if det is None:
            continue
        gt = ann.gt_text or (det.get('ocr_text', '') if ann.ocr_ok == 'BENAR' else '')
        if not gt:
            continue
        pred = det.get('ocr_text', '')
        n += 1
        if ann.ocr_ok == 'BENAR':
            exact += 1
            cers.append(0.0)
            char_acc.append(1.0)
        else:
            dist = levenshtein_distance(pred, gt)
            cers.append(cer_fn(pred, gt))
            char_acc.append(max(0.0, 1 - dist / len(gt)))
    if n == 0:
        return {'exact_match': None, 'cer_mean': None, 'char_accuracy': None, 'n_evaluated': 0}
    return {'exact_match': exact / n, 'cer_mean': statistics.fmean(cers),
            'char_accuracy': statistics.fmean(char_acc), 'n_evaluated': n}


def latency_metrics(photos) -> dict:
    vals = [p.system_result.get('latency', {}).get('total_ms') for p in photos
            if p.status == 'done' and p.system_result.get('latency', {}).get('total_ms') is not None]
    if not vals:
        return {'mean_ms': None, 'median_ms': None, 'p95_ms': None, 'fps': None, 'n': 0}
    mean = statistics.fmean(vals)
    return {'mean_ms': mean, 'median_ms': statistics.median(vals),
            'p95_ms': float(np.percentile(vals, 95)),
            'fps': 1000.0 / mean if mean else None, 'n': len(vals)}


def batch_report(batch) -> dict:
    photos = list(batch.photos.select_related('annotation').all())
    outcomes = {p.id: photo_outcome(p) for p in photos}
    complete = sum(1 for p in photos if _ann(p) is not None)
    stale = sum(1 for p in photos
                if getattr(p, 'annotation', None) and p.annotation.stale)
    annotated = sum(1 for p in photos if getattr(p, 'annotation', None) is not None)
    return {
        'detection': detection_metrics(photos),
        'roc': roc_metrics(photos),
        'ocr': ocr_metrics(photos),
        'latency': latency_metrics(photos),
        'outcomes': outcomes,
        'counts': {'total': len(photos), 'annotated': annotated, 'complete': complete,
                   'stale': stale, 'unannotated': len(photos) - annotated},
    }
