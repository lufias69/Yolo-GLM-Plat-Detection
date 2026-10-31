"""
testing/services/export.py
CSV and XLSX exports of the per-photo recap + batch summary.
"""
import csv
import io

from . import metrics

HEADERS = ['file', 'gt_presence', 'detected_ok', 'target_index', 'ocr_ok', 'gt_text',
           'catatan', 'sistem:deteksi_ada', 'sistem:conf_target', 'sistem:ocr_text',
           'outcome', 'status_penilaian', 'latensi_ms']


def _status_label(photo):
    ann = getattr(photo, 'annotation', None)
    if ann is None:
        return 'belum dinilai'
    if ann.stale:
        return 'stale'
    return 'lengkap' if ann.is_complete() else 'belum dinilai'


def rows(batch) -> list:
    photos = batch.photos.select_related('annotation').all()
    outcome = {p.id: metrics.photo_outcome(p) for p in photos}
    out = [list(HEADERS)]
    for p in photos:
        ann = getattr(p, 'annotation', None)
        det = metrics.target_detection(p) if p.status == 'done' else None
        out.append([
            p.original_name,
            ann.gt_presence if ann else '',
            ann.detected_ok if ann else '',
            ann.target_detection_index if ann else '',
            ann.ocr_ok if ann else '',
            ann.gt_text if ann else '',
            ann.note if ann else '',
            'ya' if p.operational_boxes else 'tidak',
            f"{det['confidence']:.4f}" if det else '',
            det['ocr_text'] if det else '',
            outcome[p.id],
            _status_label(p),
            p.system_result.get('latency', {}).get('total_ms', ''),
        ])
    return out


def to_csv(batch) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=';')
    for r in rows(batch):
        writer.writerow(r)
    return '﻿' + buf.getvalue()


def _na(value):
    if value is None:
        return 'N/A'
    return value


def to_xlsx(batch) -> bytes:
    from openpyxl import Workbook

    rep = metrics.batch_report(batch)
    wb = Workbook()
    ws = wb.active
    ws.title = 'Ringkasan'
    ws.append(['key', 'value'])
    d, o, l, r_ = rep['detection'], rep['ocr'], rep['latency'], rep['roc']
    counts = rep['counts']
    summary_rows = [
        ('foto_total', counts['total']),
        ('foto_dinilai_lengkap', counts['complete']),
        ('foto_stale', counts['stale']),
        ('foto_belum_dinilai', counts['unannotated']),
        ('deteksi_tp', d['tp']), ('deteksi_fp', d['fp']),
        ('deteksi_fn', d['fn']), ('deteksi_tn', d['tn']),
        ('detection_accuracy', _na(d['accuracy'])),
        ('detection_precision', _na(d['precision'])),
        ('detection_recall', _na(d['recall'])),
        ('detection_f1', _na(d['f1'])),
        ('roc_auc', _na(r_['auc'])),
        ('ocr_exact_match', _na(o['exact_match'])),
        ('ocr_char_accuracy', _na(o['char_accuracy'])),
        ('ocr_cer_mean', _na(o['cer_mean'])),
        ('ocr_n_evaluated', o['n_evaluated']),
        ('latency_mean_ms', _na(l['mean_ms'])),
        ('latency_median_ms', _na(l['median_ms'])),
        ('latency_p95_ms', _na(l['p95_ms'])),
        ('fps', _na(l['fps'])),
    ]
    for k, v in summary_rows:
        ws.append([k, v])

    ws2 = wb.create_sheet('Per Foto')
    for r in rows(batch):
        ws2.append(r)

    data = io.BytesIO()
    wb.save(data)
    return data.getvalue()
