"""
testing/services/export.py
CSV and XLSX exports of the per-photo recap + batch summary.
"""
import csv
import io

from . import metrics

OCR_ENGINES = ('easyocr', 'paddleocr', 'tesseract', 'glm-ocr')

HEADERS = ['file', 'model', 'gt_presence', 'gt_ada_plat', 'detected_ok', 'target_index', 'ocr_ok', 'gt_text',
           'catatan', 'sistem:deteksi_ada', 'sistem:conf_target', 'sistem:ocr_text',
           'outcome', 'status_penilaian', 'latensi_ms',
           'ocr_easyocr', 'skor_easyocr', 'ocr_paddleocr', 'skor_paddleocr',
           'ocr_tesseract', 'skor_tesseract', 'ocr_glm-ocr', 'skor_glm-ocr', 'fragmen_all']
JUDGE_SUFFIXES = ['nilai', 'slh_karakter', 'catatan', 'sim_tanpa_spasi', 'sim_dengan_spasi', 'ratio_tanpa_spasi', 'ratio_dengan_spasi']
for _e in OCR_ENGINES:
    for _s in JUDGE_SUFFIXES:
        HEADERS.append(f'{_s}_{_e}')

OCR_REPORT_HEADERS = ['file', 'gt_text', 'plat_di_foto', 'deteksi_sistem',
                      'bacaan_easyocr', 'nilai_easyocr', 'slh_karakter_easyocr', 'catatan_easyocr',
                      'sim_tanpa_spasi_easyocr', 'sim_dengan_spasi_easyocr',
                      'ratio_tanpa_spasi_easyocr', 'ratio_dengan_spasi_easyocr',
                      'bacaan_paddleocr', 'nilai_paddleocr', 'slh_karakter_paddleocr', 'catatan_paddleocr',
                      'sim_tanpa_spasi_paddleocr', 'sim_dengan_spasi_paddleocr',
                      'ratio_tanpa_spasi_paddleocr', 'ratio_dengan_spasi_paddleocr',
                      'bacaan_tesseract', 'nilai_tesseract', 'slh_karakter_tesseract', 'catatan_tesseract',
                      'sim_tanpa_spasi_tesseract', 'sim_dengan_spasi_tesseract',
                      'ratio_tanpa_spasi_tesseract', 'ratio_dengan_spasi_tesseract',
                      'bacaan_glm-ocr', 'nilai_glm-ocr', 'slh_karakter_glm-ocr', 'catatan_glm-ocr',
                      'sim_tanpa_spasi_glm-ocr', 'sim_dengan_spasi_glm-ocr',
                      'ratio_tanpa_spasi_glm-ocr', 'ratio_dengan_spasi_glm-ocr']


def judge_cols(photo, engines):
    ann = getattr(photo, 'annotation', None)
    js = {j.engine: j for j in ann.engine_judgments.all()} if ann else {}
    out = []
    for e in engines:
        j = js.get(e)
        out += [j.ok if j else '', j.wrong_chars if j and j.wrong_chars is not None else '',
                j.note if j else '',
                j.sim_nospasi if j and j.sim_nospasi is not None else '',
                j.sim_spasi if j and j.sim_spasi is not None else '',
                j.ratio_nospasi if j and j.ratio_nospasi is not None else '',
                j.ratio_spasi if j and j.ratio_spasi is not None else '']
    return out


def _ocr_report_engine_cols(photo, engines):
    """Columns for the OCR report export: per-engine text + all judgment fields."""
    ann = getattr(photo, 'annotation', None)
    js = {j.engine: j for j in ann.engine_judgments.all()} if ann else {}
    ocr_all = {e.get('engine'): e for e in photo.system_result.get('ocr_all', [])}
    out = []
    for e in engines:
        eng_data = ocr_all.get(e, {})
        j = js.get(e)
        out += [eng_data.get('text', ''),
                j.ok if j else '',
                j.wrong_chars if j and j.wrong_chars is not None else '',
                j.note if j else '',
                j.sim_nospasi if j and j.sim_nospasi is not None else '',
                j.sim_spasi if j and j.sim_spasi is not None else '',
                j.ratio_nospasi if j and j.ratio_nospasi is not None else '',
                j.ratio_spasi if j and j.ratio_spasi is not None else '']
    return out


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
        ocr_all = {e.get('engine'): e for e in p.system_result.get('ocr_all', [])}
        engine_cols = []
        for name in OCR_ENGINES:
            e = ocr_all.get(name, {})
            engine_cols.append(e.get('text', ''))
            engine_cols.append(e.get('conf', ''))
        frag_str = ' | '.join(
            f"{e.get('engine')}: " + ', '.join(f"{t} ({c})" for t, c in e.get('fragments', []))
            for e in p.system_result.get('ocr_all', []))
        out.append([
            p.original_name,
            p.model.name if p.model_id else '',
            ann.gt_presence if ann else '',
            'ya' if ann and ann.gt_presence == 'ADA' else ('tidak' if ann and ann.gt_presence == 'TIDAK_ADA' else ''),
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
            *engine_cols,
            frag_str,
            *judge_cols(p, OCR_ENGINES),
        ])
    return out


def ocr_rows(batch) -> list:
    """Rows matching the OCR report table: per-photo + per-engine OCR/judgment data."""
    engines = []
    for p in batch.photos.filter(status='done'):
        for e in p.system_result.get('ocr_all') or []:
            if e.get('engine') not in engines:
                engines.append(e['engine'])
    out = [list(OCR_REPORT_HEADERS)]
    for p in batch.photos.select_related('annotation').all():
        if p.status != 'done':
            continue
        ann = getattr(p, 'annotation', None)
        out.append([
            p.original_name,
            ann.gt_text if ann else '',
            ann.gt_presence if ann else '',
            ann.detected_ok if ann else '',
            *_ocr_report_engine_cols(p, engines),
        ])
    return out


def ocr_to_csv(batch) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=';')
    for r in ocr_rows(batch):
        writer.writerow(r)
    return '﻿' + buf.getvalue()


def ocr_to_xlsx(batch) -> bytes:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = 'OCR Report'
    for r in ocr_rows(batch):
        ws.append(r)
    data = io.BytesIO()
    wb.save(data)
    return data.getvalue()


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
