"""
testing/services/ocr_analysis.py
Per-batch similarity analysis for OCR engine judgments.

Answers: "If OCR is wrong, how wrong is it, and is there still similarity?"
Extended with: CER/char accuracy per engine, Wilson CI, error patterns
(confusion pairs / posisi / panjang plat), end-to-end per engine,
skenario sulit + over-confidence, and pairwise McNemar.
"""
import itertools
import math
import re
import statistics

from .metrics import levenshtein_distance


SIMILARITY_BINS = [
    (0, 25, '0–25%'),
    (25, 50, '25–50%'),
    (50, 75, '50–75%'),
    (75, 100, '75–100%'),
    (100, 101, '100%'),
]

OVERCONFIDENT_THRESHOLD = 0.8


def _bin_label(value):
    if value is None:
        return None
    for lo, hi, label in SIMILARITY_BINS:
        if lo <= value < hi:
            return label
    return None


def _mean(values):
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def wilson_interval(successes, n, z=1.96):
    """95% CI (low, high) for a proportion; None if n == 0."""
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def align_substitutions(pred, truth):
    """Levenshtein backtrace -> list of (gt_char, pred_char) substitution pairs
    and (idx, gt_char) deletions. Insertions ignored (extra chars)."""
    m, n = len(truth), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (truth[i - 1] != pred[j - 1]))
    subs, dels = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (truth[i - 1] != pred[j - 1]):
            if truth[i - 1] != pred[j - 1]:
                subs.append((truth[i - 1], pred[j - 1], i - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels.append((truth[i - 1], i - 1))
            i -= 1
        else:
            j -= 1  # insertion
    return subs, dels


def _position(idx, length):
    if idx < length / 3:
        return 'depan'
    if idx >= 2 * length / 3:
        return 'belakang'
    return 'tengah'


def mcnemar_exact(b, c):
    """Two-sided exact McNemar sign test; None when no discordant pairs."""
    n = b + c
    if n == 0:
        return None
    lo = min(b, c)
    p_one = sum(math.comb(n, k) for k in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * p_one)


def _engine_texts(photo):
    """{engine: (text, conf)} from stored system_result."""
    out = {}
    for e in (photo.system_result.get('ocr_all') or []):
        out[e.get('engine', '?')] = (e.get('text', ''), e.get('conf'))
    return out


def analyze_ocr_similarity(batch) -> dict:
    """Similarity + accuracy analysis per engine for a batch."""
    engines = {}
    overall = {'n_judged': 0, 'exact': 0, 'wrong_but_similar_80': 0,
               'sim_nospasi_values': [], 'sim_spasi_values': []}
    wrong_but_similar = []
    confusions = {}          # (gt, pred) -> count
    positions = {'depan': 0, 'tengah': 0, 'belakang': 0}
    by_length = {}           # len(gt) -> {'n', 'errors'}
    per_photo = {}           # photo_id -> {engine: ok} for McNemar + hard scenario

    for j in batch.photos.select_related('annotation', 'annotation__photo').prefetch_related(
            'annotation__engine_judgments').all().order_by('uploaded_at'):
        ann = getattr(j, 'annotation', None)
        if ann is None:
            continue
        gt = re.sub(r'\s+', '', ann.gt_text or '')
        texts = _engine_texts(j)
        detected_ok = ann.detected_ok == 'BENAR'
        hard = None  # computed below, shared across this photo's engines
        oks = {}
        for judgment in ann.engine_judgments.all():
            e = judgment.engine
            oks[e] = judgment.ok == 'BENAR'
            data = engines.setdefault(e, {
                'n': 0, 'exact': 0, 'wrong': 0,
                'sim_nospasi_values': [], 'sim_spasi_values': [],
                'ratio_nospasi_values': [], 'ratio_spasi_values': [],
                'cer_values': [], 'char_acc_values': [], 'conf_values': [],
                'conf_sim_points': [],
                'hard_n': 0, 'hard_exact': 0, 'overconfident': 0,
                'e2e_n': 0, 'e2e_success': 0, 'e2e_det_wrong': 0, 'e2e_det_fail': 0,
                'bins_nospasi': {label: 0 for _, _, label in SIMILARITY_BINS},
                'bins_spasi': {label: 0 for _, _, label in SIMILARITY_BINS},
            })

            data['n'] += 1
            overall['n_judged'] += 1
            is_exact = judgment.ok == 'BENAR'
            if is_exact:
                data['exact'] += 1
                overall['exact'] += 1
            else:
                data['wrong'] += 1

            sim_ns = judgment.sim_nospasi
            sim_sp = judgment.sim_spasi
            ratio_ns = judgment.ratio_nospasi
            ratio_sp = judgment.ratio_spasi

            # Fallback: old judgments may not store ratio; derive approximate from %.
            if ratio_ns is None and sim_ns is not None:
                ratio_ns = sim_ns / 100.0
            if ratio_sp is None and sim_sp is not None:
                ratio_sp = sim_sp / 100.0

            if sim_ns is not None:
                data['sim_nospasi_values'].append(sim_ns)
                overall['sim_nospasi_values'].append(sim_ns)
                data['bins_nospasi'][_bin_label(sim_ns)] += 1
            if sim_sp is not None:
                data['sim_spasi_values'].append(sim_sp)
                overall['sim_spasi_values'].append(sim_sp)
                data['bins_spasi'][_bin_label(sim_sp)] += 1
            if ratio_ns is not None:
                data['ratio_nospasi_values'].append(ratio_ns)
            if ratio_sp is not None:
                data['ratio_spasi_values'].append(ratio_sp)

            # CER / char accuracy per engine (spasi-insensitive, like sim_nospasi).
            text, conf = texts.get(e, ('', None))
            if conf is not None:
                data['conf_values'].append(conf)
            if conf is not None and sim_ns is not None:
                data['conf_sim_points'].append({'x': conf, 'y': sim_ns})
            pred_ns = re.sub(r'\s+', '', text)
            if gt and text:
                dist = levenshtein_distance(pred_ns, gt)
                cer = dist / len(gt)
                data['cer_values'].append(cer)
                data['char_acc_values'].append(max(0.0, 1 - cer))

            # Error pattern: substitutions + positions + length breakdown.
            if gt and text and not is_exact:
                subs, dels = align_substitutions(pred_ns, gt)
                for gt_ch, pred_ch, idx in subs:
                    confusions[(gt_ch, pred_ch)] = confusions.get((gt_ch, pred_ch), 0) + 1
                    positions[_position(idx, len(gt))] += 1
                for gt_ch, idx in dels:
                    confusions[(gt_ch, '-')] = confusions.get((gt_ch, '-'), 0) + 1
                    positions[_position(idx, len(gt))] += 1
                entry = by_length.setdefault(len(gt), {'n': 0, 'errors': 0})
                entry['n'] += 1
                entry['errors'] += len(subs) + len(dels)

            # Over-confident: high OCR confidence tapi SALAH.
            if conf is not None and conf >= OVERCONFIDENT_THRESHOLD and not is_exact:
                data['overconfident'] += 1

            # End-to-end per engine: butuh deteksi benar DAN OCR benar.
            if ann.gt_presence == 'ADA' and ann.detected_ok in ('BENAR', 'GAGAL'):
                data['e2e_n'] += 1
                if detected_ok and is_exact:
                    data['e2e_success'] += 1
                elif detected_ok:
                    data['e2e_det_wrong'] += 1
                else:
                    data['e2e_det_fail'] += 1

            if judgment.ok == 'SALAH' and sim_ns is not None and sim_ns >= 80:
                overall['wrong_but_similar_80'] += 1
                wrong_but_similar.append({
                    'photo': j,
                    'engine': e,
                    'text': text,
                    'gt': judgment.annotation.gt_text,
                    'sim_nospasi': sim_ns,
                    'sim_spasi': sim_sp,
                })
        if oks:
            per_photo[j.id] = oks

    # Skenario sulit: foto dimana minimal satu engine benar dan satu salah.
    for oks in per_photo.values():
        vals = set(oks.values())
        if len(vals) > 1:
            for e, ok in oks.items():
                if e in engines:
                    engines[e]['hard_n'] += 1
                    if ok:
                        engines[e]['hard_exact'] += 1

    # Quartiles + CI per engine.
    out_engines = {}
    for e, d in engines.items():
        q = None
        if d['sim_nospasi_values']:
            s = sorted(d['sim_nospasi_values'])
            if len(s) < 2:
                q = [s[0]] * 4
            else:
                inc = statistics.quantiles(s, n=4, method='inclusive')
                q = [s[0], inc[0], inc[2], s[-1]]
        lo, hi = wilson_interval(d['exact'], d['n'])
        out_engines[e] = {
            'n': d['n'],
            'exact': d['exact'],
            'wrong': d['wrong'],
            'exact_rate': d['exact'] / d['n'] if d['n'] else None,
            'ci_low': lo, 'ci_high': hi,
            'mean_sim_nospasi': _mean(d['sim_nospasi_values']),
            'mean_sim_spasi': _mean(d['sim_spasi_values']),
            'mean_ratio_nospasi': _mean(d['ratio_nospasi_values']),
            'mean_ratio_spasi': _mean(d['ratio_spasi_values']),
            'cer_mean': _mean(d['cer_values']),
            'char_accuracy': _mean(d['char_acc_values']),
            'mean_conf': _mean(d['conf_values']),
            'quartiles_nospasi': q,
            'hard': {'n': d['hard_n'], 'exact': d['hard_exact']},
            'overconfident': {'n': d['n'], 'count': d['overconfident']},
            'e2e': {'n': d['e2e_n'], 'success': d['e2e_success'],
                    'detected_ok_but_wrong': d['e2e_det_wrong'],
                    'detection_failed': d['e2e_det_fail']},
            'bins_nospasi': d['bins_nospasi'],
            'bins_spasi': d['bins_spasi'],
            'conf_sim_points': d['conf_sim_points'],
        }

    overall_out = {
        'n_judged': overall['n_judged'],
        'exact': overall['exact'],
        'mean_sim_nospasi': _mean(overall['sim_nospasi_values']),
        'mean_sim_spasi': _mean(overall['sim_spasi_values']),
        'wrong_but_similar_80': overall['wrong_but_similar_80'],
    }

    comparisons = {'available': False, 'pairs': []}
    eng_names = sorted(out_engines)
    for e1, e2 in itertools.combinations(eng_names, 2):
        b = sum(1 for oks in per_photo.values()
                if oks.get(e1) and oks.get(e2) is False)
        c = sum(1 for oks in per_photo.values()
                if oks.get(e1) is False and oks.get(e2))
        p = mcnemar_exact(b, c)
        comparisons['pairs'].append({'engines': [e1, e2], 'b': b, 'c': c, 'p_value': p,
                                     'significant': p is not None and p < 0.05})
        if p is not None:
            comparisons['available'] = True

    gt_totals = {}
    for (g, pr), c in confusions.items():
        gt_totals[g] = gt_totals.get(g, 0) + c
    heatmap = {'rows': sorted(gt_totals), 'cols': sorted({pr for _, pr in confusions}),
               'cells': {}}
    for (g, pr), c in confusions.items():
        heatmap['cells'].setdefault(g, {})[pr] = {'n': c, 'alpha': c / gt_totals[g]}

    error_patterns = {
        'confusions': [{'gt': g, 'pred': pr, 'count': c}
                       for (g, pr), c in sorted(confusions.items(),
                                                key=lambda kv: -kv[1])],
        'positions': positions,
        'by_length': dict(sorted(by_length.items())),
        'heatmap': heatmap,
    }

    return {
        'engines': out_engines,
        'overall': overall_out,
        'wrong_but_similar': wrong_but_similar,
        'error_patterns': error_patterns,
        'comparisons': comparisons,
    }


def chart_data(analysis: dict) -> dict:
    """Return JSON-ready chart data for similarity bin distribution."""
    labels = [label for _, _, label in SIMILARITY_BINS]
    datasets = []
    for eng in sorted(analysis['engines'].keys()):
        d = analysis['engines'][eng]
        datasets.append({
            'label': eng,
            'data': [d['bins_nospasi'].get(label, 0) for label in labels],
        })
    datasets_spasi = []
    for eng in sorted(analysis['engines'].keys()):
        d = analysis['engines'][eng]
        datasets_spasi.append({
            'label': eng,
            'data': [d['bins_spasi'].get(label, 0) for label in labels],
        })
    return {'labels': labels, 'datasets': datasets, 'datasets_spasi': datasets_spasi}


def chart_data_all(analysis: dict) -> dict:
    """Chart-ready data: exact-rate bar, sim quartiles, conf-vs-sim scatter."""
    engines = analysis['engines']
    names = sorted(engines)
    exact = {'labels': names,
             'data': [round((engines[e]['exact_rate'] or 0) * 100, 1) for e in names]}
    quartiles = {'labels': names,
                 'datasets': [{'label': e, 'data': engines[e]['quartiles_nospasi']}
                              for e in names if engines[e]['quartiles_nospasi']]}
    scatter = {'datasets': [{'label': e, 'data': engines[e]['conf_sim_points']}
                            for e in names]}
    return {'bins': chart_data(analysis), 'exact': exact,
            'quartiles': quartiles, 'scatter': scatter}
