"""Backfill kolom ratio_nospasi/ratio_spasi pada EngineJudgment lama.

Penilaian yang dibuat sebelum migrasi 0005 tidak menyimpan rasio (kolom NULL),
sehingga export CSV/XLSX kosong di kolom ratio. Rasio dihitung ulang persis
seperti forms.save_judgments: similarity_ratio(teks_engine_tanpa_spasi, gt_tanpa_spasi).
Bila teks engine tidak lagi tersedia (foto diproses ulang), fallback sim/100.
"""
import re

from django.core.management.base import BaseCommand

from testing.models import EngineJudgment
from testing.services.metrics import similarity_ratio


class Command(BaseCommand):
    help = 'Isi ratio_nospasi/ratio_spasi EngineJudgment yang masih kosong.'

    def handle(self, *args, **options):
        updated = skipped_no_data = 0
        for j in EngineJudgment.objects.filter(
                ratio_nospasi=None).exclude(sim_nospasi=None).select_related(
                'annotation', 'annotation__photo'):
            ann = j.annotation
            gt = re.sub(r'\s+', '', ann.gt_text or '')
            texts = {e.get('engine'): e.get('text', '')
                     for e in (ann.photo.system_result.get('ocr_all') or [])}
            text = texts.get(j.engine)
            if text and gt:
                j.ratio_nospasi = similarity_ratio(re.sub(r'\s+', '', text), gt)
                if j.sim_spasi is not None and text:
                    j.ratio_spasi = similarity_ratio(text, ann.gt_text or '')
            else:
                j.ratio_nospasi = j.sim_nospasi / 100.0
                if j.sim_spasi is not None:
                    j.ratio_spasi = j.sim_spasi / 100.0
                else:
                    skipped_no_data += 1
            j.save(update_fields=['ratio_nospasi', 'ratio_spasi'])
            updated += 1
        self.stdout.write(f'{updated} penilaian diperbarui; '
                          f'{skipped_no_data} tanpa sim_spasi (ratio_spasi tetap kosong).')
