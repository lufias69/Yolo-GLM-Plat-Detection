from io import StringIO

from django.core.management import call_command
from django.test import TestCase

from testing.models import Annotation, EngineJudgment, Photo, TestBatch


class BackfillRatiosTest(TestCase):
    def _judgment(self, sim_ns, sim_sp, ratio_ns=None, ratio_sp=None,
                  text='B128456', gt='B123456'):
        p = Photo.objects.create(batch=self.batch, original_name='x.jpg', status='done',
                                 system_result={'ocr_all': [{'engine': 'easyocr', 'text': text}]})
        ann = Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                        ocr_ok='SALAH', gt_text=gt)
        return EngineJudgment.objects.create(annotation=ann, engine='easyocr', ok='SALAH',
                                             sim_nospasi=sim_ns, sim_spasi=sim_sp,
                                             ratio_nospasi=ratio_ns, ratio_spasi=ratio_sp)

    def setUp(self):
        self.batch = TestBatch.objects.create(name='B')

    def test_recomputes_ratio_from_original_text(self):
        # teks 'B128456' vs GT 'B123456': jarak 1/7 -> ratio presisi 0.8571…
        j = self._judgment(86, 80)  # ratio NULL (penilaian lama)
        call_command('backfill_ocr_ratios', stdout=StringIO())
        j.refresh_from_db()
        self.assertAlmostEqual(j.ratio_nospasi, 6 / 7)
        self.assertAlmostEqual(j.ratio_spasi, 6 / 7)

    def test_falls_back_to_sim_when_text_gone(self):
        j = self._judgment(86, 80)
        # teks engine hilang dari system_result (mis. foto diproses ulang)
        p = j.annotation.photo
        p.system_result = {'ocr_all': []}
        p.save(update_fields=['system_result'])
        call_command('backfill_ocr_ratios', stdout=StringIO())
        j.refresh_from_db()
        self.assertAlmostEqual(j.ratio_nospasi, 0.86)

    def test_existing_ratios_untouched(self):
        j = self._judgment(86, 80, ratio_ns=0.8571, ratio_sp=0.7999)
        call_command('backfill_ocr_ratios', stdout=StringIO())
        j.refresh_from_db()
        self.assertAlmostEqual(j.ratio_nospasi, 0.8571)

    def test_null_sim_skipped(self):
        j = self._judgment(None, None)
        call_command('backfill_ocr_ratios', stdout=StringIO())
        j.refresh_from_db()
        self.assertIsNone(j.ratio_nospasi)
