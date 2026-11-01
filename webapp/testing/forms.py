from django import forms

from .models import Annotation, EngineJudgment

OK_CHOICES = [('BENAR', 'Benar'), ('SALAH', 'Salah')]


class AnnotationForm(forms.ModelForm):
    class Meta:
        model = Annotation
        fields = ['gt_presence', 'detected_ok', 'target_detection_index',
                  'gt_text', 'note', 'multi_plate_confirmed']
        widgets = {
            'note': forms.Textarea(attrs={'rows': 2, 'class': 'form-control'}),
            'gt_text': forms.TextInput(attrs={'class': 'form-control',
                                              'placeholder': 'teks plat yang seharusnya'}),
        }

    def __init__(self, *args, photo=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.photo = photo
        self.operational = photo.operational_detections if photo else []
        self.multiple = len(self.operational) > 1
        self.engines = [e.get('engine') for e in (photo.system_result.get('ocr_all') or [])] if photo else []
        self.main_engine = photo.system_result.get('ocr_engine', 'easyocr') if photo else 'easyocr'
        if photo and not self.engines:
            # foto lama tanpa hasil multi-engine: nilai engine utamanya saja
            self.engines = [self.main_engine]
        self.fields['gt_presence'] = self._radio('gt_presence', Annotation.GT_PRESENCE_CHOICES)
        self.fields['detected_ok'] = self._radio('detected_ok', Annotation.DETECTED_CHOICES)
        self.fields['target_detection_index'] = forms.IntegerField(required=False, min_value=0)
        existing = {}
        if photo is not None and self.instance.pk:
            existing = {j.engine: j for j in self.instance.engine_judgments.all()}
        for eng in self.engines:
            self.fields[f'eng_ok_{eng}'] = forms.ChoiceField(
                choices=OK_CHOICES, required=False, label=eng,
                widget=forms.RadioSelect(attrs={'class': 'btn-check'}),
                initial=existing.get(eng).ok if eng in existing else None)
            self.fields[f'eng_wrong_{eng}'] = forms.IntegerField(
                required=False, min_value=0, label='karakter salah',
                initial=existing.get(eng).wrong_chars if eng in existing else None)
            self.fields[f'eng_note_{eng}'] = forms.CharField(
                required=False, max_length=300, label='catatan',
                widget=forms.TextInput(attrs={'class': 'form-control form-control-sm',
                                              'placeholder': 'opsional'}),
                initial=existing.get(eng).note if eng in existing else '')

    @staticmethod
    def _radio(name, choices):
        clean = [c for c in choices if c[0] != '']
        return forms.ChoiceField(choices=clean,
                                 widget=forms.RadioSelect(attrs={'class': 'form-check-input'}))

    def clean(self):
        data = super().clean()
        gt_text = (data.get('gt_text') or '').strip()
        any_salah = any(data.get(f'eng_ok_{e}') == 'SALAH' for e in self.engines)
        for eng in self.engines:
            if not data.get(f'eng_ok_{eng}'):
                self.add_error(f'eng_ok_{eng}', f'Penilaian engine {eng} wajib diisi.')
        if any_salah and not gt_text:
            self.add_error('gt_text',
                           'Bila ada engine yang SALAH, teks plat yang sebenarnya wajib diisi.')
        if self.multiple:
            if data.get('target_detection_index') is None:
                self.add_error('target_detection_index',
                               'Ditemukan lebih dari 1 deteksi — pilih box target terlebih dahulu.')
            if not data.get('multi_plate_confirmed'):
                self.add_error('multi_plate_confirmed',
                               'Konfirmasi bahwa foto ini 1 motor dan Anda sudah memilih box target.')
        elif data.get('target_detection_index') is None and len(self.operational) == 1:
            data['target_detection_index'] = self.operational[0]['index']
        return data

    def clean_gt_text(self):
        from anpr.postprocessing import normalize_plate_text
        return normalize_plate_text(self.cleaned_data.get('gt_text', ''))

    @property
    def derived_ocr_ok(self):
        """ocr_ok Annotation lama = penilaian engine utama."""
        return self.cleaned_data.get(f'eng_ok_{self.main_engine}') or ''

    def engine_results(self):
        """(engine, text) dari system_result untuk perhitungan wrong_chars."""
        return {e['engine']: e.get('text', '') for e in
                (self.photo.system_result.get('ocr_all') or [])} if self.photo else {}

    def save_judgments(self, annotation):
        import re
        from anpr.postprocessing import normalize_plate_text
        from testing.services.metrics import levenshtein_distance, similarity, similarity_ratio
        results = self.engine_results()
        gt = annotation.gt_text or results.get(self.main_engine, '')
        for eng in self.engines:
            ok = self.cleaned_data.get(f'eng_ok_{eng}')
            wrong = self.cleaned_data.get(f'eng_wrong_{eng}')
            if wrong is None and ok == 'SALAH':
                wrong = levenshtein_distance(normalize_plate_text(results.get(eng, '')), gt)
            text = results.get(eng, '')
            text_nospasi = re.sub(r'\s+', '', text)
            gt_nospasi = re.sub(r'\s+', '', gt)
            EngineJudgment.objects.update_or_create(
                annotation=annotation, engine=eng,
                defaults={'ok': ok, 'wrong_chars': wrong,
                          'note': self.cleaned_data.get(f'eng_note_{eng}', ''),
                          'sim_nospasi': similarity(text_nospasi, gt_nospasi),
                          'sim_spasi': similarity(text, gt),
                          'ratio_nospasi': similarity_ratio(text_nospasi, gt_nospasi),
                          'ratio_spasi': similarity_ratio(text, gt)})
