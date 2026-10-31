from django import forms

from .models import Annotation


class AnnotationForm(forms.ModelForm):
    class Meta:
        model = Annotation
        fields = ['gt_presence', 'detected_ok', 'target_detection_index',
                  'ocr_ok', 'gt_text', 'note', 'multi_plate_confirmed']
        widgets = {
            'gt_presence': forms.RadioSelect(),
            'detected_ok': forms.RadioSelect(),
            'ocr_ok': forms.RadioSelect(),
            'note': forms.Textarea(attrs={'rows': 2, 'class': 'form-control'}),
            'gt_text': forms.TextInput(attrs={'class': 'form-control',
                                              'placeholder': 'teks plat yang seharusnya'}),
        }

    def __init__(self, *args, photo=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.photo = photo
        self.operational = photo.operational_detections if photo else []
        self.multiple = len(self.operational) > 1
        self.fields['target_detection_index'].required = False
        for name in ('gt_presence', 'detected_ok', 'ocr_ok'):
            choices = [c for c in self.fields[name].choices if c[0] != '']
            self.fields[name] = forms.ChoiceField(
                choices=choices, widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
                label=self.fields[name].label)
        self.fields['target_detection_index'] = forms.IntegerField(required=False, min_value=0)

    def clean(self):
        data = super().clean()
        ocr_ok = data.get('ocr_ok')
        gt_text = (data.get('gt_text') or '').strip()
        if ocr_ok == 'SALAH' and not gt_text:
            self.add_error('gt_text', 'Bila karakter terbaca SALAH, teks plat yang sebenarnya wajib diisi.')
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
