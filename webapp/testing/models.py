from pathlib import Path

from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone


class TestBatch(models.Model):
    STATUS_CHOICES = [('aktif', 'Aktif'), ('selesai', 'Selesai')]
    name = models.CharField('Nama batch', max_length=200)
    description = models.TextField('Deskripsi', blank=True, default='')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='aktif')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def photo_count(self):
        return self.photos.count()

    def annotated_count(self):
        return self.photos.filter(annotation__isnull=False).count()


class Photo(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Menunggu proses'),
        ('processing', 'Diproses'),
        ('done', 'Selesai'),
        ('error', 'Gagal'),
    ]
    batch = models.ForeignKey(TestBatch, related_name='photos', on_delete=models.CASCADE)
    model = models.ForeignKey('YoloModel', related_name='photos', null=True, blank=True,
                              on_delete=models.SET_NULL)
    image = models.FileField(upload_to='photos/%Y%m/')
    annotated_image = models.FileField(upload_to='annotated/%Y%m/', null=True, blank=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='queued')
    error_message = models.TextField(blank=True, default='')
    system_result = models.JSONField(default=dict, blank=True)
    result_version = models.PositiveIntegerField(default=1)
    original_name = models.CharField(max_length=255, blank=True, default='')
    width = models.PositiveIntegerField(default=0)
    height = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['uploaded_at']

    def __str__(self):
        return self.original_name or f'Photo #{self.pk}'

    @property
    def operational_boxes(self):
        return self.system_result.get('operational_boxes', [])

    @property
    def operational_detections(self):
        """Detections at/above the operational threshold, with their stored index."""
        return [d for d in self.system_result.get('detections', [])
                if d['index'] in self.system_result.get('operational_boxes', [])]

    def annotation_stale(self):
        ann = getattr(self, 'annotation', None)
        return bool(ann and ann.stale)

    def reprocess(self):
        """Mark photo for re-running inference; invalidates existing annotation."""
        self.status = 'queued'
        self.error_message = ''
        self.result_version += 1
        self.save(update_fields=['status', 'error_message', 'result_version'])
        ann = getattr(self, 'annotation', None)
        if ann is not None and not ann.stale:
            ann.stale = True
            ann.save(update_fields=['stale'])


class Annotation(models.Model):
    GT_PRESENCE_CHOICES = [('', '-'), ('ADA', 'Ada plat'), ('TIDAK_ADA', 'Tidak ada plat')]
    DETECTED_CHOICES = [('', '-'), ('BENAR', 'Benar'), ('GAGAL', 'Gagal'), ('TIDAK_RELEVAN', 'Tidak relevan')]
    OCR_CHOICES = [('', '-'), ('BENAR', 'Benar'), ('SALAH', 'Salah'),
                   ('TAK_TERDETEKSI', 'Tak terdeteksi'), ('TIDAK_RELEVAN', 'Tidak relevan')]

    photo = models.OneToOneField(Photo, related_name='annotation', on_delete=models.CASCADE)
    gt_presence = models.CharField(max_length=12, choices=GT_PRESENCE_CHOICES, blank=True, default='')
    detected_ok = models.CharField(max_length=15, choices=DETECTED_CHOICES, blank=True, default='')
    target_detection_index = models.PositiveIntegerField(null=True, blank=True)
    ocr_ok = models.CharField(max_length=15, choices=OCR_CHOICES, blank=True, default='')
    gt_text = models.CharField(max_length=40, blank=True, default='')
    note = models.TextField(blank=True, default='')
    multi_plate_confirmed = models.BooleanField(default=False)
    stale = models.BooleanField(default=False)
    annotated_at = models.DateTimeField(auto_now=True)

    def is_complete(self):
        if not (self.gt_presence and self.detected_ok and self.ocr_ok):
            return False
        if self.ocr_ok == 'SALAH' and not self.gt_text:
            return False
        return True

    def clean(self):
        if self.ocr_ok == 'SALAH' and not (self.gt_text or '').strip():
            raise ValidationError({'gt_text': 'Bila karakter terbaca SALAH, teks plat yang sebenarnya wajib diisi.'})

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


class ConsistencyRun(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Menunggu'),
        ('running', 'Berjalan'),
        ('done', 'Selesai'),
        ('error', 'Gagal'),
        ('cancelled', 'Dibatalkan'),
    ]
    photo = models.ForeignKey(Photo, related_name='consistency_runs', on_delete=models.CASCADE)
    n = models.PositiveIntegerField(default=100)
    seed = models.PositiveIntegerField(default=42)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='queued')
    cancel_requested = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, default='')
    summary = models.JSONField(default=dict, blank=True)
    runs = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Konsistensi {self.photo} (n={self.n}, {self.status})'


class YoloModel(models.Model):
    """Registered YOLO weight file (.pt). Exactly one is_active at a time; used
    for all NEW processed photos. Photo.model records which one produced each result."""
    name = models.CharField('Nama model', max_length=120, unique=True)
    weights = models.FileField(upload_to='models/', null=True, blank=True)
    external_path = models.CharField(max_length=500, blank=True, default='',
                                     help_text='Path .pt di disk (untuk model bawaan riset)')
    is_active = models.BooleanField(default=False)
    notes = models.CharField(max_length=200, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_active', 'name']

    def __str__(self):
        return self.name

    def effective_path(self):
        if self.weights:
            return self.weights.path
        return self.external_path

    @classmethod
    def ensure_default(cls):
        if cls.objects.exists():
            return cls.objects.filter(is_active=True).first() or cls.objects.first()
        from django.conf import settings
        defaults = {'is_active': True,
                    'external_path': settings.ANPR['WEIGHTS'],
                    'notes': 'Model bawaan hasil training riset (dari settings ANPR.WEIGHTS)'}
        name = Path(settings.ANPR['WEIGHTS']).stem if settings.ANPR['WEIGHTS'] else 'best'
        m, _ = cls.objects.get_or_create(name=f'bawaan ({name})', defaults=defaults)
        return m


class EngineJudgment(models.Model):
    """Penilaian manual hasil OCR per engine pada satu foto."""
    OK_CHOICES = [('BENAR', 'Benar'), ('SALAH', 'Salah')]
    annotation = models.ForeignKey(Annotation, related_name='engine_judgments',
                                   on_delete=models.CASCADE)
    engine = models.CharField(max_length=30)
    ok = models.CharField(max_length=6, choices=OK_CHOICES)
    wrong_chars = models.PositiveIntegerField(null=True, blank=True,
                                              help_text='Jumlah karakter salah; kosong = dihitung otomatis')
    note = models.CharField(max_length=300, blank=True, default='')
    judged_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('annotation', 'engine')
        ordering = ['engine']

    def __str__(self):
        return f'{self.engine}: {self.ok}'
