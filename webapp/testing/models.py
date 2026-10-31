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
        if self.ocr_ok == 'SALAH' and not self.gt_text:
            raise ValidationError({'gt_text': 'Bila karakter terbaca SALAH, teks plat yang sebenarnya wajib diisi.'})


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
