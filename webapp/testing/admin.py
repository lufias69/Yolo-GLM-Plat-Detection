from django.contrib import admin

from .models import Annotation, ConsistencyRun, Photo, TestBatch


class PhotoInline(admin.TabularInline):
    model = Photo
    extra = 0
    fields = ('original_name', 'status', 'result_version', 'uploaded_at')
    readonly_fields = fields


@admin.register(TestBatch)
class TestBatchAdmin(admin.ModelAdmin):
    list_display = ('name', 'status', 'photo_count', 'annotated_count', 'created_at')
    inlines = [PhotoInline]


@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ('original_name', 'batch', 'status', 'result_version', 'uploaded_at')
    list_filter = ('status', 'batch')


@admin.register(Annotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_display = ('photo', 'gt_presence', 'detected_ok', 'ocr_ok', 'stale', 'annotated_at')
    list_filter = ('gt_presence', 'detected_ok', 'ocr_ok', 'stale')


@admin.register(ConsistencyRun)
class ConsistencyRunAdmin(admin.ModelAdmin):
    list_display = ('photo', 'n', 'status', 'created_at', 'finished_at')
    list_filter = ('status',)
