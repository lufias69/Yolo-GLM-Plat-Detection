from django.urls import path

from . import views

app_name = 'testing'

urlpatterns = [
    path('', views.home, name='home'),
    path('models/', views.models_page, name='models'),
    path('models/upload/', views.model_upload, name='model_upload'),
    path('models/<int:model_id>/activate/', views.model_activate, name='model_activate'),
    path('models/<int:model_id>/delete/', views.model_delete, name='model_delete'),
    path('batch/<int:batch_id>/', views.batch_detail, name='batch_detail'),
    path('batch/<int:batch_id>/upload/', views.upload_photos, name='upload_photos'),
    path('batch/<int:batch_id>/report/', views.batch_report_view, name='batch_report'),
    path('batch/<int:batch_id>/report/export.csv', views.batch_export_csv, name='export_csv'),
    path('batch/<int:batch_id>/report/export.xlsx', views.batch_export_xlsx, name='export_xlsx'),
    path('batch/<int:batch_id>/next-unannotated/', views.next_unannotated, name='next_unannotated'),
    path('photo/<int:photo_id>/', views.photo_detail, name='photo_detail'),
    path('photo/<int:photo_id>/annotate/', views.annotate, name='annotate'),
    path('photo/<int:photo_id>/consistency/', views.photo_consistency, name='photo_consistency'),
    path('photo/<int:photo_id>/consistency/start/', views.consistency_start, name='consistency_start'),
    path('photo/<int:photo_id>/consistency/<int:run_id>/cancel/', views.consistency_cancel, name='consistency_cancel'),
    path('photo/<int:photo_id>/consistency/<int:run_id>/status/', views.consistency_status, name='consistency_status'),
    path('photo/<int:photo_id>/reprocess/', views.reprocess_photo, name='reprocess_photo'),
    path('api/photo/<int:photo_id>/status/', views.photo_status, name='photo_status'),
]
