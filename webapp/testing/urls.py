from django.urls import path

from . import views

app_name = 'testing'

urlpatterns = [
    path('', views.home, name='home'),
    path('batch/<int:batch_id>/', views.batch_detail, name='batch_detail'),
    path('batch/<int:batch_id>/upload/', views.upload_photos, name='upload_photos'),
    path('batch/<int:batch_id>/report/', views.batch_report_view, name='batch_report'),
    path('batch/<int:batch_id>/report/export.csv', views.batch_export_csv, name='export_csv'),
    path('batch/<int:batch_id>/report/export.xlsx', views.batch_export_xlsx, name='export_xlsx'),
    path('batch/<int:batch_id>/next-unannotated/', views.next_unannotated, name='next_unannotated'),
    path('photo/<int:photo_id>/', views.photo_detail, name='photo_detail'),
    path('photo/<int:photo_id>/annotate/', views.annotate, name='annotate'),
    path('photo/<int:photo_id>/reprocess/', views.reprocess_photo, name='reprocess_photo'),
    path('api/photo/<int:photo_id>/status/', views.photo_status, name='photo_status'),
]
