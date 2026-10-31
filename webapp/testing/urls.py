from django.urls import path

from . import views

app_name = 'testing'

urlpatterns = [
    path('', views.home, name='home'),
    path('batch/<int:batch_id>/', views.batch_detail, name='batch_detail'),
    path('batch/<int:batch_id>/upload/', views.upload_photos, name='upload_photos'),
    path('photo/<int:photo_id>/', views.photo_detail, name='photo_detail'),
    path('photo/<int:photo_id>/reprocess/', views.reprocess_photo, name='reprocess_photo'),
    path('api/photo/<int:photo_id>/status/', views.photo_status, name='photo_status'),
]
