from django.urls import path

from . import views

app_name = 'testing'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/photo/<int:photo_id>/status/', views.photo_status, name='photo_status'),
]
