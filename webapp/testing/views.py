from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render

from .models import Photo


def home(request):
    return render(request, 'testing/home.html')


def photo_status(request, photo_id):
    photo = get_object_or_404(Photo, pk=photo_id)
    return JsonResponse({'status': photo.status, 'error_message': photo.error_message})
