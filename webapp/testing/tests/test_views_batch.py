from io import BytesIO
from unittest.mock import patch

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from PIL import Image

from testing.models import Photo, TestBatch


def jpeg_file(name='p1.jpg'):
    buf = BytesIO()
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(buf, format='JPEG')
    return SimpleUploadedFile(name, buf.getvalue(), content_type='image/jpeg')


class BatchViewsTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_create_batch_redirects_and_saved(self):
        resp = self.client.post('/', {'name': 'Uji Senin', 'description': 'parkiran'})
        self.assertEqual(resp.status_code, 302)
        batch = TestBatch.objects.get(name='Uji Senin')
        self.assertEqual(batch.description, 'parkiran')
        self.assertRedirects(resp, f'/batch/{batch.id}/')

    def test_home_lists_batches(self):
        TestBatch.objects.create(name='Batch Lama')
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('Batch Lama', resp.content.decode())

    @patch('testing.services.workers.enqueue_photo')
    def test_upload_valid_files_creates_queued_photos(self, mock_enq):
        batch = TestBatch.objects.create(name='B')
        resp = self.client.post(f'/batch/{batch.id}/upload/',
                                {'files': [jpeg_file('a.jpg'), jpeg_file('b.png')]})
        self.assertEqual(resp.status_code, 302)
        photos = list(batch.photos.all())
        self.assertEqual(len(photos), 2)
        self.assertEqual([p.status for p in photos], ['queued', 'queued'])
        self.assertEqual([p.original_name for p in photos], ['a.jpg', 'b.png'])
        self.assertEqual(mock_enq.call_count, 2)

    @patch('testing.services.workers.enqueue_photo')
    def test_upload_rejects_non_image(self, mock_enq):
        batch = TestBatch.objects.create(name='B')
        resp = self.client.post(f'/batch/{batch.id}/upload/',
                                {'files': [SimpleUploadedFile('x.txt', b'hello')]})
        self.assertEqual(batch.photos.count(), 0)

    def test_batch_detail_shows_photo_names(self):
        batch = TestBatch.objects.create(name='B')
        Photo.objects.create(batch=batch, original_name='foto_aneh.jpg', status='done')
        resp = self.client.get(f'/batch/{batch.id}/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('foto_aneh.jpg', resp.content.decode())

    @patch('testing.services.workers.enqueue_photo')
    def test_reprocess_bumps_version_and_enqueues(self, mock_enq):
        batch = TestBatch.objects.create(name='B')
        photo = Photo.objects.create(batch=batch, original_name='p.jpg', status='done')
        resp = self.client.post(f'/photo/{photo.id}/reprocess/')
        self.assertEqual(resp.status_code, 302)
        photo.refresh_from_db()
        self.assertEqual(photo.result_version, 2)
        mock_enq.assert_called_once_with(photo.id)
