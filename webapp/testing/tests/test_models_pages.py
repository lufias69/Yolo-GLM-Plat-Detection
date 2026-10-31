from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase

from testing.models import Photo, TestBatch, YoloModel


class YoloModelTest(TestCase):
    def test_ensure_default_creates_active_from_settings(self):
        from django.conf import settings
        m = YoloModel.ensure_default()
        self.assertTrue(m.is_active)
        self.assertEqual(m.effective_path(), str(settings.ANPR['WEIGHTS']))


class ModelsPageTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_page_lists_models_and_upload_form(self):
        YoloModel.ensure_default()
        resp = self.client.get('/models/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('best.pt', resp.content.decode())

    def test_upload_pt_registers_model(self):
        resp = self.client.post('/models/upload/',
                                 {'name': 'yolov8s baru',
                                  'weights': SimpleUploadedFile('v8s.pt', b'fake-pt-bytes',
                                                               content_type='application/octet-stream')})
        self.assertEqual(resp.status_code, 302)
        m = YoloModel.objects.get(name='yolov8s baru')
        self.assertFalse(m.is_active)

    def test_upload_rejects_non_pt(self):
        self.client.post('/models/upload/', {'name': 'jahat',
                                             'weights': SimpleUploadedFile('evil.exe', b'MZ')})
        self.assertFalse(YoloModel.objects.filter(name='jahat').exists())

    def test_activate_switches(self):
        a = YoloModel.ensure_default()
        b = YoloModel.objects.create(name='b', weights=SimpleUploadedFile('b.pt', b'x'))
        self.client.post(f'/models/{b.id}/activate/')
        a.refresh_from_db(); b.refresh_from_db()
        self.assertFalse(a.is_active)
        self.assertTrue(b.is_active)

    def test_cannot_delete_active(self):
        a = YoloModel.ensure_default()
        self.client.post(f'/models/{a.id}/delete/')
        self.assertTrue(YoloModel.objects.filter(pk=a.pk).exists())

    def test_delete_inactive_ok(self):
        YoloModel.ensure_default()
        b = YoloModel.objects.create(name='b', weights=SimpleUploadedFile('b.pt', b'x'))
        self.client.post(f'/models/{b.id}/delete/')
        self.assertFalse(YoloModel.objects.filter(pk=b.pk).exists())
