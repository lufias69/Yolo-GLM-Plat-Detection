from django.core.exceptions import ValidationError
from django.test import TestCase

from testing.models import Annotation, Photo, TestBatch


class AnnotationModelTest(TestCase):
    def setUp(self):
        self.batch = TestBatch.objects.create(name='Batch A')
        self.photo = Photo.objects.create(batch=self.batch, original_name='p1.jpg')

    def test_annotated_count_zero_without_annotation(self):
        self.assertEqual(self.batch.annotated_count(), 0)

    def test_complete_annotation(self):
        ann = Annotation(
            photo=self.photo, gt_presence='ADA', detected_ok='BENAR',
            ocr_ok='BENAR', target_detection_index=0,
        )
        self.assertTrue(ann.is_complete())

    def test_wrong_ocr_requires_gt_text(self):
        ann = Annotation(photo=self.photo, gt_presence='ADA', detected_ok='BENAR',
                         ocr_ok='SALAH', gt_text='')
        with self.assertRaises(ValidationError):
            ann.full_clean()

    def test_wrong_ocr_with_gt_text_is_valid(self):
        ann = Annotation(photo=self.photo, gt_presence='ADA', detected_ok='BENAR',
                         ocr_ok='SALAH', gt_text='B 1234 ABC', target_detection_index=0)
        ann.full_clean()
        self.assertTrue(ann.is_complete())
