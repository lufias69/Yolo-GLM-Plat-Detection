import csv
import io

from django.test import TestCase
from openpyxl import load_workbook

from testing.models import Annotation, Photo, TestBatch
from testing.services import export


def sr(conf=0.9, ocr='DR 1282 AS', ops=None):
    return {'detections': [{'index': 0, 'box': [1, 2, 30, 20], 'confidence': conf,
                           'ocr_text': ocr, 'ocr_confidence': 0.8}],
            'operational_boxes': [0] if ops is None else ops,
            'latency': {'detection_ms': 10.0, 'ocr_ms': 5.0, 'total_ms': 15.0},
            'threshold': 0.25}


class ExportTest(TestCase):
    def setUp(self):
        self.batch = TestBatch.objects.create(name='B')
        p = Photo.objects.create(batch=self.batch, original_name='tp.jpg', status='done',
                                 system_result=sr(), width=100, height=50)
        Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0)
        Photo.objects.create(batch=self.batch, original_name='unrated.jpg',
                             status='done', system_result=sr(ops=[]))

    def test_rows_count_and_content(self):
        rows = export.rows(self.batch)
        self.assertEqual(len(rows), 3)  # header + 2 photos
        header = rows[0]
        self.assertIn('gt_presence', header)
        row_tp = rows[1]
        self.assertIn('tp.jpg', row_tp)
        self.assertIn('ADA', row_tp)
        self.assertIn('DR 1282 AS', row_tp)

    def test_csv_delimiter_and_bom(self):
        text = export.to_csv(self.batch)
        self.assertTrue(text.startswith('﻿'))
        reader = csv.reader(io.StringIO(text.lstrip('﻿')), delimiter=';')
        rows = list(reader)
        self.assertEqual(len(rows), 3)

    def test_xlsx_two_sheets_and_na(self):
        data = export.to_xlsx(self.batch)
        wb = load_workbook(io.BytesIO(data))
        self.assertEqual(wb.sheetnames, ['Ringkasan', 'Per Foto'])
        summary = {str(r[0].value): r[1].value for r in
                   wb['Ringkasan'].iter_rows(min_row=2, max_col=2)}
        self.assertEqual(summary.get('roc_auc'), 'N/A')  # no negatives
        self.assertAlmostEqual(float(summary['detection_accuracy']), 1.0)  # only TP annotated
        self.assertEqual(wb['Per Foto'].max_row, 3)
