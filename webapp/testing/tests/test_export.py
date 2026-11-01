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


class ExportOcrAllColumnsTest(TestCase):
    def test_engine_columns_present(self):
        batch = TestBatch.objects.create(name='BX')
        s = sr()
        s['ocr_all'] = [
            {'engine': 'easyocr', 'text': 'DR 1282 AS', 'conf': 0.32, 'fragments': [['DR 1282 AS', 0.32]]},
            {'engine': 'paddleocr', 'text': 'DR 1282 AS', 'conf': 0.99, 'fragments': [['DR 1282 AS', 0.99]]},
            {'engine': 'tesseract', 'text': '', 'conf': 0.0, 'fragments': []},
            {'engine': 'glm-ocr', 'text': 'DR 1282 AS', 'conf': 1.0, 'fragments': [['DR 1282 AS', 1.0]]},
        ]
        p = Photo.objects.create(batch=batch, original_name='a.jpg', status='done', system_result=s)
        Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                  ocr_ok='BENAR', target_detection_index=0)
        rows = export.rows(batch)
        header = rows[0]
        for col in ('ocr_easyocr', 'ocr_paddleocr', 'ocr_tesseract', 'ocr_glm-ocr',
                    'skor_glm-ocr', 'fragmen_all'):
            self.assertIn(col, header)
        data = rows[1]
        self.assertEqual(data[header.index('ocr_glm-ocr')], 'DR 1282 AS')
        self.assertIn('glm-ocr', data[header.index('fragmen_all')])


class ExportEngineJudgmentColumnsTest(TestCase):
    def test_judgment_columns(self):
        from testing.models import EngineJudgment
        batch = TestBatch.objects.create(name='BJ')
        p = Photo.objects.create(batch=batch, original_name='j.jpg', status='done', system_result=sr())
        ann = Annotation.objects.create(photo=p, gt_presence='ADA', detected_ok='BENAR',
                                        ocr_ok='BENAR', gt_text='DR 1282 AS',
                                        target_detection_index=0)
        EngineJudgment.objects.create(annotation=ann, engine='glm-ocr', ok='SALAH',
                                      wrong_chars=2, note='E jadi 8')
        rows = export.rows(batch)
        header = rows[0]
        i = header.index('nilai_glm-ocr')
        self.assertEqual(rows[1][i], 'SALAH')
        self.assertEqual(rows[1][i + 1], 2)
        self.assertEqual(rows[1][i + 2], 'E jadi 8')
