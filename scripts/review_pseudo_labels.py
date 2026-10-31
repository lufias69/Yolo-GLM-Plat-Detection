"""
scripts/review_pseudo_labels.py
Review pseudo OCR labels: print regex-invalid entries and save a correction template.
Usage:
    venv/Scripts/python scripts/review_pseudo_labels.py --input ground_truth_test_pseudo_easyocr.csv
"""
import argparse
import csv
import re
from pathlib import Path

PLATE_REGEX = re.compile(r'^[A-Z]{1,2}\s*[0-9]{1,4}\s*[A-Z]{0,3}$')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='ground_truth_test_pseudo_easyocr.csv')
    parser.add_argument('--output', default='ground_truth_test_review.csv')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    in_path = base / args.input
    out_path = base / args.output

    rows = []
    with open(in_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['plate_text'].upper().strip()
            valid = bool(PLATE_REGEX.match(text))
            row['valid_format'] = 'yes' if valid else 'no'
            rows.append(row)

    invalid = [r for r in rows if r['valid_format'] == 'no']
    print(f'Total rows: {len(rows)}')
    print(f'Valid format: {len(rows) - len(invalid)}')
    print(f'Invalid format: {len(invalid)}')
    print('\nInvalid entries (need manual review):')
    for r in invalid[:30]:
        print(f"  {r['crop_file']:60s} -> '{r['plate_text']}'")

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved review CSV to {out_path}')


if __name__ == '__main__':
    main()
