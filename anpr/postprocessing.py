"""
anpr/postprocessing.py
Post-processing for Indonesian license plate OCR output.
"""
import re

# Indonesian plate regex: 1-2 letters, 1-4 digits, 0-3 trailing letters
PLATE_REGEX = re.compile(r'^[A-Z]{1,2}\s*[0-9]{1,4}\s*[A-Z]{0,3}$')

# Characters allowed in Indonesian plates
ALLOWED = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')


def clean_text(raw: str) -> str:
    """Uppercase, strip accents/special chars, keep only allowed characters."""
    text = raw.upper().strip()
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_ambiguous_chars(text: str) -> str:
    """
    Heuristic: replace O/I with 0/1 only in the numeric segment of the plate.
    Because Indonesian plates have fixed format [Letters] [Digits] [Letters],
    we locate the digit block and apply replacement only there.
    """
    parts = text.split()
    if len(parts) >= 2:
        # Find the numeric part (middle segment)
        for i in range(len(parts)):
            if parts[i].isdigit():
                parts[i] = parts[i].replace('O', '0').replace('I', '1').replace('Q', '0')
                break
    return ' '.join(parts)


def is_valid_format(text: str) -> bool:
    return bool(PLATE_REGEX.match(text))


def normalize_plate_text(raw: str) -> str:
    text = clean_text(raw)
    text = normalize_ambiguous_chars(text)
    return text
