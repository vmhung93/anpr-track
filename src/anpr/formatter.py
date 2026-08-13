import re

# Known VN plate character layouts, after stripping to alnum-only uppercase:
#   8 chars: 2 digits + 1 letter + 5 digits           (e.g. 29A12345 -> car)
#   9 chars: 2 digits + 1 letter + 1 digit + 5 digits  (e.g. 29A112345 -> motorbike)
PLATE_RE_8 = re.compile(r"^\d{2}[A-Z]\d{5}$")
PLATE_RE_9 = re.compile(r"^\d{2}[A-Z]\d{6}$")


def vn_plate_formater(cleaned_text):
    """
    Formats a raw alphanumeric string into a standard Vietnamese license
    plate format (e.g., 29A-123.45 or 29A1-123.45).

    Rejects anything that doesn't match a known VN plate character layout
    by returning "" (no read), instead of silently formatting an OCR
    misread that merely happens to be 8 or 9 characters long.
    """
    length = len(cleaned_text)

    if length == 8 and PLATE_RE_8.match(cleaned_text):
        # Example: 29A12345 -> 29A-123.45
        return f"{cleaned_text[:3]}-{cleaned_text[3:6]}.{cleaned_text[6:]}"
    elif length == 9 and PLATE_RE_9.match(cleaned_text):
        # Example: 29A112345 -> 29A1-123.45 (Motorbike 5-digit)
        return f"{cleaned_text[:4]}-{cleaned_text[4:7]}.{cleaned_text[7:]}"
    else:
        # Doesn't match a known VN plate layout (wrong length, or right
        # length but wrong character classes) -> no read.
        return ""
