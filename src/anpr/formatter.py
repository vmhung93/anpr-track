def vn_plate_formater(cleaned_text):
    """
    Attempts to format a raw alphanumeric string into a standard
    Vietnamese license plate format (e.g., 29A1-123.45).
    """
    length = len(cleaned_text)

    # Standard format is usually 8 or 9 characters
    if length == 8:
        # Example: 29A12345 -> 29A-123.45
        return f"{cleaned_text[:3]}-{cleaned_text[3:6]}.{cleaned_text[6:]}"
    elif length == 9:
        # Example: 29A112345 -> 29A1-123.45 (Motorbike 5-digit)
        return f"{cleaned_text[:4]}-{cleaned_text[4:7]}.{cleaned_text[7:]}"
    else:
        # If it doesn't match standard lengths (maybe an old format or OCR error),
        # return the raw cleaned text.
        return cleaned_text
