from paddleocr import PaddleOCR

from anpr.formatter import vn_plate_formater

# Initialize PaddleOCR (do this once outside your loop)
# use_angle_cls=True helps if the bounding box crop is slightly tilted
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def vn_plate_parser(plate_image_crop, confidence_threshold=0.5):
    """
    Reads text from a cropped license plate image and formats it
    for 1-line (car) and 2-line (motorbike) Vietnamese plates.

    Text lines below confidence_threshold are dropped before concatenation
    rather than let a low-confidence guess drag the whole read down.

    Returns (formatted_plate, confidence) -- confidence is the average of
    the kept lines' PaddleOCR scores, or ("", 0.0) if nothing usable was
    read.
    """
    # 1. Run OCR on the crop
    result = ocr.ocr(plate_image_crop, cls=True)

    # PaddleOCR returns a list of lists. If nothing is found, return empty.
    if not result or not result[0]:
        return "", 0.0

    # detections format: [[[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ('text', confidence)]
    detections = [line for line in result[0] if line[1][1] >= confidence_threshold]
    if not detections:
        return "", 0.0

    # 2. Sort detections top-to-bottom based on the Y-coordinate of the bounding box
    detections.sort(key=lambda x: x[0][0][1])

    raw_text = ""
    confidences = []
    for line in detections:
        text, score = line[1]
        raw_text += text
        confidences.append(score)

    # 3. Clean the text: keep only alphanumeric characters
    # This removes rogue spaces, hyphens, or dots the OCR might have guessed
    cleaned_text = "".join(char.upper() for char in raw_text if char.isalnum())

    # 4. Reformat to standard Vietnamese style, e.g. XXA-YYY.YY
    formatted_plate = vn_plate_formater(cleaned_text)
    if not formatted_plate:
        return "", 0.0

    avg_confidence = sum(confidences) / len(confidences)
    return formatted_plate, avg_confidence


# --- Example Usage inside your main loop ---
# Assuming `plate_crop` is the numpy array image of the cropped license plate
# final_plate_text, final_confidence = vn_plate_parser(plate_crop)
# print(f"Detected Plate: {final_plate_text} ({final_confidence:.2f})")
