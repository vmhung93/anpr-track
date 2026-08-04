from paddleocr import PaddleOCR

from anpr.formatter import vn_plate_formater

# Initialize PaddleOCR (do this once outside your loop)
# use_angle_cls=True helps if the bounding box crop is slightly tilted
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def vn_plate_parser(plate_image_crop):
    """
    Reads text from a cropped license plate image and formats it
    for 1-line (car) and 2-line (motorbike) Vietnamese plates.
    """
    # 1. Run OCR on the crop
    result = ocr.ocr(plate_image_crop, cls=True)

    # PaddleOCR returns a list of lists. If nothing is found, return empty.
    if not result or not result[0]:
        return ""

    detections = result[0]

    # 2. Sort detections top-to-bottom based on the Y-coordinate of the bounding box
    # detections format: [[[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ('text', confidence)]
    detections.sort(key=lambda x: x[0][0][1])

    raw_text = ""
    for line in detections:
        text = line[1][0]
        raw_text += text

    # 3. Clean the text: keep only alphanumeric characters
    # This removes rogue spaces, hyphens, or dots the OCR might have guessed
    cleaned_text = "".join(char.upper() for char in raw_text if char.isalnum())

    # 4. (Optional) Reformat to standard Vietnamese style: XXA-YYYYY
    formatted_plate = vn_plate_formater(cleaned_text)

    return formatted_plate


# --- Example Usage inside your main loop ---
# Assuming `plate_crop` is the numpy array image of the cropped license plate
# final_plate_text = vn_plate_parser(plate_crop)
# print(f"Detected Plate: {final_plate_text}")
