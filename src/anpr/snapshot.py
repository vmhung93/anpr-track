import os

import cv2


def save_snapshot(image, directory, filename):
    """Write an image to directory/filename, creating the directory if needed.

    Returns the path written to, or None if the image is empty (e.g. a crop
    that landed outside the frame).
    """
    if image is None or image.size == 0:
        return None

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    cv2.imwrite(path, image)
    return path
