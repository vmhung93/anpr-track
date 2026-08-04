from ultralytics import YOLO


class VehicleDetector:
    """YOLO + ByteTrack wrapper for detecting and tracking vehicles."""

    def __init__(self, model_path, target_classes):
        self.model = YOLO(model_path)
        self.target_classes = target_classes

    def track(self, frame):
        return self.model.track(
            frame, classes=self.target_classes, persist=True, verbose=False
        )
