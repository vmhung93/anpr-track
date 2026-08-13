from ultralytics import YOLO


class VehicleDetector:
    """YOLO + ByteTrack wrapper for detecting and tracking vehicles."""

    def __init__(self, model_path, target_classes, confidence=0.5):
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.confidence = confidence

    def track(self, frame):
        return self.model.track(
            frame,
            classes=self.target_classes,
            conf=self.confidence,
            persist=True,
            verbose=False,
        )
