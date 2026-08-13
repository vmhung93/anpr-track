from ultralytics import YOLO


class PlateDetector:
    """Secondary YOLO26 model that finds license plates within a vehicle crop."""

    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, vehicle_crop):
        return self.model(vehicle_crop, conf=self.confidence, verbose=False)
