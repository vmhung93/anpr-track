class TrackState:
    """Per-track_id plate cooldown: keeps the first successful read for each vehicle."""

    def __init__(self):
        self._plates = {}

    def has_plate(self, track_id):
        return self._plates.get(track_id, "") != ""

    def get(self, track_id):
        return self._plates.get(track_id, "")

    def set(self, track_id, plate_text):
        self._plates[track_id] = plate_text

    def items(self):
        return self._plates.items()
