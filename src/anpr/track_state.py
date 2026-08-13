from collections import Counter, defaultdict


class TrackState:
    """Per-track_id plate frame-voting buffer, plus OCR-attempt throttling.

    Keeps a short window of validated (regex-passed) OCR reads per track and
    locks in the majority result once it collects min_vote_count votes,
    instead of trusting the first non-empty read. Also tracks when a track
    was last attempted so the pipeline can skip re-running detection/OCR on
    every single frame.
    """

    def __init__(self, vote_buffer_size=5, min_vote_count=2):
        self.vote_buffer_size = vote_buffer_size
        self.min_vote_count = min_vote_count
        self._votes = defaultdict(list)  # track_id -> [(plate_text, confidence), ...]
        self._confirmed = {}  # track_id -> {"text", "confidence", "frame_number"}
        self._last_attempt_frame = {}  # track_id -> frame_idx

    def has_plate(self, track_id):
        return track_id in self._confirmed

    def get(self, track_id):
        record = self._confirmed.get(track_id)
        return record["text"] if record else ""

    def get_record(self, track_id):
        """Full confirmed record ({text, confidence, frame_number}), or None."""
        return self._confirmed.get(track_id)

    def should_attempt(self, track_id, frame_idx, retry_interval_frames):
        """Whether it's time to re-run plate detection/OCR for this track.

        Always true the first time a track is seen; after that, throttled to
        at most once every retry_interval_frames.
        """
        last = self._last_attempt_frame.get(track_id)
        return last is None or (frame_idx - last) >= retry_interval_frames

    def record_attempt(self, track_id, frame_idx):
        self._last_attempt_frame[track_id] = frame_idx

    def add_vote(self, track_id, plate_text, confidence, frame_idx):
        """Record a validated OCR read and lock it in if it now has the majority.

        Returns True the moment this call causes the track to become
        confirmed (i.e. the first frame it's safe to persist/snapshot).
        """
        if not plate_text or track_id in self._confirmed:
            return False

        votes = self._votes[track_id]
        votes.append((plate_text, confidence))
        if len(votes) > self.vote_buffer_size:
            votes.pop(0)

        counts = Counter(text for text, _ in votes)
        best_text, best_count = counts.most_common(1)[0]
        if best_count >= self.min_vote_count:
            matching_confidences = [c for t, c in votes if t == best_text]
            self._confirmed[track_id] = {
                "text": best_text,
                "confidence": sum(matching_confidences) / len(matching_confidences),
                "frame_number": frame_idx,
            }
            return True
        return False

    def items(self):
        return ((track_id, record["text"]) for track_id, record in self._confirmed.items())
