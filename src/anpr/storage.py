import os
import sqlite3
from datetime import UTC, datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_source TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    vehicle_class TEXT,
    plate_text TEXT NOT NULL,
    plate_color TEXT,
    ocr_confidence REAL,
    snapshot_path TEXT,
    frame_number INTEGER,
    created_at TEXT NOT NULL
)
"""


class Storage:
    """SQLite-backed log of confirmed plate reads (one row per confirmed read)."""

    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def insert_detection(
        self,
        video_source,
        track_id,
        vehicle_class,
        plate_text,
        ocr_confidence,
        frame_number,
        snapshot_path=None,
        plate_color=None,
    ):
        self.conn.execute(
            """
            INSERT INTO detections (
                video_source, track_id, vehicle_class, plate_text, plate_color,
                ocr_confidence, snapshot_path, frame_number, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_source,
                track_id,
                vehicle_class,
                plate_text,
                plate_color,
                ocr_confidence,
                snapshot_path,
                frame_number,
                datetime.now(UTC).isoformat(),
            ),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
