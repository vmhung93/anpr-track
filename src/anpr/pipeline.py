import logging
import os
import time

import cv2

from anpr.deskew import find_plate_corners_from_crop, four_point_transform
from anpr.ocr import vn_plate_parser
from anpr.plate_detector import PlateDetector
from anpr.snapshot import save_snapshot
from anpr.storage import Storage
from anpr.track_state import TrackState
from anpr.vehicle_detector import VehicleDetector

PROGRESS_LOG_INTERVAL = 30  # log a progress line every N frames

logger = logging.getLogger("anpr")


def _setup_logging(logs_dir):
    """Log to both the console and a timestamped file under logs_dir."""
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"anpr_{time.strftime('%Y%m%d_%H%M%S')}.log")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(file_handler)

    return log_path


def run(video_path, output_path, config):
    """Orchestrate the per-frame detect -> track -> OCR loop over a video file."""
    log_path = _setup_logging(config.get("logs", {}).get("dir", "logs"))
    logger.info(f"Logging to {log_path}")

    detection_config = config["detection"]
    ocr_config = config.get("ocr", {})
    output_config = config.get("output", {})

    logger.info(f"Loading vehicle detector: {config['models']['vehicle_detector']}")
    vehicle_detector = VehicleDetector(
        config["models"]["vehicle_detector"],
        detection_config["target_classes"],
        confidence=detection_config.get("vehicle_confidence", 0.5),
    )
    logger.info(f"Loading plate detector: {config['models']['plate_detector']}")
    plate_detector = PlateDetector(
        config["models"]["plate_detector"],
        confidence=detection_config.get("plate_confidence", 0.5),
    )

    min_crop_px = detection_config.get("min_vehicle_crop_px", 80)
    ocr_confidence_threshold = ocr_config.get("confidence_threshold", 0.5)
    retry_interval_frames = ocr_config.get("retry_interval_frames", 5)

    track_state = TrackState(
        vote_buffer_size=ocr_config.get("vote_buffer_size", 5),
        min_vote_count=ocr_config.get("vote_min_count", 2),
    )

    snapshot_dir = output_config.get("snapshot_dir", "output/snapshots")
    storage = Storage(output_config.get("db_path", "output/detections.db"))
    video_source = os.path.basename(video_path)

    debug_config = config.get("debug", {})
    debug_enabled = debug_config.get("enabled", False)
    debug_dir = debug_config.get("dir", "output/debug")
    debug_frame_dir = os.path.join(debug_dir, "frames")
    debug_vehicle_dir = os.path.join(debug_dir, "vehicles")
    debug_plate_dir = os.path.join(debug_dir, "plates")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Input video: {video_path} ({width}x{height} @ {fps} fps, {total_frames} frames)"
    )
    logger.info(f"Output video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    seen_track_ids = set()
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if debug_enabled:
                save_snapshot(frame, debug_frame_dir, f"frame{frame_idx}.jpg")

            # persist=True enables the tracker (ByteTrack/BoT-SORT)
            results = vehicle_detector.track(frame)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                class_names = results[0].names

                for box, track_id, cls in zip(boxes, ids, classes):
                    x1, y1, x2, y2 = box
                    seen_track_ids.add(int(track_id))
                    vehicle_class = class_names.get(int(cls), str(cls))

                    # Draw vehicle bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                    )

                    # If we haven't confirmed this vehicle's plate yet
                    if not track_state.has_plate(track_id):
                        vehicle_crop = frame[y1:y2, x1:x2]
                        crop_h, crop_w = vehicle_crop.shape[:2]
                        big_enough = crop_h >= min_crop_px and crop_w >= min_crop_px
                        due_for_attempt = track_state.should_attempt(
                            track_id, frame_idx, retry_interval_frames
                        )

                        # Skip plate detection/OCR on crops too small/far to
                        # read reliably, and throttle retries on the rest so
                        # a track samples a few different frames instead of
                        # spending compute on every single one.
                        if vehicle_crop.size != 0 and big_enough and due_for_attempt:
                            track_state.record_attempt(track_id, frame_idx)

                            # Detect plate within the vehicle crop
                            plate_results = plate_detector.detect(vehicle_crop)

                            if len(plate_results[0].boxes) > 0:
                                # Get the highest confidence plate box (Ultralytics
                                # does not guarantee boxes are confidence-sorted)
                                plate_boxes = plate_results[0].boxes
                                best_idx = plate_boxes.conf.cpu().numpy().argmax()
                                px1, py1, px2, py2 = (
                                    plate_boxes.xyxy[best_idx].cpu().numpy().astype(int)
                                )
                                plate_crop = vehicle_crop[py1:py2, px1:px2]

                                if debug_enabled:
                                    save_snapshot(
                                        vehicle_crop,
                                        debug_vehicle_dir,
                                        f"frame{frame_idx}_track{track_id}.jpg",
                                    )
                                    save_snapshot(
                                        plate_crop,
                                        debug_plate_dir,
                                        f"frame{frame_idx}_track{track_id}.jpg",
                                    )

                                # Deskew: fall back to the raw crop if 4 corners aren't found
                                corners = find_plate_corners_from_crop(plate_crop)
                                if corners is not None:
                                    plate_crop = four_point_transform(plate_crop, corners)

                                # Read the text
                                text, confidence = vn_plate_parser(
                                    plate_crop,
                                    confidence_threshold=ocr_confidence_threshold,
                                )

                                if text:
                                    just_confirmed = track_state.add_vote(
                                        track_id, text, confidence, frame_idx
                                    )
                                    if just_confirmed:
                                        record = track_state.get_record(track_id)
                                        logger.info(
                                            f"[frame {frame_idx}] Track {track_id}: "
                                            f"plate confirmed -> {record['text']} "
                                            f"(confidence {record['confidence']:.2f})"
                                        )
                                        snapshot_path = save_snapshot(
                                            plate_crop,
                                            snapshot_dir,
                                            f"frame{frame_idx}_track{track_id}_{record['text']}.jpg",
                                        )
                                        storage.insert_detection(
                                            video_source=video_source,
                                            track_id=int(track_id),
                                            vehicle_class=vehicle_class,
                                            plate_text=record["text"],
                                            ocr_confidence=record["confidence"],
                                            frame_number=record["frame_number"],
                                            snapshot_path=snapshot_path,
                                        )

                    # If we have a confirmed plate for this ID, display it
                    if track_state.has_plate(track_id):
                        plate_text = track_state.get(track_id)
                        cv2.putText(
                            frame,
                            f"Plate: {plate_text}",
                            (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

            out.write(frame)

            if frame_idx % PROGRESS_LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                processing_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                progress = (
                    f"{frame_idx}/{total_frames}" if total_frames > 0 else str(frame_idx)
                )
                logger.info(
                    f"[frame {frame_idx}] progress {progress} | "
                    f"{processing_fps:.1f} fps | "
                    f"{len(list(track_state.items()))} plates confirmed so far"
                )
    finally:
        cap.release()
        out.release()
        storage.close()

    elapsed = time.time() - start_time
    logger.info(f"Finished processing {frame_idx} frames in {elapsed:.1f}s")

    plates = list(track_state.items())
    logger.info(f"Total Unique Vehicles Detected: {len(seen_track_ids)}")
    logger.info(f"Plates Confirmed: {len(plates)}")
    for vid, plate in plates:
        logger.info(f"Vehicle {vid}: {plate}")
