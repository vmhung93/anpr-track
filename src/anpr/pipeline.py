import os
import time

import cv2

from anpr.deskew import find_plate_corners_from_crop, four_point_transform
from anpr.ocr import vn_plate_parser
from anpr.plate_detector import PlateDetector
from anpr.snapshot import save_snapshot
from anpr.track_state import TrackState
from anpr.vehicle_detector import VehicleDetector

PROGRESS_LOG_INTERVAL = 30  # log a progress line every N frames


def run(video_path, output_path, config):
    """Orchestrate the per-frame detect -> track -> OCR loop over a video file."""
    print(f"Loading vehicle detector: {config['models']['vehicle_detector']}")
    vehicle_detector = VehicleDetector(
        config["models"]["vehicle_detector"], config["detection"]["target_classes"]
    )
    print(f"Loading plate detector: {config['models']['plate_detector']}")
    plate_detector = PlateDetector(config["models"]["plate_detector"])
    track_state = TrackState()

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

    print(
        f"Input video: {video_path} ({width}x{height} @ {fps} fps, {total_frames} frames)"
    )
    print(f"Output video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()

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

            for box, track_id, _cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = box

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

                # If we haven't successfully read this vehicle's plate yet
                if not track_state.has_plate(track_id):
                    # Crop the vehicle from the frame
                    vehicle_crop = frame[y1:y2, x1:x2]

                    # Ensure the crop is valid
                    if vehicle_crop.size != 0:
                        # Detect plate within the vehicle crop
                        plate_results = plate_detector.detect(vehicle_crop)

                        if len(plate_results[0].boxes) > 0:
                            # Get the highest confidence plate box
                            px1, py1, px2, py2 = (
                                plate_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
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
                            text = vn_plate_parser(plate_crop)

                            if (
                                len(text) > 5
                            ):  # Basic validation: Vietnamese plates have 7-9 characters
                                track_state.set(track_id, text)
                                print(
                                    f"[frame {frame_idx}] Track {track_id}: plate confirmed -> {text}"
                                )

                # If we have a plate for this ID, display it
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
            print(
                f"[frame {frame_idx}] progress {progress} | "
                f"{processing_fps:.1f} fps | {len(track_state.items())} plates confirmed so far"
            )

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"Finished processing {frame_idx} frames in {elapsed:.1f}s")

    plates = list(track_state.items())
    print(f"Total Unique Vehicles Detected: {len(plates)}")
    print("Detected Plates:")
    for vid, plate in plates:
        if plate != "":
            print(f"Vehicle {vid}: {plate}")
