# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vietnamese license plate detection and tracking. Detects vehicles in a video with YOLO26 + ByteTrack, locates the plate on each vehicle, and reads it with PaddleOCR — covering white (personal), yellow (commercial), and blue (government) plates.

See `implementation-plan.html` for the full phased build-out and known open issues — it's kept in sync with `src/anpr/`, `config.yaml`, and `pyproject.toml`, and is the authoritative source for what's implemented vs. still planned. Read it before making non-trivial changes; several gaps below are already scoped there.

## Commands

```bash
uv sync                                                                  # install deps (editable)
uv run anpr --video path/to/clip.mp4 --config config.yaml --output output/output_tracked.mp4
uv run python train_model.py                                            # train the custom plate detector
```

There is no linter, formatter, or test suite configured yet (`tests/` is empty, `pytest` is not a dependency). Phase 5 of the implementation plan calls for adding `pytest` once `formatter.py`/`track_state.py` get unit-covered.

System dependency: `libgomp.so.1` must be present on the host for `paddlepaddle` to import (`sudo apt-get install -y libgomp1`).

## Architecture

Single per-frame loop in `pipeline.py`, run via `cli.py` (`uv run anpr`), config-driven through `config.yaml` (model paths, target COCO classes, confidence thresholds, output paths — no hardcoded paths in source):

1. **Detect + track vehicles** — `vehicle_detector.py` wraps `YOLO.track(..., persist=True, conf=detection.vehicle_confidence)` (ByteTrack), filtered to COCO classes 2/3/5/7 (car/motorcycle/bus/truck) from `config.yaml`.
2. **Crop + detect plate** — for each tracked vehicle without a confirmed plate yet, `plate_detector.py` runs a second, custom-trained YOLO26 model (`train_model.py` output, `conf=detection.plate_confidence`) on the vehicle crop. Skipped when the crop is smaller than `detection.min_vehicle_crop_px` on either side, or when the track was attempted more recently than `ocr.retry_interval_frames` ago (`track_state.should_attempt`) — avoids burning detection+OCR on every single frame of every unconfirmed track. Among returned plate boxes, `pipeline.py` picks `argmax(boxes.conf)`, not just index 0.
3. **Deskew** — `deskew.py` finds the plate's 4 corners via contour approximation and applies a perspective transform; falls back to the raw crop if 4 corners aren't found (common on low-contrast crops).
4. **OCR + format** — `ocr.py`'s `vn_plate_parser` runs PaddleOCR (2.x API — see pinning note below), drops text lines below `ocr.confidence_threshold`, sorts the rest top-to-bottom, concatenates, strips non-alphanumerics, and hands off to `formatter.py`'s `vn_plate_formater`, which validates against a regex per known VN plate layout (`^\d{2}[A-Z]\d{5}$` for 8 chars, `^\d{2}[A-Z]\d{6}$` for 9) and formats to `XXA-YYY.YY` style, rejecting to `""` on any other length or mismatch. Returns `(text, confidence)`, confidence being the mean of the kept lines' OCR scores.
5. **Frame-voting + cooldown** — `track_state.py` buffers each track's last `ocr.vote_buffer_size` validated reads and locks one in once it gets `ocr.vote_min_count` votes (majority within the window, not first-match). Once locked, OCR is skipped for that track for the rest of the video.
6. **Persistence** — on lock-in, `pipeline.py` saves the exact post-deskew crop to `output/snapshots/frame{N}_track{id}_{plate_text}.jpg` and writes a row (video source, track id, vehicle class, plate text, OCR confidence, snapshot path, frame number, timestamp) via `storage.py`'s `Storage` class to SQLite at `output/detections.db` (schema in `implementation-plan.html` Phase 4; `plate_color` column exists but is always `NULL` until color classification lands). Results also go to the console and a timestamped file under `logs/` (`pipeline._setup_logging`), plus the annotated output video.

Separately, debug snapshots are gated by `config.yaml`'s `debug.enabled` (on by default): when on, `pipeline.py` writes the raw frame, vehicle crop, and plate crop to `output/debug/{frames,vehicles,plates}/frame{N}_track{id}.jpg` on every detection *attempt* (not just confirmed reads) — useful for eyeballing near-miss reads, distinct from the confirmed-only `output/snapshots/`.

### Known correctness gaps (see implementation-plan.html Phase 3 for detail)

- No plate-color (white/yellow/blue) or layout (1-line/2-line) classification yet, so there's no color-aware preprocessing and OCR line-sorting relies purely on however many text regions PaddleOCR returns.

### Dependency pinning

`paddleocr` and `paddlepaddle` are pinned `<3` deliberately — PaddleOCR 3.x renamed constructor args (`use_angle_cls` → `use_textline_orientation`, dropped `show_log`) and its default pipeline is heavier than the 2.x path `ocr.py` is built against. Don't upgrade these without updating `ocr.py` accordingly.

### Training

`train_model.py` fine-tunes YOLO26 (from `yolo26n.pt`) on `input/datasets/vn_license_plate_dataset/data.yaml` (Roboflow export, single class `plate`) to produce `models/license_plate_detector.pt`. Current run config: 50 epochs, imgsz=416, batch=8, CPU.
