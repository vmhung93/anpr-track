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

1. **Detect + track vehicles** — `vehicle_detector.py` wraps `YOLO.track(..., persist=True)` (ByteTrack), filtered to COCO classes 2/3/5/7 (car/motorcycle/bus/truck) from `config.yaml`.
2. **Crop + detect plate** — for each tracked vehicle without a confirmed plate yet, `plate_detector.py` runs a second, custom-trained YOLO26 model (`train_model.py` output) on the vehicle crop.
3. **Deskew** — `deskew.py` finds the plate's 4 corners via contour approximation and applies a perspective transform; falls back to the raw crop if 4 corners aren't found (common on low-contrast crops).
4. **OCR + format** — `ocr.py`'s `vn_plate_parser` runs PaddleOCR (2.x API — see pinning note below), sorts detected text lines top-to-bottom, concatenates, strips non-alphanumerics, and hands off to `formatter.py`'s `vn_plate_formater`, which reformats by string length (8 or 9 chars) into `XXA-YYY.YY` style.
5. **Cooldown** — `track_state.py` is a plain dict keyed by `track_id`: once a track has a plate, OCR is skipped for it on subsequent frames (first successful read wins, no voting yet).
6. **Persistence** — `storage.py` is currently an empty stub (Phase 4: SQLite at `output/detections.db`). Right now results only go to stdout, the annotated output video, and per-plate snapshot crops.

Every confirmed plate read saves the exact post-deskew crop to `output/snapshots/frame{N}_track{id}_{plate_text}.jpg`, so OCR output can be eyeballed against the source image — use these when debugging recognition quality instead of guessing from text output alone.

### Known correctness gaps (see implementation-plan.html Phase 3/5 for detail)

- `pipeline.py` takes `plate_results[0].boxes.xyxy[0]` as "the" plate box — Ultralytics does not guarantee confidence ordering, so this should select `argmax(boxes.conf)` when multiple plate candidates exist.
- `vn_plate_formater` accepts any 8- or 9-character string with no character-class check, so OCR misreads of that length are silently formatted as valid plates. Intended fix is a regex per known VN plate format, rejecting to "no read" on mismatch.
- No plate-color (white/yellow/blue) or layout (1-line/2-line) classification yet, so there's no color-aware preprocessing and OCR line-sorting relies purely on however many text regions PaddleOCR returns.
- `track_state.py` keeps the *first* non-empty OCR read per track, not a majority vote — Phase 5 plans a short per-track buffer with frame-voting before locking in a plate.

### Dependency pinning

`paddleocr` and `paddlepaddle` are pinned `<3` deliberately — PaddleOCR 3.x renamed constructor args (`use_angle_cls` → `use_textline_orientation`, dropped `show_log`) and its default pipeline is heavier than the 2.x path `ocr.py` is built against. Don't upgrade these without updating `ocr.py` accordingly.

### Training

`train_model.py` fine-tunes YOLO26 (from `yolo26n.pt`) on `data/datasets/vn_license_plate_dataset/data.yaml` (Roboflow export, single class `plate`) to produce `models/license_plate_detector.pt`. Current run config: 50 epochs, imgsz=416, batch=8, CPU.
