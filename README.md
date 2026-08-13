# anpr-track

Vietnamese license plate detection and tracking. Detects vehicles in a video with YOLO26 + ByteTrack, locates the plate on each vehicle, and reads it with PaddleOCR — covering white (personal), yellow (commercial), and blue (government) plates.

See [`implementation-plan.html`](implementation-plan.html) for the full architecture, phased build-out, and known open issues.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- `libgomp.so.1` on the host system (OpenMP runtime required by `paddlepaddle`) — install via `sudo apt-get install -y libgomp1` if `paddlepaddle` fails to import

## Setup

```bash
uv sync
```

Place model weights in `models/`:

- `yolo26n.pt` — pre-trained YOLO26, filtered to COCO classes 2/3/5/7 (car, motorcycle, bus, truck)
- `license_plate_detector.pt` — custom-trained YOLO26 plate detector (see `train_model.py`)

## Usage

```bash
uv run anpr --video path/to/clip.mp4 --config config.yaml --output output/output_tracked.mp4
```

Model paths, detection/OCR thresholds, and output locations live in `config.yaml`. For every plate the pipeline confirms (after frame-voting — see below), it saves the exact deskewed crop to `output/snapshots/frame{N}_track{id}_{plate_text}.jpg` and logs a row (video source, track id, vehicle class, plate text, OCR confidence, snapshot path, frame number, timestamp) to a SQLite table at `output/detections.db`. Console output is also mirrored to a timestamped file under `logs/`.

When `debug.enabled` is on (the default), the pipeline separately writes the raw frame, vehicle crop, and plate crop to `output/debug/{frames,vehicles,plates}/frame{N}_track{id}.jpg` on *every* detection attempt (not just confirmed reads) — useful for inspecting near-misses, distinct from the confirmed-only `output/snapshots/`.

A plate isn't locked in on the first OCR read: `track_state.py` keeps the last few validated reads per vehicle and only confirms one once it wins a majority vote (`ocr.vote_min_count` out of `ocr.vote_buffer_size`), to avoid one lucky-but-wrong frame locking in a bad plate.

## Project layout

```
config.yaml           # model paths, detection thresholds, output locations
models/                # YOLO26 vehicle + plate detector weights
input/
  videos/               # input clips
  datasets/              # training data (YOLO-format)
src/anpr/
  cli.py                  # argparse entry point (`uv run anpr`)
  pipeline.py              # per-frame detect -> track -> OCR loop
  vehicle_detector.py       # YOLO26 + ByteTrack wrapper
  plate_detector.py          # plate detection on vehicle crops
  deskew.py                   # four-point perspective correction
  ocr.py                         # PaddleOCR wrapper
  formatter.py                    # raw OCR text -> standard VN plate format
  track_state.py                   # per-vehicle frame-voting buffer + cooldown
  snapshot.py                       # writes a crop to disk
  storage.py                         # SQLite logger for confirmed reads
tests/
logs/                    # per-run timestamped log files
output/
  snapshots/               # deskewed crop per confirmed plate read
  debug/                   # per-frame/vehicle/plate crops when debug.enabled is true
.vscode/
  launch.json              # debugpy config for `anpr.cli`
train_model.py         # trains the custom plate-detector model
```

## Training the plate detector

```bash
uv run python train_model.py
```

Trains on `input/datasets/vn_license_plate_dataset/data.yaml`; see that dataset's `README.roboflow.txt` for source/license details.
