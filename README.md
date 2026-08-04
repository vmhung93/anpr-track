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

Model paths, target classes, and confidence thresholds live in `config.yaml`. For every plate the pipeline confirms, it saves the exact (deskewed) crop that was OCR'd to `output/snapshots/frame{N}_track{id}_{plate_text}.jpg`, so you can manually check the image against the recognized text.

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
  track_state.py                   # per-vehicle plate cooldown
  storage.py                        # SQLite/CSV logger (Phase 4)
tests/
logs/
output/
  snapshots/               # plate crops saved for manual verification
.vscode/
  launch.json              # debugpy config for `anpr.cli`
train_model.py         # trains the custom plate-detector model
```

## Training the plate detector

```bash
uv run python train_model.py
```

Trains on `input/datasets/vn_license_plate_dataset/data.yaml`; see that dataset's `README.roboflow.txt` for source/license details.
