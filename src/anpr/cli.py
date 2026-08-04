import argparse
from pathlib import Path

import yaml

from anpr import pipeline

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Vietnamese ANPR pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output",
        default="output/output_tracked.mp4",
        help="Path for the annotated output video",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    pipeline.run(args.video, args.output, config)


if __name__ == "__main__":
    main()
