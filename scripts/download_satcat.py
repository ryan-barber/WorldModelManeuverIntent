#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the CelesTrak SATCAT CSV.")
    parser.add_argument("--output-path", default="data/raw/satcat/satcat.csv", help="Destination CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rso_world_model.data.satcat import download_satcat_csv

    print(download_satcat_csv(args.output_path))


if __name__ == "__main__":
    main()
