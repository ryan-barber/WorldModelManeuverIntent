#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CelesTrak GP JSON for one or more groups.")
    parser.add_argument("--groups", nargs="+", required=True, help="CelesTrak groups, e.g. active starlink geo.")
    parser.add_argument("--output-dir", default="data/raw/celestrak", help="Directory for cached group JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rso_world_model.data.celestrak import download_group_gp

    paths = download_group_gp(args.groups, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
