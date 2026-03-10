#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache Space-Track GP history by NORAD ID.")
    parser.add_argument("--norad-ids", nargs="+", required=True, type=int, help="NORAD catalog IDs to download.")
    parser.add_argument("--output-dir", default="data/raw/spacetrack", help="Directory for cached JSON files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rso_world_model.data.spacetrack import SpaceTrackClient, SpaceTrackCredentials

    client = SpaceTrackClient(SpaceTrackCredentials.from_env())
    paths = client.download_gp_history_cache(args.norad_ids, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
