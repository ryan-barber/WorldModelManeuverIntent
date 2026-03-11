#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Space-Track historical archive assets into the project directory."
    )
    parser.add_argument(
        "--urls-file",
        required=True,
        help="Path to a newline-separated file of authenticated Space-Track archive URLs to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="downloads/spacetrack_archives",
        help="Project-local directory for downloaded archive files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally.",
    )
    return parser.parse_args()


def _filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1] or "archive.bin"


def main() -> None:
    from rso_world_model.data.io import ensure_dir
    from rso_world_model.data.spacetrack import SpaceTrackClient, SpaceTrackCredentials

    args = parse_args()
    urls = [
        line.strip()
        for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    output_dir = ensure_dir(args.output_dir)

    client = SpaceTrackClient(SpaceTrackCredentials.from_env(), timeout_s=120.0)
    client.login()

    for url in urls:
        target = output_dir / _filename_from_url(url)
        if args.skip_existing and target.exists():
            print(f"SKIP {target}")
            continue

        response = client.session.get(url, timeout=120.0, stream=True)
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        print(target)


if __name__ == "__main__":
    main()

