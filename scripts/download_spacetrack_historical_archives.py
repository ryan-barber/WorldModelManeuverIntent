#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Space-Track historical archive assets into the project directory."
    )
    parser.add_argument("--urls-file", help="Path to a newline-separated file of archive URLs to download.")
    parser.add_argument(
        "--manifest-json",
        help="Path to a JSON manifest containing objects with download_url and optional file_name.",
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
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Download archive URLs directly without first logging into Space-Track.",
    )
    return parser.parse_args()


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    name = Path(path).name
    return unquote(name) or "archive.bin"


def _load_entries(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    if bool(args.urls_file) == bool(args.manifest_json):
        raise SystemExit("Provide exactly one of --urls-file or --manifest-json.")

    if args.urls_file:
        urls = [
            line.strip()
            for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return [(url, None) for url in urls]

    payload = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("Manifest JSON must be a list of objects.")

    entries: list[tuple[str, str | None]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        url = item.get("download_url")
        if not url:
            continue
        file_name = item.get("file_name") or item.get("suggested_filename")
        entries.append((str(url), str(file_name) if file_name else None))
    return entries


def main() -> None:
    from rso_world_model.data.io import ensure_dir
    from rso_world_model.data.spacetrack import SpaceTrackClient, SpaceTrackCredentials

    args = parse_args()
    entries = _load_entries(args)
    output_dir = ensure_dir(args.output_dir)

    if args.no_auth:
        session = requests.Session()
    else:
        client = SpaceTrackClient(SpaceTrackCredentials.from_env(), timeout_s=120.0)
        client.login()
        session = client.session

    for url, explicit_name in entries:
        target = output_dir / (explicit_name or _filename_from_url(url))
        if args.skip_existing and target.exists():
            print(f"SKIP {target}")
            continue

        response = session.get(url, timeout=120.0, stream=True)
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        print(target)


if __name__ == "__main__":
    main()
