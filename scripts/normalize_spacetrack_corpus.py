#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import pi
from pathlib import Path
from typing import Iterable

from sgp4.api import Satrec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize archive, recent-tail, and current GP data into per-object Space-Track-style JSON files."
    )
    parser.add_argument(
        "--archive-dir",
        default="downloads/spacetrack_archives",
        help="Directory containing the downloaded historical TLE zip archives.",
    )
    parser.add_argument(
        "--tail-dir",
        default=None,
        help="Optional directory containing recent-tail GP_History JSON windows.",
    )
    parser.add_argument(
        "--current-gp-path",
        default=None,
        help="Optional current GP snapshot JSON file to merge in.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/spacetrack_unified",
        help="Output directory for normalized per-NORAD JSON arrays.",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Optional scratch directory for intermediate JSONL shards. Defaults to <output-dir>/.scratch.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output and scratch directories before rebuilding.",
    )
    return parser.parse_args()


def _tle_epoch_to_datetime(epoch_year: int, epoch_days: float) -> datetime:
    year = 1900 + epoch_year if epoch_year >= 57 else 2000 + epoch_year
    day_int = int(epoch_days)
    frac = epoch_days - day_int
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int - 1, seconds=frac * 86400.0)


def _satrec_to_gp_row(sat: Satrec, line1: str, line2: str, source_file: str) -> dict:
    epoch = _tle_epoch_to_datetime(int(sat.epochyr), float(sat.epochdays))
    return {
        "ARG_OF_PERICENTER": float(sat.argpo * 180.0 / pi),
        "BSTAR": float(sat.bstar),
        "COMMENT": "NORMALIZED_FROM_ARCHIVE_TLE",
        "ECCENTRICITY": float(sat.ecco),
        "EPOCH": epoch.isoformat().replace("+00:00", "Z"),
        "INCLINATION": float(sat.inclo * 180.0 / pi),
        "MEAN_ANOMALY": float(sat.mo * 180.0 / pi),
        "MEAN_MOTION": float(sat.no_kozai * 1440.0 / (2.0 * pi)),
        "MEAN_MOTION_DDOT": float(sat.nddot),
        "MEAN_MOTION_DOT": float(sat.ndot),
        "NORAD_CAT_ID": int(sat.satnum),
        "OBJECT_NAME": None,
        "RA_OF_ASC_NODE": float(sat.nodeo * 180.0 / pi),
        "SOURCE_FILE": source_file,
        "TLE_LINE1": line1.rstrip(),
        "TLE_LINE2": line2.rstrip(),
    }


def _iter_archive_rows(archive_dir: Path) -> Iterable[dict]:
    for zip_path in sorted(archive_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            inner_name = zf.namelist()[0]
            with zf.open(inner_name) as handle:
                while True:
                    line1_bytes = handle.readline()
                    if not line1_bytes:
                        break
                    line2_bytes = handle.readline()
                    if not line2_bytes:
                        break
                    line1 = line1_bytes.decode("utf-8", "ignore").rstrip("\r\n\\")
                    line2 = line2_bytes.decode("utf-8", "ignore").rstrip("\r\n\\")
                    if not line1.startswith("1 ") or not line2.startswith("2 "):
                        continue
                    sat = Satrec.twoline2rv(line1, line2)
                    yield _satrec_to_gp_row(sat, line1, line2, zip_path.name)


def _iter_json_rows(path: Path) -> Iterable[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row


def _append_row(target: Path, row: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _iter_all_rows(archive_dir: Path, tail_dir: Path | None, current_gp_path: Path | None) -> Iterable[dict]:
    yield from _iter_archive_rows(archive_dir)

    if tail_dir and tail_dir.exists():
        for path in sorted(tail_dir.glob("*.json")):
            yield from _iter_json_rows(path)

    if current_gp_path and current_gp_path.exists():
        yield from _iter_json_rows(current_gp_path)


def _row_sort_key(row: dict) -> tuple[str, str]:
    return (
        str(row.get("EPOCH") or ""),
        str(row.get("CREATION_DATE") or ""),
    )


def main() -> None:
    args = parse_args()
    from rso_world_model.data.io import ensure_dir, write_json

    archive_dir = Path(args.archive_dir)
    tail_dir = Path(args.tail_dir) if args.tail_dir else None
    current_gp_path = Path(args.current_gp_path) if args.current_gp_path else None
    output_dir = Path(args.output_dir)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else output_dir / ".scratch"

    if args.clean_output:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    ensure_dir(output_dir)
    ensure_dir(scratch_dir)

    counts = defaultdict(int)
    for row in _iter_all_rows(archive_dir, tail_dir, current_gp_path):
        norad_cat_id = row.get("NORAD_CAT_ID")
        if norad_cat_id in (None, ""):
            continue
        norad_cat_id = int(norad_cat_id)
        shard_path = scratch_dir / f"{norad_cat_id}.jsonl"
        _append_row(shard_path, row)
        counts[norad_cat_id] += 1

    manifest = []
    for shard_path in sorted(scratch_dir.glob("*.jsonl")):
        norad_cat_id = int(shard_path.stem)
        rows = [json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        deduped: dict[str, dict] = {}
        for row in rows:
            key = str(row.get("EPOCH") or "")
            existing = deduped.get(key)
            if existing is None or _row_sort_key(row) >= _row_sort_key(existing):
                deduped[key] = row

        normalized_rows = sorted(deduped.values(), key=_row_sort_key)
        target = output_dir / f"{norad_cat_id}.json"
        write_json(target, normalized_rows)
        manifest.append(
            {
                "norad_cat_id": norad_cat_id,
                "path": str(target.resolve()),
                "rows": len(normalized_rows),
            }
        )
        print(json.dumps({"norad_cat_id": norad_cat_id, "rows": len(normalized_rows), "path": str(target)}, sort_keys=True))

    write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
