#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk, resumable current and historical Space-Track downloads.")
    parser.add_argument(
        "--mode",
        choices=["current", "history", "both"],
        default="both",
        help="Which Space-Track dataset types to download.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw/spacetrack_bulk",
        help="Root directory for current and historical Space-Track cache.",
    )
    parser.add_argument(
        "--satcat-path",
        default="data/raw/satcat/satcat.csv",
        help="SATCAT CSV used to source NORAD IDs when no id file is provided.",
    )
    parser.add_argument(
        "--current-output-name",
        default="gp_current_all.json",
        help="Filename for the all-current GP snapshot inside the current/ directory.",
    )
    parser.add_argument(
        "--id-source",
        choices=["satcat", "current-gp", "id-file"],
        default="current-gp",
        help="Source of NORAD IDs for historical downloads.",
    )
    parser.add_argument("--id-file", default=None, help="Optional newline-separated NORAD ID file.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset into the resolved NORAD ID list.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process the resolved NORAD ID list in descending order.",
    )
    parser.add_argument(
        "--max-ids",
        type=int,
        default=2000,
        help="Maximum NORAD IDs to process this run for history downloads. Use <=0 for no cap.",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="NORAD IDs per GP_History request.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Delay between GP_History batch requests.")
    parser.add_argument(
        "--rate-limit-sleep-seconds",
        type=float,
        default=60.0,
        help="Sleep interval when Space-Track returns an explicit rate-limit response.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries after a rate-limit response before giving up on the batch.",
    )
    parser.add_argument(
        "--hard-rate-limit-sleep-seconds",
        type=float,
        default=300.0,
        help="Long backoff after repeated rate-limit failures before retrying the same batch.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log non-rate-limit batch failures and continue with later batches instead of exiting.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip historical object files that already exist in the cache.",
    )
    return parser.parse_args()


def _load_ids_from_satcat(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        ids = {
            int(row["NORAD_CAT_ID"])
            for row in reader
            if row.get("NORAD_CAT_ID") and row["NORAD_CAT_ID"].isdigit()
        }
    return sorted(ids)


def _load_ids_from_current_gp(path: Path) -> list[int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return sorted({int(row["NORAD_CAT_ID"]) for row in rows if row.get("NORAD_CAT_ID") is not None})


def _load_ids_from_file(path: Path) -> list[int]:
    return sorted({int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})


def _resolve_history_ids(args: argparse.Namespace, current_gp_path: Path) -> list[int]:
    if args.id_source == "id-file":
        if not args.id_file:
            raise RuntimeError("--id-file is required when --id-source id-file is used.")
        return _load_ids_from_file(Path(args.id_file))
    if args.id_source == "satcat":
        return _load_ids_from_satcat(Path(args.satcat_path))
    if not current_gp_path.exists():
        raise RuntimeError("Current GP snapshot is required for --id-source current-gp.")
    return _load_ids_from_current_gp(current_gp_path)


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_error_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


class SpaceTrackRateLimitError(RuntimeError):
    pass


def _normalize_history_rows(payload: list[dict]) -> list[dict]:
    if payload and isinstance(payload[0], dict) and "error" in payload[0]:
        message = str(payload[0]["error"])
        if "rate limit" in message.lower():
            raise SpaceTrackRateLimitError(message)
        raise RuntimeError(message)
    return [row for row in payload if isinstance(row, dict) and "NORAD_CAT_ID" in row]


def _fetch_history_rows_with_fallback(
    client,
    batch: list[int],
    rate_limit_sleep_seconds: float,
    max_retries: int,
) -> list[dict]:
    try:
        retries_remaining = max_retries
        while True:
            try:
                return _normalize_history_rows(client.fetch_gp_history_batch(batch))
            except SpaceTrackRateLimitError:
                if retries_remaining <= 0:
                    raise
                time.sleep(rate_limit_sleep_seconds)
                retries_remaining -= 1
    except Exception:
        if len(batch) == 1:
            retries_remaining = max_retries
            while True:
                try:
                    return _normalize_history_rows(client.fetch_gp_history(batch[0]))
                except SpaceTrackRateLimitError:
                    if retries_remaining <= 0:
                        raise
                    time.sleep(rate_limit_sleep_seconds)
                    retries_remaining -= 1
        midpoint = len(batch) // 2
        return _fetch_history_rows_with_fallback(
            client,
            batch[:midpoint],
            rate_limit_sleep_seconds=rate_limit_sleep_seconds,
            max_retries=max_retries,
        ) + _fetch_history_rows_with_fallback(
            client,
            batch[midpoint:],
            rate_limit_sleep_seconds=rate_limit_sleep_seconds,
            max_retries=max_retries,
        )


def main() -> None:
    args = parse_args()
    from rso_world_model.data.io import ensure_dir, write_json
    from rso_world_model.data.spacetrack import SpaceTrackClient, SpaceTrackCredentials

    output_root = ensure_dir(args.output_root)
    current_dir = ensure_dir(output_root / "current")
    history_dir = ensure_dir(output_root / "history")
    current_gp_path = current_dir / args.current_output_name
    progress_path = output_root / "progress.json"
    error_log_path = output_root / "errors.jsonl"

    client = SpaceTrackClient(SpaceTrackCredentials.from_env())

    if args.mode in {"current", "both"}:
        path = client.download_current_gp_cache(current_gp_path)
        print(path)

    if args.mode not in {"history", "both"}:
        return

    if args.mode == "history":
        client.login()

    all_ids = _resolve_history_ids(args, current_gp_path)
    if args.reverse:
        all_ids = list(reversed(all_ids))
    candidate_ids = all_ids[args.start_index :]
    if args.skip_existing:
        candidate_ids = [norad_cat_id for norad_cat_id in candidate_ids if not (history_dir / f"{norad_cat_id}.json").exists()]
    if args.max_ids > 0:
        candidate_ids = candidate_ids[: args.max_ids]

    written = 0
    batches = _chunked(candidate_ids, args.batch_size)
    batch_index = 0
    while batch_index < len(batches):
        batch = batches[batch_index]
        try:
            rows = _fetch_history_rows_with_fallback(
                client,
                batch,
                rate_limit_sleep_seconds=args.rate_limit_sleep_seconds,
                max_retries=args.max_retries,
            )
        except SpaceTrackRateLimitError as exc:
            _write_progress(
                progress_path,
                {
                    "mode": args.mode,
                    "id_source": args.id_source,
                    "start_index": args.start_index,
                    "reverse": args.reverse,
                    "max_ids": args.max_ids,
                    "batch_size": args.batch_size,
                    "sleep_seconds": args.sleep_seconds,
                    "written_files": written,
                    "processed_ids": written,
                    "last_batch_ids": batch,
                    "remaining_ids": max(len(candidate_ids) - written, 0),
                    "last_error": str(exc),
                    "status": "rate_limited",
                    "timestamp_unix": time.time(),
                },
            )
            time.sleep(args.hard_rate_limit_sleep_seconds)
            continue
        except Exception as exc:
            _append_error_log(
                error_log_path,
                {
                    "batch_ids": batch,
                    "error": str(exc),
                    "timestamp_unix": time.time(),
                },
            )
            if not args.continue_on_error:
                raise
            batch_index += 1
            continue

        grouped: dict[int, list[dict]] = {int(norad_cat_id): [] for norad_cat_id in batch}
        for row in rows:
            grouped.setdefault(int(row["NORAD_CAT_ID"]), []).append(row)
        for norad_cat_id, payload in grouped.items():
            target = history_dir / f"{norad_cat_id}.json"
            write_json(target, payload)
            print(target)
            written += 1

        batch_index += 1
        _write_progress(
            progress_path,
            {
                "mode": args.mode,
                "id_source": args.id_source,
                "start_index": args.start_index,
                "reverse": args.reverse,
                "max_ids": args.max_ids,
                "batch_size": args.batch_size,
                "sleep_seconds": args.sleep_seconds,
                "written_files": written,
                "processed_ids": written,
                "last_batch_ids": batch,
                "remaining_ids": max(len(candidate_ids) - written, 0),
                "status": "running",
                "timestamp_unix": time.time(),
            },
        )
        if args.sleep_seconds > 0 and batch_index < len(batches):
            time.sleep(args.sleep_seconds)
        batch_index += 1


if __name__ == "__main__":
    main()
