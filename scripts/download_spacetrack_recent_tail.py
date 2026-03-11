#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download narrow GP_History tail windows for recent Space-Track coverage."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Inclusive end date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=1,
        help="Epoch-date window size per request. Defaults to 1 day.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/spacetrack_tail",
        help="Directory for cached recent-tail JSON windows.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already-downloaded window files.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per window/query mode before failing over. Defaults to 5.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=10.0,
        help="Base retry sleep in seconds. Defaults to 10.",
    )
    parser.add_argument(
        "--failed-log-path",
        default="data/raw/spacetrack_tail/failed_windows.jsonl",
        help="JSONL file for permanently failed windows.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failed windows and continue instead of exiting.",
    )
    return parser.parse_args()


def _iter_windows(start: date, end: date, window_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start
    delta = timedelta(days=max(window_days, 1) - 1)
    while cursor <= end:
        window_end = min(cursor + delta, end)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _window_name(start: date, end: date) -> str:
    return f"gp_history_epoch_{start.isoformat()}_{end.isoformat()}.json"


def _sleep_with_backoff(base_sleep_s: float, attempt: int) -> None:
    time.sleep(base_sleep_s * max(attempt, 1))


def _fetch_rows_with_retry(
    client,
    fetch_fn,
    window_start: date,
    window_end: date,
    max_retries: int,
    retry_sleep_s: float,
    mode: str,
):
    query_end = window_end + timedelta(days=1)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            rows = fetch_fn(window_start, query_end)
            if rows and isinstance(rows[0], dict) and "error" in rows[0]:
                raise RuntimeError(rows[0]["error"])
            return rows
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            print(
                json.dumps(
                    {
                        "event": "retry",
                        "mode": mode,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "start_date": window_start.isoformat(),
                        "end_date": window_end.isoformat(),
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if attempt < max_retries:
                _sleep_with_backoff(retry_sleep_s, attempt)
    if last_error is None:
        raise RuntimeError(f"{mode} fetch failed without a captured exception")
    raise last_error


def _append_failed_window(
    failed_log_path: Path,
    window_start: date,
    window_end: date,
    error: Exception,
) -> None:
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "start_date": window_start.isoformat(),
                    "end_date": window_end.isoformat(),
                    "error": str(error),
                    "logged_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
                sort_keys=True,
            )
            + "\n"
        )


def main() -> None:
    args = parse_args()
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if end_date < start_date:
        raise SystemExit("--end-date must be on or after --start-date.")

    from rso_world_model.data.io import ensure_dir, write_json
    from rso_world_model.data.spacetrack import SpaceTrackClient, SpaceTrackCredentials

    output_dir = ensure_dir(args.output_dir)
    failed_log_path = Path(args.failed_log_path)
    client = SpaceTrackClient(SpaceTrackCredentials.from_env())
    client.login()

    for window_start, window_end in _iter_windows(start_date, end_date, args.window_days):
        target = output_dir / _window_name(window_start, window_end)
        if args.skip_existing and target.exists():
            print(f"SKIP {target}")
            continue

        try:
            try:
                rows = _fetch_rows_with_retry(
                    client=client,
                    fetch_fn=client.fetch_gp_history_creation_range,
                    window_start=window_start,
                    window_end=window_end,
                    max_retries=args.max_retries,
                    retry_sleep_s=args.retry_sleep_seconds,
                    mode="creation_date",
                )
            except Exception as creation_error:
                print(
                    json.dumps(
                        {
                            "event": "fallback",
                            "from_mode": "creation_date",
                            "to_mode": "epoch",
                            "start_date": window_start.isoformat(),
                            "end_date": window_end.isoformat(),
                            "error": str(creation_error),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                rows = _fetch_rows_with_retry(
                    client=client,
                    fetch_fn=client.fetch_gp_history_epoch_range,
                    window_start=window_start,
                    window_end=window_end,
                    max_retries=args.max_retries,
                    retry_sleep_s=args.retry_sleep_seconds,
                    mode="epoch",
                )
        except Exception as error:
            if not args.continue_on_error:
                raise
            _append_failed_window(failed_log_path, window_start, window_end, error)
            print(
                json.dumps(
                    {
                        "event": "failed_window",
                        "failed_log_path": str(failed_log_path),
                        "start_date": window_start.isoformat(),
                        "end_date": window_end.isoformat(),
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        write_json(target, rows)
        summary = {
            "path": str(target),
            "rows": len(rows),
            "start_date": window_start.isoformat(),
            "end_date": window_end.isoformat(),
        }
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
