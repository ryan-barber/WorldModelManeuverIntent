#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


STARLINK_IDS = {44714, 44718, 44723, 44724, 44725, 44736, 44741, 44744, 44747, 44748}
GEO_IDS = {19548, 20253, 20776, 21639, 22314, 22787, 22988, 23467, 23613, 23712}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prepared world-model windows and heuristic labels.")
    parser.add_argument(
        "--manifest",
        default="data/processed/train_windows/manifest.json",
        help="Path to prepared dataset manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    by_object: dict[int, dict[str, int]] = {}
    by_group = {
        "starlink": {"total": 0, "positive": 0},
        "geo": {"total": 0, "positive": 0},
        "other": {"total": 0, "positive": 0},
    }
    class_counts: dict[int, int] = {}
    purpose_counts: dict[int, int] = {}
    bucket_counts: dict[int, int] = {}
    next_times: list[float] = []
    positive_windows = 0

    for item in items:
        window = np.load(item["path"])
        norad_cat_id = int(item["norad_cat_id"])
        positive = int(float(window["maneuver_probability"]) > 0.5)
        positive_windows += positive

        object_stats = by_object.setdefault(norad_cat_id, {"total": 0, "positive": 0})
        object_stats["total"] += 1
        object_stats["positive"] += positive

        group = "starlink" if norad_cat_id in STARLINK_IDS else "geo" if norad_cat_id in GEO_IDS else "other"
        by_group[group]["total"] += 1
        by_group[group]["positive"] += positive

        if float(window["next_maneuver_time_mask"]) > 0.5:
            next_times.append(float(window["next_maneuver_time"]))
        if float(window["maneuver_class_mask"]) > 0.5:
            label = int(window["maneuver_class"])
            class_counts[label] = class_counts.get(label, 0) + 1
        if float(window["maneuver_purpose_mask"]) > 0.5:
            label = int(window["maneuver_purpose"])
            purpose_counts[label] = purpose_counts.get(label, 0) + 1
        if float(window["delta_v_bucket_mask"]) > 0.5:
            label = int(window["delta_v_bucket"])
            bucket_counts[label] = bucket_counts.get(label, 0) + 1

    summary = {
        "window_count": len(items),
        "object_count": len(by_object),
        "positive_window_count": positive_windows,
        "positive_window_ratio": positive_windows / max(len(items), 1),
        "group_positive_ratio": {
            key: value["positive"] / max(value["total"], 1) for key, value in by_group.items()
        },
        "class_counts": dict(sorted(class_counts.items())),
        "purpose_counts": dict(sorted(purpose_counts.items())),
        "delta_v_bucket_counts": dict(sorted(bucket_counts.items())),
        "next_maneuver_time_seconds": {
            "count": len(next_times),
            "p10": float(np.percentile(next_times, 10)) if next_times else None,
            "p50": float(np.percentile(next_times, 50)) if next_times else None,
            "p90": float(np.percentile(next_times, 90)) if next_times else None,
        },
        "top_objects_by_positive_windows": [
            {
                "norad_cat_id": norad_cat_id,
                "positive_windows": stats["positive"],
                "total_windows": stats["total"],
                "positive_ratio": stats["positive"] / max(stats["total"], 1),
            }
            for norad_cat_id, stats in sorted(
                by_object.items(),
                key=lambda item: (-item[1]["positive"], -item[1]["total"]),
            )[:15]
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

