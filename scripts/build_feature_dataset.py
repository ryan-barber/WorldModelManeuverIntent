#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw orbital history into prepared TCN training windows.")
    parser.add_argument("--spacetrack-dir", default="data/raw/spacetrack", help="Directory with Space-Track JSON.")
    parser.add_argument("--celestrak-dir", default="data/raw/celestrak", help="Directory with CelesTrak JSON.")
    parser.add_argument("--satcat-path", default="data/raw/satcat/satcat.csv", help="SATCAT CSV path.")
    parser.add_argument("--output-dir", default="data/processed/train_windows", help="Output directory for .npz windows.")
    parser.add_argument("--window-size", type=int, default=96, help="Sequence length per training sample.")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride.")
    parser.add_argument("--prediction-horizon-steps", type=int, default=16, help="Lookahead steps for labels.")
    parser.add_argument("--ephemeris-path", default=None, help="Optional local Skyfield ephemeris file.")
    parser.add_argument(
        "--position-residual-threshold-km",
        type=float,
        default=25.0,
        help="Residual threshold in kilometers for flagging a maneuver event.",
    )
    parser.add_argument(
        "--velocity-residual-threshold-m-s",
        type=float,
        default=2.5,
        help="Velocity residual threshold in m/s for flagging a maneuver event.",
    )
    return parser.parse_args()


def _load_sequence_sources(spacetrack_dir: Path, celestrak_dir: Path) -> dict[int, tuple[str, list[dict]]]:
    from rso_world_model.data.io import read_json

    merged: dict[int, tuple[str, list[dict]]] = {}

    for path in sorted(spacetrack_dir.glob("*.json")):
        payload = read_json(path)
        if not payload:
            continue
        norad_cat_id = int(payload[0]["NORAD_CAT_ID"])
        merged[norad_cat_id] = ("spacetrack", payload)

    for path in sorted(celestrak_dir.glob("*.json")):
        payload = read_json(path)
        grouped: dict[int, list[dict]] = {}
        for row in payload:
            norad_cat_id = int(row["NORAD_CAT_ID"])
            grouped.setdefault(norad_cat_id, []).append(row)
        for norad_cat_id, rows in grouped.items():
            merged.setdefault(norad_cat_id, ("celestrak", rows))
    return merged


def _save_window(path: Path, sample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sequence_features=sample.sequence_features,
        feature_mask=sample.feature_mask,
        static_features=sample.static_features,
        static_feature_mask=sample.static_feature_mask,
        maneuver_probability=sample.targets["maneuver_probability"],
        maneuver_probability_mask=sample.target_mask["maneuver_probability"],
        next_maneuver_time=sample.targets["next_maneuver_time"],
        next_maneuver_time_mask=sample.target_mask["next_maneuver_time"],
        maneuver_class=sample.targets["maneuver_class"],
        maneuver_class_mask=sample.target_mask["maneuver_class"],
        maneuver_purpose=sample.targets["maneuver_purpose"],
        maneuver_purpose_mask=sample.target_mask["maneuver_purpose"],
        delta_v_bucket=sample.targets["delta_v_bucket"],
        delta_v_bucket_mask=sample.target_mask["delta_v_bucket"],
        remaining_delta_v_estimate=sample.targets["remaining_delta_v_estimate"],
        remaining_delta_v_estimate_mask=sample.target_mask["remaining_delta_v_estimate"],
        residual_growth=sample.targets["residual_growth"],
        residual_growth_mask=sample.target_mask["residual_growth"],
    )


def main() -> None:
    args = parse_args()
    from rso_world_model.data.satcat import load_satcat_metadata
    from rso_world_model.data.schemas import PreparedSequence
    from rso_world_model.features.builder import FeatureBuilderConfig, WorldModelFeatureBuilder
    from rso_world_model.features.maneuvers import ManeuverDetectionConfig
    from rso_world_model.training.windowing import make_window_samples

    spacetrack_dir = Path(args.spacetrack_dir)
    celestrak_dir = Path(args.celestrak_dir)
    satcat = load_satcat_metadata(args.satcat_path) if Path(args.satcat_path).exists() else {}
    builder = WorldModelFeatureBuilder(
        FeatureBuilderConfig(
            ephemeris_path=Path(args.ephemeris_path) if args.ephemeris_path else None,
            maneuver_detection=ManeuverDetectionConfig(
                position_residual_threshold_km=args.position_residual_threshold_km,
                velocity_residual_threshold_m_s=args.velocity_residual_threshold_m_s,
            ),
        )
    )

    sources = _load_sequence_sources(spacetrack_dir, celestrak_dir)
    output_dir = Path(args.output_dir)
    manifest = []

    for norad_cat_id, (source, rows) in sorted(sources.items()):
        sequence: PreparedSequence | None = builder.build_sequence(
            norad_cat_id=norad_cat_id,
            rows=rows,
            metadata=satcat.get(norad_cat_id),
            source=source,
        )
        if sequence is None:
            continue

        samples = make_window_samples(
            sequence,
            window_size=args.window_size,
            stride=args.stride,
            prediction_horizon_steps=args.prediction_horizon_steps,
        )
        for index, sample in enumerate(samples):
            path = output_dir / f"{norad_cat_id}_{index:04d}.npz"
            _save_window(path, sample)
            manifest.append({"norad_cat_id": norad_cat_id, "path": str(path.resolve())})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
