from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from rso_world_model.data.io import ensure_dir, write_json
from rso_world_model.data.schemas import OrbitalElementRecord, PreparedSequence, SatelliteMetadata
from rso_world_model.features.environment import EnvironmentalFeatureComputer
from rso_world_model.features.maneuvers import (
    ManeuverDetectionConfig,
    detect_maneuver,
    maneuver_history_features,
    propulsion_features,
)
from rso_world_model.features.orbital import compute_delta_features, compute_orbital_features, propagate_record


def _stable_hash_fraction(value: str | None) -> float:
    if not value:
        return 0.0
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _scale_unknown(value: float) -> tuple[float, float]:
    if np.isnan(value):
        return 0.0, 0.0
    return value, 1.0


@dataclass(slots=True)
class FeatureBuilderConfig:
    ephemeris_path: Path | None = None
    maneuver_detection: ManeuverDetectionConfig = field(default_factory=ManeuverDetectionConfig)


class WorldModelFeatureBuilder:
    def __init__(self, config: FeatureBuilderConfig | None = None) -> None:
        self.config = config or FeatureBuilderConfig()
        self.environment = EnvironmentalFeatureComputer(self.config.ephemeris_path)

    def build_sequence(
        self,
        norad_cat_id: int,
        rows: Iterable[dict],
        metadata: SatelliteMetadata | None = None,
        source: str = "unknown",
    ) -> PreparedSequence | None:
        records = sorted(
            [OrbitalElementRecord.from_gp_row(row, source=source) for row in rows],
            key=lambda record: record.epoch,
        )
        if len(records) < 8:
            return None

        timestamps: list[str] = []
        feature_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        maneuver_events = []

        previous_record = None
        previous_state = None

        for record in records:
            timestamps.append(record.epoch.isoformat())
            current_state = propagate_record(record)
            orbital = compute_orbital_features(record, current_state)
            environment = self.environment.compute(current_state)
            delta = compute_delta_features(previous_record, record, previous_state, current_state)
            event = detect_maneuver(previous_record, record, self.config.maneuver_detection)
            if event is not None:
                maneuver_events.append(event)
            history = maneuver_history_features(maneuver_events, record.epoch)
            propulsion = propulsion_features(metadata, history["cumulative_delta_v_m_s"])

            row = np.asarray(
                [
                    *current_state.position_km.tolist(),
                    *current_state.velocity_km_s.tolist(),
                    orbital["position_norm_km"],
                    orbital["velocity_norm_km_s"],
                    record.inclination_deg,
                    record.eccentricity,
                    record.raan_deg,
                    record.argument_of_perigee_deg,
                    record.mean_anomaly_deg,
                    record.mean_motion_rev_per_day,
                    record.bstar,
                    orbital["semi_major_axis_km"],
                    orbital["true_anomaly_deg"],
                    np.sin(np.deg2rad(record.inclination_deg)),
                    np.cos(np.deg2rad(record.inclination_deg)),
                    np.sin(np.deg2rad(record.raan_deg)),
                    np.cos(np.deg2rad(record.raan_deg)),
                    np.sin(np.deg2rad(record.argument_of_perigee_deg)),
                    np.cos(np.deg2rad(record.argument_of_perigee_deg)),
                    np.sin(np.deg2rad(record.mean_anomaly_deg)),
                    np.cos(np.deg2rad(record.mean_anomaly_deg)),
                    orbital["latitude_deg"],
                    orbital["longitude_deg"],
                    orbital["altitude_km"],
                    *environment.sun_vector_km.tolist(),
                    *environment.moon_vector_km.tolist(),
                    *environment.earth_vector_km.tolist(),
                    np.linalg.norm(environment.sun_vector_km),
                    np.linalg.norm(environment.moon_vector_km),
                    np.linalg.norm(environment.earth_vector_km),
                    environment.sun_angle_deg,
                    environment.moon_angle_deg,
                    environment.earth_angle_deg,
                    environment.in_earth_shadow,
                    *delta.delta_position_km.tolist(),
                    *delta.delta_velocity_km_s.tolist(),
                    *delta.delta_elements.tolist(),
                    delta.delta_time_s,
                    0.0 if event is None else event.residual_position_km,
                    0.0 if event is None else event.residual_velocity_km_s,
                    *([0.0, 0.0, 0.0] if event is None else (event.delta_v_vector_m_s / 1000.0).tolist()),
                    0.0 if event is None else event.delta_v_magnitude_m_s,
                    0.0 if event is None else float(event.maneuver_detected),
                    history["time_since_last_maneuver_s"],
                    history["cumulative_delta_v_m_s"],
                    history["maneuver_frequency"],
                    history["delta_v_last_event_m_s"],
                    propulsion["total_delta_v_capacity_m_s"],
                    propulsion["estimated_remaining_delta_v_m_s"],
                ],
                dtype=np.float32,
            )

            mask = np.isfinite(row).astype(np.float32)
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            feature_rows.append(row)
            mask_rows.append(mask)

            previous_record = record
            previous_state = current_state

        static_vector, static_mask, static_names = self._build_static_features(metadata, records[-1].epoch)
        return PreparedSequence(
            norad_cat_id=norad_cat_id,
            timestamps=timestamps,
            sequence_features=np.stack(feature_rows),
            feature_mask=np.stack(mask_rows),
            static_features=static_vector,
            static_feature_mask=static_mask,
            static_feature_names=static_names,
            feature_names=self.feature_names(),
            maneuver_events=maneuver_events,
        )

    @staticmethod
    def _build_static_features(metadata: SatelliteMetadata | None, reference_epoch: datetime) -> tuple[np.ndarray, np.ndarray, list[str]]:
        launch_age_days = float("nan")
        if metadata and metadata.launch_date:
            launch_age_days = (reference_epoch - metadata.launch_date).total_seconds() / 86400.0

        raw_values = np.asarray(
            [
                launch_age_days,
                _stable_hash_fraction(metadata.country if metadata else None),
                _stable_hash_fraction(metadata.operator if metadata else None),
                _stable_hash_fraction(metadata.object_type if metadata else None),
                _stable_hash_fraction(metadata.orbit_class if metadata else None),
                _stable_hash_fraction(metadata.constellation_group if metadata else None),
                _stable_hash_fraction(metadata.mission_class if metadata else None),
                1.0 if metadata and metadata.country else 0.0,
                1.0 if metadata and metadata.operator else 0.0,
                1.0 if metadata and metadata.object_type else 0.0,
                1.0 if metadata and metadata.orbit_class else 0.0,
                1.0 if metadata and metadata.constellation_group else 0.0,
                1.0 if metadata and metadata.mission_class else 0.0,
                1.0 if metadata and metadata.launch_date else 0.0,
                1.0 if metadata else 0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        mask = np.isfinite(raw_values).astype(np.float32)
        raw_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)
        names = [
            "launch_age_days",
            "country_hash",
            "operator_hash",
            "object_type_hash",
            "orbit_class_hash",
            "constellation_hash",
            "mission_class_hash",
            "country_known",
            "operator_known",
            "object_type_known",
            "orbit_class_known",
            "constellation_known",
            "mission_class_known",
            "launch_date_known",
            "metadata_available",
            "static_bias",
        ]
        return raw_values, mask, names

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "position_x_km",
            "position_y_km",
            "position_z_km",
            "velocity_x_km_s",
            "velocity_y_km_s",
            "velocity_z_km_s",
            "position_norm_km",
            "velocity_norm_km_s",
            "inclination_deg",
            "eccentricity",
            "raan_deg",
            "argument_of_perigee_deg",
            "mean_anomaly_deg",
            "mean_motion_rev_per_day",
            "bstar",
            "semi_major_axis_km",
            "true_anomaly_deg",
            "sin_inclination",
            "cos_inclination",
            "sin_raan",
            "cos_raan",
            "sin_argument_of_perigee",
            "cos_argument_of_perigee",
            "sin_mean_anomaly",
            "cos_mean_anomaly",
            "latitude_deg",
            "longitude_deg",
            "altitude_km",
            "sun_vector_x_km",
            "sun_vector_y_km",
            "sun_vector_z_km",
            "moon_vector_x_km",
            "moon_vector_y_km",
            "moon_vector_z_km",
            "earth_vector_x_km",
            "earth_vector_y_km",
            "earth_vector_z_km",
            "sun_distance_km",
            "moon_distance_km",
            "earth_distance_km",
            "sun_angle_deg",
            "moon_angle_deg",
            "earth_angle_deg",
            "in_earth_shadow",
            "delta_position_x_km",
            "delta_position_y_km",
            "delta_position_z_km",
            "delta_velocity_x_km_s",
            "delta_velocity_y_km_s",
            "delta_velocity_z_km_s",
            "delta_inclination_deg",
            "delta_eccentricity",
            "delta_raan_deg",
            "delta_argument_of_perigee_deg",
            "delta_mean_anomaly_deg",
            "delta_mean_motion_rev_per_day",
            "delta_bstar",
            "delta_time_s",
            "residual_position_km",
            "residual_velocity_km_s",
            "delta_v_x_km_s",
            "delta_v_y_km_s",
            "delta_v_z_km_s",
            "delta_v_magnitude_m_s",
            "maneuver_detected",
            "time_since_last_maneuver_s",
            "cumulative_delta_v_m_s",
            "maneuver_frequency",
            "delta_v_last_event_m_s",
            "total_delta_v_capacity_m_s",
            "estimated_remaining_delta_v_m_s",
        ]


def save_prepared_sequence(output_dir: str | Path, sequence: PreparedSequence) -> Path:
    output = ensure_dir(output_dir)
    target = output / f"{sequence.norad_cat_id}.npz"
    np.savez_compressed(
        target,
        norad_cat_id=sequence.norad_cat_id,
        timestamps=np.asarray(sequence.timestamps),
        sequence_features=sequence.sequence_features,
        feature_mask=sequence.feature_mask,
        static_features=sequence.static_features,
        static_feature_mask=sequence.static_feature_mask,
        feature_names=np.asarray(sequence.feature_names),
        static_feature_names=np.asarray(sequence.static_feature_names),
    )
    return target


def write_manifest(output_dir: str | Path, items: list[dict]) -> Path:
    target = ensure_dir(output_dir) / "manifest.json"
    write_json(target, items)
    return target
