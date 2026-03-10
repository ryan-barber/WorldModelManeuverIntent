from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from rso_world_model.data.schemas import ManeuverEvent, OrbitalElementRecord, SatelliteMetadata
from rso_world_model.features.orbital import mean_motion_to_semi_major_axis_km, propagate_record


@dataclass(slots=True)
class ManeuverDetectionConfig:
    position_residual_threshold_km: float = 25.0
    velocity_residual_threshold_m_s: float = 2.5


def classify_maneuver(previous: OrbitalElementRecord, current: OrbitalElementRecord, delta_v_m_s: float) -> tuple[str, str]:
    delta_sma = mean_motion_to_semi_major_axis_km(current.mean_motion_rev_per_day) - mean_motion_to_semi_major_axis_km(
        previous.mean_motion_rev_per_day
    )
    delta_inc = current.inclination_deg - previous.inclination_deg
    if abs(delta_inc) > 0.05:
        return "plane_change", "reorientation"
    if delta_sma > 5.0:
        return "orbit_raise", "relocation"
    if delta_sma < -5.0:
        return "orbit_lower", "disposal"
    if delta_v_m_s > 1.0:
        return "stationkeeping", "stationkeeping"
    return "phasing", "phasing"


def detect_maneuver(
    previous_record: OrbitalElementRecord | None,
    current_record: OrbitalElementRecord,
    config: ManeuverDetectionConfig,
) -> ManeuverEvent | None:
    if previous_record is None:
        return None

    predicted = propagate_record(previous_record, current_record.epoch)
    observed = propagate_record(current_record, current_record.epoch)

    position_residual = observed.position_km - predicted.position_km
    velocity_residual = observed.velocity_km_s - predicted.velocity_km_s
    position_residual_km = float(np.linalg.norm(position_residual))
    velocity_residual_km_s = float(np.linalg.norm(velocity_residual))
    delta_v_vector_m_s = velocity_residual.astype(np.float32) * 1000.0
    delta_v_magnitude_m_s = float(np.linalg.norm(delta_v_vector_m_s))
    detected = (
        position_residual_km >= config.position_residual_threshold_km
        or delta_v_magnitude_m_s >= config.velocity_residual_threshold_m_s
    )
    event_class, event_purpose = classify_maneuver(previous_record, current_record, delta_v_magnitude_m_s)
    return ManeuverEvent(
        norad_cat_id=current_record.norad_cat_id,
        event_epoch=current_record.epoch,
        residual_position_km=position_residual_km,
        residual_velocity_km_s=velocity_residual_km_s,
        delta_v_vector_m_s=delta_v_vector_m_s,
        delta_v_magnitude_m_s=delta_v_magnitude_m_s,
        propagated_from_epoch=previous_record.epoch,
        maneuver_detected=detected,
        event_class=event_class if detected else "none",
        event_purpose=event_purpose if detected else "unknown",
    )


def maneuver_history_features(events: list[ManeuverEvent], epoch: datetime) -> dict[str, float]:
    detected_events = [event for event in events if event.maneuver_detected and event.event_epoch <= epoch]
    if not detected_events:
        return {
            "time_since_last_maneuver_s": -1.0,
            "cumulative_delta_v_m_s": 0.0,
            "maneuver_frequency": 0.0,
            "delta_v_last_event_m_s": 0.0,
        }

    last_event = detected_events[-1]
    cumulative_delta_v = float(sum(event.delta_v_magnitude_m_s for event in detected_events))
    horizon_days = max((epoch - detected_events[0].event_epoch).total_seconds() / 86400.0, 1e-6)
    frequency = len(detected_events) / horizon_days
    return {
        "time_since_last_maneuver_s": float((epoch - last_event.event_epoch).total_seconds()),
        "cumulative_delta_v_m_s": cumulative_delta_v,
        "maneuver_frequency": float(frequency),
        "delta_v_last_event_m_s": last_event.delta_v_magnitude_m_s,
    }


def estimate_total_delta_v_capacity(metadata: SatelliteMetadata | None) -> float:
    if metadata is None:
        return float("nan")

    object_type = (metadata.object_type or "").lower()
    orbit_class = (metadata.orbit_class or "").lower()
    if "payload" not in object_type and "sat" not in object_type:
        return float("nan")
    if "geo" in orbit_class:
        return 1500.0
    if "leo" in orbit_class:
        return 300.0
    if "meo" in orbit_class:
        return 800.0
    return 500.0


def propulsion_features(metadata: SatelliteMetadata | None, cumulative_delta_v_m_s: float) -> dict[str, float]:
    total_capacity = estimate_total_delta_v_capacity(metadata)
    if np.isnan(total_capacity):
        return {
            "total_delta_v_capacity_m_s": float("nan"),
            "estimated_remaining_delta_v_m_s": float("nan"),
        }
    remaining = max(total_capacity - cumulative_delta_v_m_s, 0.0)
    return {
        "total_delta_v_capacity_m_s": total_capacity,
        "estimated_remaining_delta_v_m_s": remaining,
    }

