from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from rso_world_model.data.schemas import ManeuverEvent, PreparedSequence


MANEUVER_CLASS_TO_ID = {
    "none": 0,
    "stationkeeping": 1,
    "orbit_raise": 2,
    "orbit_lower": 3,
    "plane_change": 4,
    "phasing": 5,
}

MANEUVER_PURPOSE_TO_ID = {
    "unknown": 0,
    "stationkeeping": 1,
    "relocation": 2,
    "disposal": 3,
    "reorientation": 4,
    "phasing": 5,
}


def bucket_delta_v(delta_v_m_s: float) -> int:
    if delta_v_m_s < 1.0:
        return 0
    if delta_v_m_s < 5.0:
        return 1
    if delta_v_m_s < 20.0:
        return 2
    if delta_v_m_s < 50.0:
        return 3
    return 4


@dataclass(slots=True)
class WindowSample:
    sequence_features: np.ndarray
    feature_mask: np.ndarray
    static_features: np.ndarray
    static_feature_mask: np.ndarray
    targets: dict[str, np.ndarray]
    target_mask: dict[str, np.ndarray]


def make_window_samples(
    sequence: PreparedSequence,
    window_size: int = 96,
    stride: int = 16,
    prediction_horizon_steps: int = 16,
) -> list[WindowSample]:
    feature_dim = sequence.sequence_features.shape[1]
    samples: list[WindowSample] = []
    detected_events = [event for event in sequence.maneuver_events if event.maneuver_detected]
    event_lookup = {event.event_epoch.isoformat(): event for event in detected_events}

    for start in range(0, max(sequence.sequence_features.shape[0] - window_size + 1, 0), stride):
        end = start + window_size
        if end > sequence.sequence_features.shape[0]:
            break

        window_timestamps = sequence.timestamps[start:end]
        future_timestamps = sequence.timestamps[end : min(end + prediction_horizon_steps, len(sequence.timestamps))]
        future_events = [event_lookup[ts] for ts in future_timestamps if ts in event_lookup]
        next_event = future_events[0] if future_events else None

        next_time_s = -1.0
        if next_event is not None:
            current_epoch = datetime.fromisoformat(window_timestamps[-1])
            next_time_s = max(0.0, (next_event.event_epoch - current_epoch).total_seconds())

        remaining_delta_v = float(sequence.sequence_features[end - 1, -1])
        residual_growth = float(sequence.sequence_features[end - 1, 58])
        maneuver_class = MANEUVER_CLASS_TO_ID.get(next_event.event_class if next_event else "none", 0)
        maneuver_purpose = MANEUVER_PURPOSE_TO_ID.get(next_event.event_purpose if next_event else "unknown", 0)
        delta_v_bucket = bucket_delta_v(next_event.delta_v_magnitude_m_s if next_event else 0.0)
        probability = 1.0 if next_event else 0.0

        target_mask = {
            "maneuver_probability": np.asarray(1.0, dtype=np.float32),
            "next_maneuver_time": np.asarray(1.0 if next_event else 0.0, dtype=np.float32),
            "maneuver_class": np.asarray(1.0 if next_event else 0.0, dtype=np.float32),
            "maneuver_purpose": np.asarray(1.0 if next_event else 0.0, dtype=np.float32),
            "delta_v_bucket": np.asarray(1.0 if next_event else 0.0, dtype=np.float32),
            "remaining_delta_v_estimate": np.asarray(1.0 if np.isfinite(remaining_delta_v) else 0.0, dtype=np.float32),
            "residual_growth": np.asarray(1.0, dtype=np.float32),
        }
        targets = {
            "maneuver_probability": np.asarray(probability, dtype=np.float32),
            "next_maneuver_time": np.asarray(next_time_s, dtype=np.float32),
            "maneuver_class": np.asarray(maneuver_class, dtype=np.int64),
            "maneuver_purpose": np.asarray(maneuver_purpose, dtype=np.int64),
            "delta_v_bucket": np.asarray(delta_v_bucket, dtype=np.int64),
            "remaining_delta_v_estimate": np.asarray(0.0 if not np.isfinite(remaining_delta_v) else remaining_delta_v, dtype=np.float32),
            "residual_growth": np.asarray(residual_growth, dtype=np.float32),
        }

        samples.append(
            WindowSample(
                sequence_features=sequence.sequence_features[start:end].reshape(window_size, feature_dim),
                feature_mask=sequence.feature_mask[start:end].reshape(window_size, feature_dim),
                static_features=sequence.static_features.copy(),
                static_feature_mask=sequence.static_feature_mask.copy(),
                targets=targets,
                target_mask=target_mask,
            )
        )
    return samples
