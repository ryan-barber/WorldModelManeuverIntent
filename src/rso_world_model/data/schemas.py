from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


def parse_epoch(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "null", "NULL"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class OrbitalElementRecord:
    norad_cat_id: int
    epoch: datetime
    inclination_deg: float
    eccentricity: float
    raan_deg: float
    argument_of_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float
    bstar: float = float("nan")
    mean_motion_dot: float = 0.0
    mean_motion_ddot: float = 0.0
    object_name: str | None = None
    source: str = "unknown"

    @classmethod
    def from_gp_row(cls, row: dict[str, Any], source: str) -> "OrbitalElementRecord":
        return cls(
            norad_cat_id=int(row["NORAD_CAT_ID"]),
            epoch=parse_epoch(row["EPOCH"]),
            inclination_deg=safe_float(row.get("INCLINATION")),
            eccentricity=safe_float(row.get("ECCENTRICITY")),
            raan_deg=safe_float(row.get("RA_OF_ASC_NODE")),
            argument_of_perigee_deg=safe_float(row.get("ARG_OF_PERICENTER")),
            mean_anomaly_deg=safe_float(row.get("MEAN_ANOMALY")),
            mean_motion_rev_per_day=safe_float(row.get("MEAN_MOTION")),
            bstar=safe_float(row.get("BSTAR")),
            mean_motion_dot=safe_float(row.get("MEAN_MOTION_DOT"), 0.0),
            mean_motion_ddot=safe_float(row.get("MEAN_MOTION_DDOT"), 0.0),
            object_name=row.get("OBJECT_NAME"),
            source=source,
        )


@dataclass(slots=True)
class SatelliteMetadata:
    norad_cat_id: int
    object_name: str | None = None
    country: str | None = None
    launch_date: datetime | None = None
    object_type: str | None = None
    operator: str | None = None
    orbit_class: str | None = None
    constellation_group: str | None = None
    mission_class: str | None = None

    @classmethod
    def from_satcat_row(cls, row: dict[str, Any]) -> "SatelliteMetadata":
        launch_date = row.get("LAUNCH_DATE")
        parsed_launch_date = None
        if launch_date:
            parsed_launch_date = datetime.fromisoformat(launch_date).replace(tzinfo=timezone.utc)

        return cls(
            norad_cat_id=int(row["NORAD_CAT_ID"]),
            object_name=row.get("OBJECT_NAME"),
            country=row.get("COUNTRY"),
            launch_date=parsed_launch_date,
            object_type=row.get("OBJECT_TYPE"),
            operator=row.get("OPERATOR"),
            orbit_class=row.get("ORBIT_CLASS"),
            constellation_group=row.get("GROUP"),
            mission_class=row.get("MISSION_CLASS") or row.get("OBJECT_TYPE"),
        )


@dataclass(slots=True)
class StateVector:
    epoch: datetime
    position_km: np.ndarray
    velocity_km_s: np.ndarray


@dataclass(slots=True)
class ManeuverEvent:
    norad_cat_id: int
    event_epoch: datetime
    residual_position_km: float
    residual_velocity_km_s: float
    delta_v_vector_m_s: np.ndarray
    delta_v_magnitude_m_s: float
    propagated_from_epoch: datetime
    maneuver_detected: bool
    event_class: str = "none"
    event_purpose: str = "unknown"


@dataclass(slots=True)
class PreparedSequence:
    norad_cat_id: int
    timestamps: list[str]
    sequence_features: np.ndarray
    feature_mask: np.ndarray
    static_features: np.ndarray
    static_feature_mask: np.ndarray
    feature_names: list[str]
    static_feature_names: list[str]
    maneuver_events: list[ManeuverEvent] = field(default_factory=list)

