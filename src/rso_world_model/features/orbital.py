from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import atan2, cos, pi, sin, sqrt

import numpy as np
from sgp4.api import Satrec, WGS72, jday

from rso_world_model.data.schemas import OrbitalElementRecord, StateVector


EARTH_MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563


def _datetime_to_julian(dt: datetime) -> tuple[float, float]:
    utc = dt.astimezone(timezone.utc)
    return jday(
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
        utc.second + utc.microsecond / 1_000_000.0,
    )


def mean_motion_to_rad_per_min(mean_motion_rev_per_day: float) -> float:
    return mean_motion_rev_per_day * 2.0 * pi / 1440.0


def mean_motion_to_semi_major_axis_km(mean_motion_rev_per_day: float) -> float:
    mean_motion_rad_s = mean_motion_rev_per_day * 2.0 * pi / 86400.0
    return (EARTH_MU_KM3_S2 / (mean_motion_rad_s ** 2)) ** (1.0 / 3.0)


def solve_true_anomaly_deg(eccentricity: float, mean_anomaly_deg: float, iterations: int = 10) -> float:
    mean_anomaly = np.deg2rad(mean_anomaly_deg)
    eccentric_anomaly = mean_anomaly
    for _ in range(iterations):
        numerator = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        denominator = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        eccentric_anomaly -= numerator / max(denominator, 1e-8)
    true_anomaly = 2.0 * atan2(
        sqrt(1.0 + eccentricity) * sin(eccentric_anomaly / 2.0),
        sqrt(max(1.0 - eccentricity, 1e-8)) * cos(eccentric_anomaly / 2.0),
    )
    return float(np.rad2deg(true_anomaly) % 360.0)


def build_satrec(record: OrbitalElementRecord) -> Satrec:
    jd, fr = _datetime_to_julian(record.epoch)
    epoch_days = (jd + fr) - 2433281.5
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        int(record.norad_cat_id),
        epoch_days,
        0.0 if np.isnan(record.bstar) else record.bstar,
        record.mean_motion_dot,
        record.mean_motion_ddot,
        record.eccentricity,
        np.deg2rad(record.argument_of_perigee_deg),
        np.deg2rad(record.inclination_deg),
        np.deg2rad(record.mean_anomaly_deg),
        mean_motion_to_rad_per_min(record.mean_motion_rev_per_day),
        np.deg2rad(record.raan_deg),
    )
    return sat


def propagate_record(record: OrbitalElementRecord, target_epoch: datetime | None = None) -> StateVector:
    epoch = target_epoch or record.epoch
    jd, fr = _datetime_to_julian(epoch)
    sat = build_satrec(record)
    error_code, position_km, velocity_km_s = sat.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(
            f"SGP4 propagation failed for NORAD {record.norad_cat_id} at {epoch.isoformat()} with error {error_code}."
        )
    return StateVector(
        epoch=epoch,
        position_km=np.asarray(position_km, dtype=np.float32),
        velocity_km_s=np.asarray(velocity_km_s, dtype=np.float32),
    )


def greenwich_sidereal_angle_rad(epoch: datetime) -> float:
    jd, fr = _datetime_to_julian(epoch)
    jd_full = jd + fr
    t = (jd_full - 2451545.0) / 36525.0
    theta_deg = (
        280.46061837
        + 360.98564736629 * (jd_full - 2451545.0)
        + 0.000387933 * (t ** 2)
        - (t ** 3) / 38710000.0
    )
    return np.deg2rad(theta_deg % 360.0)


def eci_to_ecef(position_km: np.ndarray, epoch: datetime) -> np.ndarray:
    theta = greenwich_sidereal_angle_rad(epoch)
    rotation = np.array(
        [
            [cos(theta), sin(theta), 0.0],
            [-sin(theta), cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation @ position_km.astype(np.float64)


def ecef_to_geodetic(position_ecef_km: np.ndarray) -> tuple[float, float, float]:
    x, y, z = position_ecef_km
    semi_major = EARTH_RADIUS_KM
    eccentricity_sq = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2
    longitude = atan2(y, x)
    radius_xy = sqrt(x * x + y * y)
    latitude = atan2(z, radius_xy * (1 - eccentricity_sq))

    for _ in range(5):
        sin_lat = sin(latitude)
        n = semi_major / sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
        altitude = radius_xy / max(cos(latitude), 1e-8) - n
        latitude = atan2(z, radius_xy * (1.0 - eccentricity_sq * n / (n + altitude)))

    sin_lat = sin(latitude)
    n = semi_major / sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
    altitude = radius_xy / max(cos(latitude), 1e-8) - n
    return float(np.rad2deg(latitude)), float(np.rad2deg(longitude)), float(altitude)


def compute_geodetic_features(state: StateVector) -> dict[str, float]:
    ecef = eci_to_ecef(state.position_km, state.epoch)
    latitude_deg, longitude_deg, altitude_km = ecef_to_geodetic(ecef)
    return {
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "altitude_km": altitude_km,
    }


def compute_orbital_features(record: OrbitalElementRecord, state: StateVector) -> dict[str, float]:
    semi_major_axis_km = mean_motion_to_semi_major_axis_km(record.mean_motion_rev_per_day)
    true_anomaly_deg = solve_true_anomaly_deg(record.eccentricity, record.mean_anomaly_deg)
    position_norm = float(np.linalg.norm(state.position_km))
    velocity_norm = float(np.linalg.norm(state.velocity_km_s))
    geodetic = compute_geodetic_features(state)
    return {
        "semi_major_axis_km": semi_major_axis_km,
        "true_anomaly_deg": true_anomaly_deg,
        "position_norm_km": position_norm,
        "velocity_norm_km_s": velocity_norm,
        **geodetic,
    }


@dataclass(slots=True)
class DeltaFeatures:
    delta_time_s: float
    delta_position_km: np.ndarray
    delta_velocity_km_s: np.ndarray
    delta_elements: np.ndarray


def compute_delta_features(
    previous_record: OrbitalElementRecord | None,
    current_record: OrbitalElementRecord,
    previous_state: StateVector | None,
    current_state: StateVector,
) -> DeltaFeatures:
    if previous_record is None or previous_state is None:
        return DeltaFeatures(
            delta_time_s=0.0,
            delta_position_km=np.zeros(3, dtype=np.float32),
            delta_velocity_km_s=np.zeros(3, dtype=np.float32),
            delta_elements=np.zeros(7, dtype=np.float32),
        )

    delta_time_s = (current_record.epoch - previous_record.epoch).total_seconds()
    delta_elements = np.asarray(
        [
            current_record.inclination_deg - previous_record.inclination_deg,
            current_record.eccentricity - previous_record.eccentricity,
            current_record.raan_deg - previous_record.raan_deg,
            current_record.argument_of_perigee_deg - previous_record.argument_of_perigee_deg,
            current_record.mean_anomaly_deg - previous_record.mean_anomaly_deg,
            current_record.mean_motion_rev_per_day - previous_record.mean_motion_rev_per_day,
            current_record.bstar - previous_record.bstar,
        ],
        dtype=np.float32,
    )
    return DeltaFeatures(
        delta_time_s=float(delta_time_s),
        delta_position_km=(current_state.position_km - previous_state.position_km).astype(np.float32),
        delta_velocity_km_s=(current_state.velocity_km_s - previous_state.velocity_km_s).astype(np.float32),
        delta_elements=delta_elements,
    )

