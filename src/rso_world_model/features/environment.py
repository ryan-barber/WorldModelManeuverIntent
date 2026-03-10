from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from skyfield.api import load, load_file

from rso_world_model.data.schemas import StateVector
from rso_world_model.features.orbital import EARTH_RADIUS_KM


def _safe_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


@dataclass(slots=True)
class EnvironmentalContext:
    sun_vector_km: np.ndarray
    moon_vector_km: np.ndarray
    earth_vector_km: np.ndarray
    sun_angle_deg: float
    moon_angle_deg: float
    earth_angle_deg: float
    in_earth_shadow: float


class EnvironmentalFeatureComputer:
    def __init__(self, ephemeris_path: str | Path | None = None) -> None:
        self.timescale = load.timescale()
        self.ephemeris_path = Path(ephemeris_path) if ephemeris_path else None
        self.ephemeris = load_file(self.ephemeris_path) if self.ephemeris_path else None

    def available(self) -> bool:
        return self.ephemeris is not None

    def compute(self, state: StateVector) -> EnvironmentalContext:
        if self.ephemeris is None:
            nan3 = np.full(3, np.nan, dtype=np.float32)
            return EnvironmentalContext(
                sun_vector_km=nan3.copy(),
                moon_vector_km=nan3.copy(),
                earth_vector_km=(-state.position_km).astype(np.float32),
                sun_angle_deg=float("nan"),
                moon_angle_deg=float("nan"),
                earth_angle_deg=180.0,
                in_earth_shadow=float("nan"),
            )

        earth = self.ephemeris["earth"]
        sun = self.ephemeris["sun"]
        moon = self.ephemeris["moon"]
        t = self.timescale.from_datetime(state.epoch)

        sun_from_earth = earth.at(t).observe(sun).position.km.astype(np.float32)
        moon_from_earth = earth.at(t).observe(moon).position.km.astype(np.float32)
        sun_vector = sun_from_earth - state.position_km
        moon_vector = moon_from_earth - state.position_km
        earth_vector = (-state.position_km).astype(np.float32)

        sun_direction = sun_from_earth / max(np.linalg.norm(sun_from_earth), 1e-8)
        sat_projection = float(np.dot(state.position_km, sun_direction))
        lateral_distance = float(np.linalg.norm(state.position_km - sat_projection * sun_direction))
        in_shadow = 1.0 if sat_projection < 0.0 and lateral_distance < EARTH_RADIUS_KM else 0.0

        return EnvironmentalContext(
            sun_vector_km=sun_vector.astype(np.float32),
            moon_vector_km=moon_vector.astype(np.float32),
            earth_vector_km=earth_vector,
            sun_angle_deg=_safe_angle_deg(state.position_km, sun_vector),
            moon_angle_deg=_safe_angle_deg(state.position_km, moon_vector),
            earth_angle_deg=_safe_angle_deg(state.position_km, earth_vector),
            in_earth_shadow=in_shadow,
        )

