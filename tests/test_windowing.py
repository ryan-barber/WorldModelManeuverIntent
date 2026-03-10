from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from rso_world_model.data.schemas import ManeuverEvent, PreparedSequence
from rso_world_model.training.windowing import make_window_samples


def test_window_generation() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    sequence = PreparedSequence(
        norad_cat_id=12345,
        timestamps=[(epoch + timedelta(minutes=5 * i)).isoformat() for i in range(20)],
        sequence_features=np.random.randn(20, 71).astype(np.float32),
        feature_mask=np.ones((20, 71), dtype=np.float32),
        static_features=np.ones(16, dtype=np.float32),
        static_feature_mask=np.ones(16, dtype=np.float32),
        feature_names=[f"f{i}" for i in range(71)],
        static_feature_names=[f"s{i}" for i in range(16)],
        maneuver_events=[
            ManeuverEvent(
                norad_cat_id=12345,
                event_epoch=epoch + timedelta(minutes=5 * 15),
                residual_position_km=50.0,
                residual_velocity_km_s=0.01,
                delta_v_vector_m_s=np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
                delta_v_magnitude_m_s=3.0,
                propagated_from_epoch=epoch + timedelta(minutes=5 * 14),
                maneuver_detected=True,
                event_class="stationkeeping",
                event_purpose="stationkeeping",
            )
        ],
    )
    samples = make_window_samples(sequence, window_size=8, stride=4, prediction_horizon_steps=8)
    assert samples
    assert samples[0].sequence_features.shape == (8, 71)
    assert "maneuver_probability" in samples[0].targets

