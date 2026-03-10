from __future__ import annotations

import torch

from rso_world_model.config import ModelConfig
from rso_world_model.model.world_model import RSOWorldModel


def test_world_model_output_shapes() -> None:
    config = ModelConfig(
        sequence_input_dim=71,
        static_input_dim=16,
        temporal_channels=[64, 64, 96, 96, 128],
    )
    model = RSOWorldModel(config)
    batch = 4
    steps = 96
    outputs = model(
        sequence_features=torch.randn(batch, steps, 71),
        feature_mask=torch.ones(batch, steps, 71),
        static_features=torch.randn(batch, 16),
        static_feature_mask=torch.ones(batch, 16),
    )

    assert outputs["maneuver_probability"].shape == (batch,)
    assert outputs["next_maneuver_time"].shape == (batch,)
    assert outputs["maneuver_class"].shape == (batch, config.maneuver_class_count)
    assert outputs["maneuver_purpose"].shape == (batch, config.maneuver_purpose_count)
    assert outputs["delta_v_bucket"].shape == (batch, config.delta_v_bucket_count)
    assert outputs["remaining_delta_v_estimate"].shape == (batch,)
    assert outputs["residual_growth"].shape == (batch,)

