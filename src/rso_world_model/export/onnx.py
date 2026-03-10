from __future__ import annotations

from pathlib import Path

import torch

from rso_world_model.config import AppConfig
from rso_world_model.model.world_model import RSOWorldModel


def export_to_onnx(config: AppConfig, checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RSOWorldModel(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target = Path(output_path or config.export.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    batch = 1
    sequence_length = 96
    sequence_features = torch.randn(batch, sequence_length, config.model.sequence_input_dim)
    feature_mask = torch.ones(batch, sequence_length, config.model.sequence_input_dim)
    static_features = torch.randn(batch, config.model.static_input_dim)
    static_feature_mask = torch.ones(batch, config.model.static_input_dim)

    torch.onnx.export(
        model,
        (sequence_features, feature_mask, static_features, static_feature_mask),
        target,
        input_names=["sequence_features", "feature_mask", "static_features", "static_feature_mask"],
        output_names=[
            "maneuver_probability",
            "next_maneuver_time",
            "maneuver_class",
            "maneuver_purpose",
            "delta_v_bucket",
            "remaining_delta_v_estimate",
            "residual_growth",
        ],
        dynamic_axes={
            "sequence_features": {0: "batch", 1: "sequence"},
            "feature_mask": {0: "batch", 1: "sequence"},
            "static_features": {0: "batch"},
            "static_feature_mask": {0: "batch"},
        },
        opset_version=config.export.opset,
    )
    return target
