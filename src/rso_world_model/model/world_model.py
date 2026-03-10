from __future__ import annotations

import torch
from torch import nn

from rso_world_model.config import ModelConfig
from rso_world_model.model.tcn import TemporalConvNet


class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RSOWorldModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.sequence_projection = nn.Sequential(
            nn.Linear(config.sequence_input_dim * 2, config.input_projection_dim),
            nn.ReLU(),
        )
        self.static_projection = nn.Sequential(
            nn.Linear(config.static_input_dim * 2, config.static_hidden_dim),
            nn.ReLU(),
        )
        self.temporal_backbone = TemporalConvNet(
            num_inputs=config.input_projection_dim,
            channels=config.temporal_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        fused_dim = config.temporal_channels[-1] + config.static_hidden_dim
        self.maneuver_probability = PredictionHead(fused_dim, config.head_hidden_dim, 1)
        self.next_maneuver_time = PredictionHead(fused_dim, config.head_hidden_dim, 1)
        self.maneuver_class = PredictionHead(fused_dim, config.head_hidden_dim, config.maneuver_class_count)
        self.maneuver_purpose = PredictionHead(fused_dim, config.head_hidden_dim, config.maneuver_purpose_count)
        self.delta_v_bucket = PredictionHead(fused_dim, config.head_hidden_dim, config.delta_v_bucket_count)
        self.remaining_delta_v_estimate = PredictionHead(fused_dim, config.head_hidden_dim, 1)
        self.residual_growth = PredictionHead(fused_dim, config.head_hidden_dim, 1)

    def forward(
        self,
        sequence_features: torch.Tensor,
        feature_mask: torch.Tensor,
        static_features: torch.Tensor,
        static_feature_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        masked_features = sequence_features * feature_mask
        temporal_input = torch.cat([masked_features, feature_mask], dim=-1)
        temporal_input = self.sequence_projection(temporal_input)
        temporal_input = temporal_input.transpose(1, 2)
        temporal_features = self.temporal_backbone(temporal_input)
        temporal_weights = (feature_mask.sum(dim=-1, keepdim=True) > 0).float().transpose(1, 2)
        pooled = (temporal_features * temporal_weights).sum(dim=-1) / temporal_weights.sum(dim=-1).clamp_min(1.0)

        masked_static = static_features * static_feature_mask
        static_input = torch.cat([masked_static, static_feature_mask], dim=-1)
        static_context = self.static_projection(static_input)
        fused = torch.cat([pooled, static_context], dim=-1)
        return {
            "maneuver_probability": self.maneuver_probability(fused).squeeze(-1),
            "next_maneuver_time": self.next_maneuver_time(fused).squeeze(-1),
            "maneuver_class": self.maneuver_class(fused),
            "maneuver_purpose": self.maneuver_purpose(fused),
            "delta_v_bucket": self.delta_v_bucket(fused),
            "remaining_delta_v_estimate": self.remaining_delta_v_estimate(fused).squeeze(-1),
            "residual_growth": self.residual_growth(fused).squeeze(-1),
        }
