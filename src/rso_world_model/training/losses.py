from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rso_world_model.config import LossWeights


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = values * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    items: dict[str, torch.Tensor]


class MultiTaskWorldModelLoss(nn.Module):
    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.regression = nn.SmoothL1Loss(reduction="none")
        self.classification = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBreakdown:
        items: dict[str, torch.Tensor] = {}

        maneuver_probability_loss = _masked_mean(
            self.bce(outputs["maneuver_probability"], batch["maneuver_probability"].float()),
            batch["maneuver_probability_mask"],
        )
        items["maneuver_probability"] = maneuver_probability_loss * self.weights.maneuver_probability

        next_maneuver_time_loss = _masked_mean(
            self.regression(outputs["next_maneuver_time"], batch["next_maneuver_time"].float()),
            batch["next_maneuver_time_mask"],
        )
        items["next_maneuver_time"] = next_maneuver_time_loss * self.weights.next_maneuver_time

        maneuver_class_loss = _masked_mean(
            self.classification(outputs["maneuver_class"], batch["maneuver_class"].long()),
            batch["maneuver_class_mask"],
        )
        items["maneuver_class"] = maneuver_class_loss * self.weights.maneuver_class

        maneuver_purpose_loss = _masked_mean(
            self.classification(outputs["maneuver_purpose"], batch["maneuver_purpose"].long()),
            batch["maneuver_purpose_mask"],
        )
        items["maneuver_purpose"] = maneuver_purpose_loss * self.weights.maneuver_purpose

        delta_v_bucket_loss = _masked_mean(
            self.classification(outputs["delta_v_bucket"], batch["delta_v_bucket"].long()),
            batch["delta_v_bucket_mask"],
        )
        items["delta_v_bucket"] = delta_v_bucket_loss * self.weights.delta_v_bucket

        remaining_delta_v_loss = _masked_mean(
            self.regression(outputs["remaining_delta_v_estimate"], batch["remaining_delta_v_estimate"].float()),
            batch["remaining_delta_v_estimate_mask"],
        )
        items["remaining_delta_v_estimate"] = remaining_delta_v_loss * self.weights.remaining_delta_v_estimate

        residual_growth_loss = _masked_mean(
            self.regression(outputs["residual_growth"], batch["residual_growth"].float()),
            batch["residual_growth_mask"],
        )
        items["residual_growth"] = residual_growth_loss * self.weights.residual_growth

        total = sum(items.values())
        return LossBreakdown(total=total, items=items)

