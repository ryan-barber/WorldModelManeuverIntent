from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, manifest_path: str | Path) -> None:
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            self.items = json.load(handle)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        with np.load(item["path"]) as payload:
            sample = {
                "sequence_features": torch.from_numpy(payload["sequence_features"]).float(),
                "feature_mask": torch.from_numpy(payload["feature_mask"]).float(),
                "static_features": torch.from_numpy(payload["static_features"]).float(),
                "static_feature_mask": torch.from_numpy(payload["static_feature_mask"]).float(),
            }
            for key in [
                "maneuver_probability",
                "next_maneuver_time",
                "maneuver_class",
                "maneuver_purpose",
                "delta_v_bucket",
                "remaining_delta_v_estimate",
                "residual_growth",
            ]:
                sample[key] = torch.from_numpy(payload[key])
                sample[f"{key}_mask"] = torch.from_numpy(payload[f"{key}_mask"]).float()
            return sample

