from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from rso_world_model.config import AppConfig, dataclass_to_dict
from rso_world_model.model.world_model import RSOWorldModel
from rso_world_model.training.dataset import PreparedSequenceDataset
from rso_world_model.training.losses import MultiTaskWorldModelLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_dataloaders(config: AppConfig) -> tuple[DataLoader, DataLoader]:
    dataset = PreparedSequenceDataset(config.data.manifest_path)
    validation_size = int(len(dataset) * config.data.validation_split)
    train_size = len(dataset) - validation_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    common = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.training.device.startswith("cuda"),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskWorldModelLoss,
    device: torch.device,
    optimizer: AdamW | None,
    grad_clip_norm: float,
    mixed_precision: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    scaler = torch.amp.GradScaler(enabled=mixed_precision and device.type == "cuda")
    totals: dict[str, float] = {"loss": 0.0}
    batches = 0

    for batch in dataloader:
        batch = _move_batch(batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
                outputs = model(
                    batch["sequence_features"],
                    batch["feature_mask"],
                    batch["static_features"],
                    batch["static_feature_mask"],
                )
                breakdown = criterion(outputs, batch)

            if is_training:
                scaler.scale(breakdown.total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        totals["loss"] += float(breakdown.total.detach().cpu())
        for key, value in breakdown.items.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_world_model(config: AppConfig) -> Path:
    set_seed(config.training.seed)
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(config)

    model = RSOWorldModel(config.model).to(device)
    criterion = MultiTaskWorldModelLoss(config.training.loss_weights)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)

    checkpoint_dir = config.training.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_val_loss = float("inf")

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=config.training.grad_clip_norm,
            mixed_precision=config.training.mixed_precision,
        )
        val_metrics = _run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip_norm=config.training.grad_clip_norm,
            mixed_precision=False,
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dataclass_to_dict(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(state, best_path)

        metrics_path = checkpoint_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}))
            handle.write("\n")

    return best_path

