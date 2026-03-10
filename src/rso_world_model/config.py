from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    manifest_path: Path
    batch_size: int = 32
    num_workers: int = 0
    validation_split: float = 0.1


@dataclass(slots=True)
class ModelConfig:
    sequence_input_dim: int
    static_input_dim: int
    input_projection_dim: int = 64
    temporal_channels: list[int] = field(default_factory=lambda: [64, 64, 96, 96, 128])
    kernel_size: int = 3
    dropout: float = 0.1
    static_hidden_dim: int = 32
    head_hidden_dim: int = 128
    maneuver_class_count: int = 6
    maneuver_purpose_count: int = 5
    delta_v_bucket_count: int = 5


@dataclass(slots=True)
class LossWeights:
    maneuver_probability: float = 1.0
    next_maneuver_time: float = 0.5
    maneuver_class: float = 1.0
    maneuver_purpose: float = 1.0
    delta_v_bucket: float = 0.75
    remaining_delta_v_estimate: float = 0.25
    residual_growth: float = 0.25


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda"
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    log_every: int = 20
    seed: int = 7
    loss_weights: LossWeights = field(default_factory=LossWeights)


@dataclass(slots=True)
class ExportConfig:
    opset: int = 17
    output_path: Path = Path("artifacts/export/rso_world_model.onnx")


@dataclass(slots=True)
class AppConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig = field(default_factory=ExportConfig)


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def load_app_config(path: str | Path) -> AppConfig:
    import yaml

    with _expand_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    data_cfg = payload["data"]
    training_cfg = payload["training"]
    export_cfg = payload.get("export", {})

    return AppConfig(
        data=DataConfig(
            manifest_path=_expand_path(data_cfg["manifest_path"]),
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 0),
            validation_split=data_cfg.get("validation_split", 0.1),
        ),
        model=ModelConfig(**payload["model"]),
        training=TrainingConfig(
            epochs=training_cfg.get("epochs", 20),
            learning_rate=training_cfg.get("learning_rate", 5e-4),
            weight_decay=training_cfg.get("weight_decay", 1e-4),
            grad_clip_norm=training_cfg.get("grad_clip_norm", 1.0),
            mixed_precision=training_cfg.get("mixed_precision", True),
            device=training_cfg.get("device", "cuda"),
            checkpoint_dir=_expand_path(training_cfg.get("checkpoint_dir", "artifacts/checkpoints")),
            log_every=training_cfg.get("log_every", 20),
            seed=training_cfg.get("seed", 7),
            loss_weights=LossWeights(**training_cfg.get("loss_weights", {})),
        ),
        export=ExportConfig(
            opset=export_cfg.get("opset", 17),
            output_path=_expand_path(export_cfg.get("output_path", "artifacts/export/rso_world_model.onnx")),
        ),
    )


def dataclass_to_dict(instance: Any) -> dict[str, Any]:
    if not hasattr(instance, "__dataclass_fields__"):
        return instance

    result: dict[str, Any] = {}
    for key in instance.__dataclass_fields__:
        value = getattr(instance, key)
        result[key] = dataclass_to_dict(value)
    return result
