#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RSO maneuver intent world model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rso_world_model.config import load_app_config
    from rso_world_model.training.train import train_world_model

    checkpoint = train_world_model(load_app_config(args.config))
    print(checkpoint)


if __name__ == "__main__":
    main()
