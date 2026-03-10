#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained RSO world model to ONNX.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by training.")
    parser.add_argument("--output", default=None, help="Override ONNX output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rso_world_model.config import load_app_config
    from rso_world_model.export.onnx import export_to_onnx

    config = load_app_config(args.config)
    print(export_to_onnx(config, checkpoint_path=args.checkpoint, output_path=args.output))


if __name__ == "__main__":
    main()
