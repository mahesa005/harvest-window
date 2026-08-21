#!/usr/bin/env python3
"""
HarvestWindow — Model 1 Evaluation (mAP)

ultralytics computes mAP50 and mAP50-95 natively via model.val() — no
custom metric code needed. This script just wires the config through
and prints/saves the results in one place.

Usage:
    python evaluate_model1.py --config configs/eval_model1_config.yaml
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights_path = Path(cfg["model"]["weights"])
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found — train Model 1 first")

    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating on split: {cfg['data']['split']}")
    metrics = model.val(
        data=cfg["data"]["data_yaml"],
        split=cfg["data"]["split"],
        project=str(output_dir),
        name="eval",
        plots=True,   # writes confusion matrix + PR curves for the eval run
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("\nPer-class mAP50-95:")
    for i, class_name in metrics.names.items():
        try:
            print(f"  {class_name}: {metrics.box.maps[i]:.4f}")
        except (IndexError, TypeError):
            pass

    print(f"\nFull plots + confusion matrix saved in: {output_dir / 'eval'}")


if __name__ == "__main__":
    main()
