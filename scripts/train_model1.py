#!/usr/bin/env python3
"""
HarvestWindow — Model 1 Training (YOLO detector, B1-B4 classes)

Fully config-driven — see configs/model1_config.yaml.

Checkpointing/patience: ultralytics handles this natively (save=True saves
best.pt + last.pt every run, patience=N triggers early stopping) — no
custom code needed here, just wired through from the config.

Metric: mAP (mAP50 and mAP50-95), computed automatically by ultralytics
during and after training. Per-class breakdown included.

Usage:
    python train_model1.py --config configs/model1_config.yaml
"""

import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

from augmentation import build_yolo_augmentation_kwargs


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_yaml = Path(cfg["data"]["data_yaml"])
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found — check data.data_yaml in your config")

    print(f"Config: {args.config}")
    print(f"Model: {cfg['model']['variant']} | Data: {data_yaml}")
    print(f"Epochs: {cfg['training']['epochs']} | Patience: {cfg['training']['patience']}")

    model = YOLO(cfg["model"]["variant"])
    aug_kwargs = build_yolo_augmentation_kwargs(cfg)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["imgsz"],
        batch=cfg["training"]["batch"],
        patience=cfg["training"]["patience"],
        seed=cfg["training"].get("seed", 42),
        project=cfg["output"]["project"],
        name=cfg["output"]["name"],
        save=True,      # writes best.pt + last.pt automatically
        plots=True,     # confusion matrix, PR curves, etc. auto-generated
        **aug_kwargs,
    )

    optimizer = cfg["training"].get("optimizer", "auto")
    if optimizer != "auto":
        train_kwargs["optimizer"] = optimizer
        train_kwargs["lr0"] = cfg["training"].get("lr0", 0.01)

    train_kwargs["workers"] = cfg["training"].get("workers", 0)

    results = model.train(**train_kwargs)

    best_path = Path(cfg["output"]["project"]) / cfg["output"]["name"] / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_path}")
    print(f"Plots and metrics saved in: {Path(cfg['output']['project']) / cfg['output']['name']}")
    print("\nNext: python scripts/evaluate_model1.py --config configs/eval_model1_config.yaml")


if __name__ == "__main__":
    main()
