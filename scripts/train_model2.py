#!/usr/bin/env python3
"""
HarvestWindow — Model 2 Training (ripeness classifier, 5 classes)

Fully config-driven — see configs/model2_config.yaml.
Backbone is a config parameter (mobilenet_v2 or efficientnet_b0) — same
training code handles either, so switching later is a one-line config
change, not a rewrite.

Checkpointing: saves best_model2.pt only when val accuracy improves.
Patience: stops training early if val accuracy hasn't improved in N
epochs (config: training.patience).

Usage:
    python scripts/train_model2.py --config configs/model2_config.yaml
"""

import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_kamal import (
    build_official_or_fallback_split,
    compute_class_weights,
    KamalRipenessDataset,
    CLASS_NAMES,
)
from models import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, opt_cfg: dict):
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Backbone: {cfg['model']['backbone']}")

    torch.manual_seed(cfg["training"].get("seed", 42))

    root = Path(cfg["data"]["root"])
    splits = build_official_or_fallback_split(root)
    class_weights = compute_class_weights(splits["train"]).to(device)

    image_size = cfg["training"].get("image_size", 224)
    aug_spec = cfg  # augmentation.py reads cfg["augmentation"] directly

    train_ds = KamalRipenessDataset(splits["train"], "train", aug_spec, image_size)
    val_ds = KamalRipenessDataset(splits["val"], "val", aug_spec, image_size)

    batch = cfg["training"]["batch"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    model = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, cfg["training"]["optimizer"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": cfg["model"]["backbone"],
                "num_classes": cfg["model"]["num_classes"],
                "class_names": CLASS_NAMES,
                "val_acc": val_acc,
                "epoch": epoch,
            }, output_dir / "best_model2.pt")
            print(f"  -> new best (val_acc={val_acc:.4f}), saved to {output_dir / 'best_model2.pt'}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping — no improvement for {patience} epochs.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best weights: {output_dir / 'best_model2.pt'}")
    print("\nNext: python scripts/evaluate_model2.py --config configs/eval_model2_config.yaml")


if __name__ == "__main__":
    main()
