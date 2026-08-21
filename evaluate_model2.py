#!/usr/bin/env python3
"""
HarvestWindow — Model 2 Evaluation (confusion matrix, per-class P/R/F1)

Usage:
    python evaluate_model2.py --config configs/eval_model2_config.yaml
"""

import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

from dataset_kamal import build_official_or_fallback_split, KamalRipenessDataset, CLASS_NAMES
from models import build_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Model 2 — Confusion Matrix (test set)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Confusion matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = Path(cfg["model"]["weights"])
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found — train Model 2 first")

    checkpoint = torch.load(weights_path, map_location=device)
    backbone = checkpoint.get("backbone", cfg["model"]["backbone"])
    num_classes = checkpoint.get("num_classes", cfg["model"]["num_classes"])
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc={checkpoint.get('val_acc', '?')}")

    model = build_model(backbone, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    root = Path(cfg["data"]["root"])
    splits = build_official_or_fallback_split(root)
    test_items = splits[cfg["data"]["split"]]

    # Empty augmentation spec for eval — build_eval_transform ignores it anyway
    test_ds = KamalRipenessDataset(test_items, "test", {"augmentation": {}}, image_size=224)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (test set)")
    print("=" * 50)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    print(report)
    (output_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f}")
    print(f"\nWatch overripe/decayed specifically — smallest classes, highest")
    print(f"real-world stakes (mill-rejection losses tie directly to these).")


if __name__ == "__main__":
    main()
