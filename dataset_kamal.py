#!/usr/bin/env python3
"""
HarvestWindow — Model 2 Dataset class (Kamal, 5-class ripeness)

Same as before, now delegates transform-building to augmentation.py
instead of defining transforms inline, so training and eval always use
the exact same augmentation spec source (the YAML config), never two
copies that can drift out of sync.
"""

import random
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image

from augmentation import build_torchvision_train_transform, build_eval_transform

FOLDER_TO_CLASS = {
    "0Immature": "unripe",
    "1PartiallyRipe": "partially_ripe",
    "2FullyRipe": "ripe",
    "3OverRipe": "overripe",
    "4Decayed": "decayed",
}
CLASS_NAMES = ["unripe", "partially_ripe", "ripe", "overripe", "decayed"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def build_official_or_fallback_split(root: Path, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Try Kamal's own split files first. If they don't parse against files
    that actually exist on disk, fall back to a fresh stratified split
    and say so clearly — don't silently proceed on a broken assumption.
    """
    images_dir = root / "Images"
    split_dir = root / "Train_val_test_split"

    all_files = {}
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = images_dir / folder_name
        if not folder.exists():
            continue
        for img_path in list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
            all_files[img_path.name] = (img_path, class_name)

    print(f"Total images found on disk: {len(all_files)}")

    def try_load(fname):
        f = split_dir / fname
        if not f.exists():
            return None
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        matched = [l for l in lines if Path(l).name in all_files]
        if len(matched) < len(lines) * 0.5:
            print(f"  !! {fname}: only {len(matched)}/{len(lines)} entries matched files on disk — treating as unreliable")
            return None
        return matched

    train_list = try_load("Training.txt")
    val_list = try_load("Validation.txt")
    test_list = try_load("Testing.txt")

    if train_list and val_list and test_list:
        print("Using Kamal's official train/val/test split files.")
        splits = {"train": train_list, "val": val_list, "test": test_list}
        return {
            split: [(all_files[Path(f).name][0], all_files[Path(f).name][1]) for f in files]
            for split, files in splits.items()
        }

    print("!! Official split files missing or unreliable — building a fresh")
    print("   stratified split instead (by class, not officially validated).")
    random.seed(seed)
    by_class = {}
    for name, (path, cls) in all_files.items():
        by_class.setdefault(cls, []).append(path)

    splits = {"train": [], "val": [], "test": []}
    for cls, paths in by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        splits["val"].extend((p, cls) for p in paths[:n_val])
        splits["test"].extend((p, cls) for p in paths[n_val:n_val + n_test])
        splits["train"].extend((p, cls) for p in paths[n_val + n_test:])

    return splits


def compute_class_weights(train_items):
    counts = Counter(cls for _, cls in train_items)
    total = sum(counts.values())
    weights = torch.zeros(len(CLASS_NAMES))
    for cls, idx in CLASS_TO_IDX.items():
        c = counts.get(cls, 1)
        weights[idx] = total / (len(CLASS_NAMES) * c)
    print(f"Class counts (train): {dict(counts)}")
    print(f"Class weights: {weights.tolist()}")
    return weights


class KamalRipenessDataset(Dataset):
    def __init__(self, items, split: str, aug_spec: dict, image_size: int = 224):
        self.items = items
        if split == "train":
            self.transform = build_torchvision_train_transform(aug_spec, image_size)
        else:
            self.transform = build_eval_transform(image_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, class_name = self.items[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_IDX[class_name]
        return image, label
