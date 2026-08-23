#!/usr/bin/env python3
"""
HarvestWindow — Model 2 Checkpoint Adapter

The friend-trained checkpoint (weights/model2_ripeness_mobilenetv3large.pt)
has a different shape than what train_model2.py produces:

    Friend's format:  {"state_dict": ..., "classes": [...]}
    Our format:       {"model_state_dict": ..., "backbone": ..., "num_classes": ...,
                        "class_names": ..., "val_acc": ..., "epoch": ...}

This adapter loads either format transparently, so pipeline.py and
evaluate_model2.py don't need to care which one they're given.

Class order verification: the friend's checkpoint lists classes as
['0Immature', '1PartiallyRipe', '2FullyRipe', '3OverRipe', '4Decayed']
— same index order as our CLASS_NAMES (unripe/partially_ripe/ripe/
overripe/decayed), just with the raw folder-name prefixes still attached.
This adapter maps between them and ASSERTS the order matches rather than
assuming it silently.
"""

from pathlib import Path
import torch

from models import build_model

CLASS_NAMES = ["unripe", "partially_ripe", "ripe", "overripe", "decayed"]

# Maps the friend's raw folder-style class strings to our canonical names,
# used only to verify order — not to relabel anything at inference time.
FRIEND_LABEL_TO_CANONICAL = {
    "0Immature": "unripe",
    "1PartiallyRipe": "partially_ripe",
    "2FullyRipe": "ripe",
    "3OverRipe": "overripe",
    "4Decayed": "decayed",
}


def load_model2(checkpoint_path: str, device: str = "cpu", default_backbone: str = "mobilenet_v3_large"):
    """
    Loads a Model 2 checkpoint in EITHER format and returns
    (model, class_names, metadata_dict) — metadata_dict has whatever
    val_acc/epoch info is available, empty if the checkpoint doesn't
    carry it (the friend's format doesn't).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        # Our own training script's format
        backbone = checkpoint["backbone"]
        num_classes = checkpoint["num_classes"]
        class_names = checkpoint["class_names"]
        state_dict = checkpoint["model_state_dict"]
        metadata = {"val_acc": checkpoint.get("val_acc"), "epoch": checkpoint.get("epoch")}

    elif "state_dict" in checkpoint:
        # Friend's format — infer backbone (only mobilenet_v3_large seen so far),
        # verify class order rather than assume it
        friend_classes = checkpoint.get("classes", [])
        expected_order = list(FRIEND_LABEL_TO_CANONICAL.keys())
        if friend_classes != expected_order:
            raise ValueError(
                f"Checkpoint class order {friend_classes} does not match the "
                f"expected order {expected_order}. Do not silently proceed — "
                f"the rule engine and API contract assume this exact index "
                f"mapping. Fix the mismatch before running inference."
            )

        backbone = default_backbone
        num_classes = len(friend_classes)
        class_names = [FRIEND_LABEL_TO_CANONICAL[c] for c in friend_classes]
        state_dict = checkpoint["state_dict"]
        metadata = {"val_acc": None, "epoch": None}  # not present in this format

    else:
        raise ValueError(
            f"Unrecognized checkpoint format at {checkpoint_path} — "
            f"expected either 'model_state_dict' or 'state_dict' key, "
            f"found keys: {list(checkpoint.keys())}"
        )

    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Resolved class_names {class_names} != canonical {CLASS_NAMES} — "
            f"mismatch would silently scramble every ripeness prediction."
        )

    model = build_model(backbone, num_classes, pretrained=False).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("!! strict=True state_dict load failed. This usually means the")
        print("   assumed architecture (backbone + classifier head shape) doesn't")
        print("   exactly match how the checkpoint was actually trained.")
        print(f"   Full error:\n{e}")
        raise

    model.eval()
    return model, class_names, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    args = parser.parse_args()

    model, class_names, metadata = load_model2(args.checkpoint)
    print(f"Loaded successfully.")
    print(f"Class names (verified order): {class_names}")
    print(f"Metadata: {metadata}")
    print(f"Model type: {type(model).__name__}")
