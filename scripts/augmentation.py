#!/usr/bin/env python3
"""
HarvestWindow — Shared Augmentation Module

One augmentation SPEC (a plain dict, usually loaded from YAML), two
translations:
  - build_yolo_augmentation_kwargs()  -> kwargs for ultralytics model.train()
  - build_torchvision_transform()     -> torchvision.transforms.Compose

Only ever applied to the TRAINING split. Validation/test always get the
plain resize+normalize-only transform (build_eval_transform) — augmenting
eval data would make your metrics measure something other than real
performance.

Known limitation, documented rather than hidden: ultralytics' built-in
augmentation set (HSV jitter, scale, translate, flip, mosaic) has no
first-class "grayscale" op the way torchvision does. Model 1's "lighting
variation" comes from HSV jitter instead — covers similar ground, not
identical. Model 2 gets true grayscale via torchvision.
"""

from torchvision import transforms


def build_yolo_augmentation_kwargs(spec: dict) -> dict:
    """
    Translate the shared spec into ultralytics train() kwargs.
    Only pulls the keys relevant to YOLO; ignores torchvision-only keys
    (grayscale_prob) since there's no direct equivalent.
    """
    aug = spec.get("augmentation", {})
    return {
        "hsv_h": aug.get("hue", 0.015),
        "hsv_s": aug.get("saturation", 0.5),
        "hsv_v": aug.get("brightness", 0.4),
        "scale": aug.get("crop_zoom_jitter", 0.5),
        "translate": aug.get("translate", 0.1),
        "fliplr": aug.get("hflip_prob", 0.5),
        "flipud": aug.get("vflip_prob", 0.0),
        "degrees": aug.get("rotation_degrees", 0.0),
    }


def build_torchvision_train_transform(spec: dict, image_size: int = 224) -> transforms.Compose:
    """Training-only transform — augmentation applied."""
    aug = spec.get("augmentation", {})

    crop_min = aug.get("crop_scale_min", 0.7)
    crop_max = aug.get("crop_scale_max", 1.0)
    brightness = aug.get("brightness", 0.3)
    contrast = aug.get("contrast", 0.3)
    saturation = aug.get("saturation", 0.2)
    hue = aug.get("hue", 0.02)
    grayscale_prob = aug.get("grayscale_prob", 0.1)
    hflip_prob = aug.get("hflip_prob", 0.5)
    rotation_degrees = aug.get("rotation_degrees", 10)

    return transforms.Compose([
        # Crop-margin/zoom jitter — targets the detection-crop framing gap
        # (Model 2 sees auto-cropped regions from Model 1 at inference,
        # not deliberately composed photos like its training data)
        transforms.RandomResizedCrop(image_size, scale=(crop_min, crop_max), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(hflip_prob),
        transforms.RandomRotation(rotation_degrees),
        # Lighting/color jitter — targets the on-tree/off-tree domain gap
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.RandomGrayscale(p=grayscale_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    """No augmentation — used for val/test and real inference."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    # Quick smoke test with a default spec
    example_spec = {
        "augmentation": {
            "brightness": 0.3, "contrast": 0.3, "saturation": 0.2, "hue": 0.02,
            "grayscale_prob": 0.1, "crop_scale_min": 0.7, "crop_scale_max": 1.0,
            "hflip_prob": 0.5, "rotation_degrees": 10,
        }
    }
    print("YOLO kwargs:", build_yolo_augmentation_kwargs(example_spec))
    print("Torchvision train transform:", build_torchvision_train_transform(example_spec))
    print("Eval transform:", build_eval_transform())
