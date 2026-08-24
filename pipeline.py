#!/usr/bin/env python3
"""
HarvestWindow — Inference Pipeline

photo -> quality guard -> Model 1 (YOLO: detect + B-stage) -> crop primary
detection -> Model 2 (ripeness on crop) -> rule engine -> JSON

Output shape matches the API contract exactly (Level 6, system design doc):
status = "rejected" | "anomaly" | "classified".

Uses checkpoint_adapter.py for Model 2 so it transparently handles both
our own training script's checkpoint format and the friend-trained
externally-produced one (different backbone, different dict shape).

KNOWN SIMPLIFICATION, not a bug: this script reloads both models from disk
on every call. Fine for smoke-testing the pipeline logic right now. The
actual FastAPI backend must load both models ONCE at startup and reuse
them across requests — reloading per-request would make real latency
unusable. Flagging so this isn't silently carried into production code.

Usage:
    python pipeline.py --image photo.jpg \
        --model1 weights/model1_stage_detector_yolov8n_balanced.pt \
        --model2 weights/model2_ripeness_mobilenetv3large.pt \
        --device cuda
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from recommend import compute_recommendation, compute_confidence, compute_savings_idr, STAGES
from checkpoint_adapter import load_model2


def check_photo_quality(image_path: Path, blur_threshold: float = 100.0):
    """
    Classical CV quality guard (FR2) - blur via Laplacian variance,
    crude lighting check via mean brightness. No ML, deliberately cheap,
    runs before either model to fail fast on unusable input.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "poor_lighting"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, "blurry"

    mean_brightness = gray.mean()
    if mean_brightness < 30 or mean_brightness > 225:
        return False, "poor_lighting"

    return True, None


def run_model1(model, image_path: Path, device: str):
    """
    Single YOLO forward pass. Accepts an already-loaded YOLO object so
    callers can reuse it across requests without reloading from disk.

    Reads the stage label from the MODEL'S OWN embedded class names
    (result.names), not by assuming index position matches STAGES — a
    checkpoint trained outside this codebase (like the friend-trained
    yolo11s one) isn't guaranteed to use the same class-id ordering as
    our data.yaml, even if it was trained on the same classes.
    Validates the label is a real stage rather than silently trusting it.
    """
    results = model.predict(source=str(image_path), device=device, verbose=False)
    result = results[0]

    if len(result.boxes) == 0:
        return None  # no bunch detected

    confs = result.boxes.conf.cpu().numpy()
    best_idx = confs.argmax()  # primary detection = highest confidence
    box = result.boxes.xyxy[best_idx].cpu().numpy()
    cls_id = int(result.boxes.cls[best_idx].item())

    stage = result.names.get(cls_id)
    if stage not in STAGES:
        raise ValueError(
            f"Model 1 returned class '{stage}' (id {cls_id}), which isn't one "
            f"of {STAGES}. The checkpoint's embedded class names don't match "
            f"what the rule engine expects - do not silently proceed, this "
            f"would corrupt every downstream recommendation. Checkpoint "
            f"names dict: {result.names}"
        )

    return {
        "box": box,
        "stage": stage,
        "confidence": float(confs[best_idx]),
    }


def crop_image(image_path: Path, box, padding_frac: float = 0.1) -> Image.Image:
    """Crop to the primary detection, with a small padding margin so Model 2
    doesn't receive a razor-tight crop that cuts off part of the bunch."""
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * padding_frac, h * padding_frac
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(image.width, x2 + pad_x), min(image.height, y2 + pad_y)
    return image.crop((x1, y1, x2, y2))


def run_model2(model, class_names: list, crop: Image.Image, device: str):
    """Accepts an already-loaded model and class_names (from checkpoint_adapter)
    so callers can reuse both across requests without reloading from disk."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    return class_names[pred_idx], confidence


def classify(image_path: str, model1_path: str, model2_path: str, device: str = "cpu") -> dict:
    """CLI entry point — loads both models from disk then runs the pipeline.
    The FastAPI server must NOT call this; it loads models once at startup
    and calls run_model1/run_model2 directly with the already-loaded objects."""
    image_path = Path(image_path)

    model1 = YOLO(model1_path)
    model2, class_names, _ = load_model2(model2_path, device=device)

    passed, reason = check_photo_quality(image_path)
    if not passed:
        return {"status": "rejected", "reason": reason}

    detection = run_model1(model1, image_path, device)
    if detection is None:
        return {"status": "rejected", "reason": "no_bunch_detected"}

    crop = crop_image(image_path, detection["box"])
    ripeness, ripeness_conf = run_model2(model2, class_names, crop, device)
    stage = detection["stage"]

    rec = compute_recommendation(stage, ripeness)

    if rec.status == "anomaly":
        return {
            "status": "anomaly",
            "stage": stage,
            "ripeness": ripeness,
            "message": rec.reasoning,
        }

    confidence = compute_confidence(detection["confidence"], ripeness_conf)
    savings = compute_savings_idr(stage, ripeness)

    return {
        "status": "classified",
        "stage": stage,
        "ripeness": ripeness,
        "recommendation": rec.recommendation_text,
        "recheck_window_weeks": rec.recheck_window_weeks,
        "confidence": confidence,
        "reasoning": rec.reasoning,
        "estimated_savings_idr": savings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--model1", default="weights/model1_stage_detector_yolov8n_balanced.pt", type=str,
                         help="Path to Model 1 YOLO weights")
    parser.add_argument("--model2", default="weights/model2_ripeness_mobilenetv3large.pt", type=str,
                         help="Path to Model 2 classifier checkpoint")
    parser.add_argument("--device", default="cpu", type=str, help="cpu | cuda | cuda:0")
    args = parser.parse_args()

    result = classify(args.image, args.model1, args.model2, args.device)
    print(json.dumps(result, indent=2))
