#!/usr/bin/env python3
"""
HarvestWindow — FastAPI inference server

Models are loaded ONCE at startup via lifespan and reused across all requests.
See pipeline.py's KNOWN SIMPLIFICATION note — this server is the fix for that.

Usage:
    python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from checkpoint_adapter import load_model2
from pipeline import check_photo_quality, run_model1, run_model2, crop_image
from recommend import compute_recommendation, compute_confidence, compute_savings_idr

MODEL1_PATH = os.getenv("MODEL1", "weights/model1_stage_detector_yolov8n_balanced.pt")
MODEL2_PATH = os.getenv("MODEL2", "weights/model2_ripeness_mobilenetv3large.pt")
DEVICE = os.getenv("DEVICE", "cuda")

# Module-level handles populated at startup
_model1 = None
_model2 = None
_model2_class_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model1, _model2, _model2_class_names
    print(f"[startup] Loading Model 1 ({MODEL1_PATH}) on {DEVICE}...")
    _model1 = YOLO(MODEL1_PATH)
    _model1.to(DEVICE)
    print(f"[startup] Loading Model 2 ({MODEL2_PATH}) on {DEVICE}...")
    _model2, _model2_class_names, meta = load_model2(MODEL2_PATH, device=DEVICE)
    print(f"[startup] Models ready. Model2 class_names={_model2_class_names} val_acc={meta.get('val_acc')}")
    yield
    print("[shutdown] Bye.")


app = FastAPI(title="HarvestWindow", lifespan=lifespan)


@app.post("/api/classify")
async def classify_endpoint(
    image: UploadFile = File(...),
    language: str = Form(default="id"),
):
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await image.read())
        tmp_path = Path(f.name)

    try:
        result = _run_pipeline(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return JSONResponse(result)


def _run_pipeline(image_path: Path) -> dict:
    passed, reason = check_photo_quality(image_path)
    if not passed:
        return {"status": "rejected", "reason": reason}

    detection = run_model1(_model1, image_path, DEVICE)
    if detection is None:
        return {"status": "rejected", "reason": "no_bunch_detected"}

    crop = crop_image(image_path, detection["box"])
    ripeness, ripeness_conf = run_model2(_model2, _model2_class_names, crop, DEVICE)
    stage = detection["stage"]

    rec = compute_recommendation(stage, ripeness)

    if rec.status == "anomaly":
        return {
            "status": "anomaly",
            "stage": stage,
            "ripeness": ripeness,
            "message": rec.reasoning,
        }

    return {
        "status": "classified",
        "stage": stage,
        "ripeness": ripeness,
        "recommendation": rec.recommendation_text,
        "recheck_window_weeks": rec.recheck_window_weeks,
        "confidence": compute_confidence(detection["confidence"], ripeness_conf),
        "reasoning": rec.reasoning,
        "estimated_savings_idr": compute_savings_idr(stage, ripeness),
    }


# Serve static/ as root — must come last so API routes aren't shadowed
app.mount("/", StaticFiles(directory="static", html=True), name="static")
