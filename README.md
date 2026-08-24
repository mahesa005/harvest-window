# HarvestWindow

AI-powered oil palm (kelapa sawit) harvest-readiness assessment. Scan a bunch photo once, get an immediate recommendation — harvest now, or an estimated return-visit window if it's not ready yet. Built for AIC COMPFEST 18.

## What it does

A single photo goes through two models in sequence:

1. **Model 1** (YOLO detector) — finds the bunch in the photo and classifies its temporal stage (B1–B4, Black Bunch Census method: how many months out from harvest)
2. **Model 2** (ripeness classifier) — runs on the cropped detection, classifies visual ripeness (unripe / partially ripe / ripe / overripe / decayed)

A rule engine combines both outputs into one plain-language recommendation — including a **return-visit window** (e.g. "come back in 4–6 weeks") when the bunch isn't ready yet, computed directly from a single scan with no tracking or history required. Implausible stage/ripeness combinations (e.g. a bunch estimated ~3 months out already showing overripe) are flagged for manual verification instead of forcing a confident answer.

## Quick start

Requires Docker and Docker Compose.

```bash
git clone <this-repo-url>
cd harvestwindow
docker compose up --build
```

Open **http://localhost:8000** in a browser. Upload a bunch photo and the result appears in a few seconds.

To stop: `docker compose down`

## API

Single endpoint, synchronous, matches the competition's single-input/single-output constraint.

```
POST /api/classify
Content-Type: multipart/form-data

Fields:
  image     (file, required)                     — the bunch photo
  language  (string, optional, default "id")     — "id" | "en"
```

Example:
```bash
curl -X POST http://localhost:8000/api/classify \
  -F "image=@sample.jpg"
```

Response is always `200 OK`, one of three shapes depending on outcome:

**Rejected** (photo quality issue or no bunch detected):
```json
{"status": "rejected", "reason": "blurry"}
```

**Anomaly** (stage and ripeness readings don't plausibly match):
```json
{"status": "anomaly", "stage": "B2", "ripeness": "overripe", "message": "..."}
```

**Classified** (normal result):
```json
{
  "status": "classified",
  "stage": "B2",
  "ripeness": "unripe",
  "recommendation": "Come back in about 4–6 weeks",
  "recheck_window_weeks": [4, 6],
  "confidence": 87,
  "reasoning": "Bunch stage indicates roughly 2 months out, not yet ready",
  "estimated_savings_idr": 37500
}
```

## Repository structure

```
├── server.py               FastAPI app — loads both models once at startup, exposes POST /api/classify
├── pipeline.py             Core inference: quality guard -> Model 1 -> crop -> Model 2 -> rule engine
├── recommend.py            Rule engine — the locked B-stage x ripeness recommendation table
├── checkpoint_adapter.py   Loads Model 2 checkpoints regardless of training-script origin
├── models.py               Model 2 backbone factory (mobilenet_v2 / efficientnet_b0 / mobilenet_v3_large)
├── static/index.html       Web frontend
├── scripts/
│   ├── train_model1.py / train_model2.py       Training scripts (config-driven)
│   ├── evaluate_model1.py / evaluate_model2.py Evaluation scripts (mAP / confusion matrix)
│   ├── augmentation.py     Shared augmentation spec -> YOLO kwargs + torchvision transforms
│   └── dataset_kamal.py    Model 2 dataset class + train/val/test split logic
├── configs/                YAML configs for training and evaluation, per model
└── weights/                Trained model checkpoints
```

## Model performance (test set)

| Model | Metric | Value |
|---|---|---|
| Model 1 (YOLO, B1-B4 stage) | mAP50 / mAP50-95 | 0.527 / 0.250 |
| Model 2 (ripeness, 5-class) | Test accuracy | 96.5% |

Full per-class breakdowns available in `evals/model1/` and `evals/model2/` after running the evaluation scripts.

**Framing note:** Model 1's B-stage output represents agreement with expert Black Bunch Census (BBC) classification, not a validated prediction of actual harvest date — BBC itself carries known real-world error against realized production (independently reported MAPE up to ~76% for bunch-count estimation). The product's claim is that it automates BBC-standard assessment for users without access to a trained surveyor, not that it exceeds the accuracy of the method it replicates.

## Training your own weights

```bash
pip install -r requirements.txt

python scripts/train_model1.py --config configs/model1_config.yaml
python scripts/evaluate_model1.py --config configs/eval_model1_config.yaml

python scripts/train_model2.py --config configs/model2_config.yaml
python scripts/evaluate_model2.py --config configs/eval_model2_config.yaml
```

Fast smoke-test configs (1 epoch each) are available at `configs/test_model1_config.yaml` / `configs/test_model2_config.yaml` for a quick sanity check before committing to a full run.

## Scope note (qualifying round)

Per competition rules, this MVP is single-input/single-output, fully synchronous, with no persistent storage or background jobs. The product's full vision — per-tree tracking across repeat visits and block-level revisit scheduling — is documented in the proposal but intentionally not part of this build, since it requires persistence the qualifying round doesn't allow.
