# HarvestWindow

AI-powered oil palm (kelapa sawit) harvest-readiness assessment. Scan a bunch photo once, get an immediate recommendation — harvest now, or an estimated return-visit window if it's not ready yet. Built for AIC COMPFEST 18.

## Architecture

A single photo passes through two models in sequence, never in parallel:

```
Photo
  └─► Quality guard (blur + lighting check)
        └─► Model 1 — YOLO: detects the bunch AND classifies B-stage in one pass
              └─► Crop (primary detection box, padded)
                    └─► Model 2 — ripeness classifier on the crop only
                                  (never sees the original photo)
                          └─► Rule engine → recommendation JSON
```

**Model 1** (YOLO) locates the bunch and classifies its temporal stage (B1–B4, Black Bunch Census method). B-stage indicates how many weeks out from typical harvest the bunch is — B1 is nearest, B4 is ~4 months out.

**Model 2** (MobileNetV3-Large) classifies the visual ripeness of the cropped detection into five classes: unripe / partially ripe / ripe / overripe / decayed.

The **rule engine** is a deterministic 4×5 lookup table (stage × ripeness). Implausible combinations (e.g. B3 stage + overripe color) are returned as `anomaly` rather than forcing a confident wrong answer. Harvest-ready combinations return `harvest_now` or `harvest_immediately`. Not-ready combinations return a recheck window in weeks.

## Quick start

Requires Docker and Docker Compose. Model weights are committed to the repo and baked into the image — no separate download needed to run inference.

```bash
git clone https://github.com/mahesa005/harvest-window.git
cd harvest-window
docker compose up --build
```

Open **http://localhost:8000** in a browser. Upload a bunch photo — result appears in a few seconds.

To stop: `docker compose down`

**CPU-only machines:** the server defaults to CUDA. Override with:

```bash
DEVICE=cpu docker compose up --build
```

## API

Single endpoint, synchronous, `multipart/form-data`.

```
POST /api/classify
```

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | yes | Bunch photo (JPEG/PNG) |
| `language` | string | no | Default `"id"` — accepted but reserved for future use |

**Example:**
```bash
curl -X POST http://localhost:8000/api/classify \
  -F "image=@photo.jpg"
```

Response is always `200 OK`. Three possible shapes:

**`rejected`** — photo quality failed or no bunch detected:
```json
{
  "status": "rejected",
  "reason": "blurry"
}
```
`reason` values: `"blurry"` · `"poor_lighting"` · `"no_bunch_detected"`

**`anomaly`** — stage and ripeness readings are implausible together:
```json
{
  "status": "anomaly",
  "stage": "B3",
  "ripeness": "overripe",
  "message": "Overripe at a ~3-month-out stage — implausible, verify manually"
}
```

**`classified`** — normal result:
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

`recheck_window_weeks` is `null` for harvest-now/immediately outcomes.
`confidence` is an integer percentage (average of Model 1 and Model 2 softmax confidence).

## Datasets

The `data/` directory is gitignored. Download and extract both datasets before training.

### Dataset 1 — SawitMVC (Model 1)

**Download:** https://zenodo.org/records/20336323/files/SawitMVC.zip?download=1 (2.4 GB)

**License:** CC BY-NC 4.0 (non-commercial)

Extract so the directory layout matches exactly:

```
data/
└── SawitMVC/
    ├── data.yaml           ← referenced by configs/model1_config.yaml
    ├── data_balanced.yaml  ← referenced by configs/model1_config_balanced.yaml
    ├── images/
    ├── labels/
    ├── data/
    └── json/
```

**Citation:**
> Indriani, F., Saputro, S. W., Muttaqin, M. Z., Rahmi, A., Saragih, T. H., Budianoor, R., Hartoni, & Kartini, D., Said, N. (2026). SawitMVC: A Multi-View Oil Palm Fruit Bunch Dataset for Detection and Counting (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.20336323

---

### Dataset 2 — Ordinal Dataset for Ripeness Level Classification (Model 2)

**Download:** https://data.mendeley.com/datasets/424y96m6sw/1

**Citation:** see the "Cite this dataset" section on the Mendeley page above (DOI: 10.17632/424y96m6sw.1)

Extract so the directory layout matches exactly (the nested folder name is part of the dataset's own structure, not a mistake):

```
data/
└── An Ordinal Dataset for Ripeness Level Classification in Oil Palm Fruit Quality Grading/
    └── An Ordinal Dataset for Ripeness Level Classification in Oil Palm Fruit Quality Grading/
        └── Dataset/
            └── Dataset/            ← this path is what configs/model2_config.yaml points to
                ├── Images/
                │   ├── 0Immature/
                │   ├── 1PartiallyRipe/
                │   ├── 2FullyRipe/
                │   ├── 3OverRipe/
                │   └── 4Decayed/
                └── Train_val_test_split/
                    ├── Training.txt
                    ├── Validation.txt
                    └── Testing.txt
```

If `Train_val_test_split/` files are missing or fewer than 50% of entries match files on disk, `dataset_kamal.py` will automatically fall back to a fresh stratified split (seed 42) and log a warning — it will not silently proceed on a broken split.

## Training your own weights

**Prerequisites:** Python 3.12, `pip install -r requirements.txt`. Both datasets extracted as shown above.

**Config note:** `configs/model1_config.yaml`, `configs/model1_config_balanced.yaml`, and `configs/model2_config.yaml` contain hardcoded absolute Windows paths in their `output.project` / `output.dir` fields. Update these to your own paths before running training. The test and eval configs use relative paths and do not need changes.

### Model 1 (YOLO detector, B1–B4)

```bash
# Full training run (~50 epochs, early stop at patience=15)
python scripts/train_model1.py --config configs/model1_config.yaml

# Balanced variant (oversampled training set, separate run dir)
python scripts/train_model1.py --config configs/model1_config_balanced.yaml

# Smoke test (1 epoch, imgsz=320, relative output path — no path edit needed)
python scripts/train_model1.py --config configs/test_model1_config.yaml
```

Best weights are saved to `<output.project>/<output.name>/weights/best.pt` by ultralytics automatically.

Evaluate on the test split:

```bash
python scripts/evaluate_model1.py --config configs/eval_model1_config.yaml
```

Outputs: mAP50, mAP50-95, per-class breakdown, confusion matrix, PR curves — written to `evals/model1/eval/`.

---

### Model 2 (ripeness classifier, 5-class)

```bash
# Full training run (~30 epochs, early stop at patience=7)
python scripts/train_model2.py --config configs/model2_config.yaml

# Smoke test (1 epoch)
python scripts/train_model2.py --config configs/test_model2_config.yaml
```

Best checkpoint saved to `<output.dir>/best_model2.pt`.

Evaluate on the test split:

```bash
python scripts/evaluate_model2.py --config configs/eval_model2_config.yaml
```

Outputs: classification report and confusion matrix PNG — written to `evals/model2/`.

---

### Pipeline smoke test (no training required)

```bash
python pipeline.py \
  --image path/to/photo.jpg \
  --model1 weights/model1_stage_detector_yolov8n_balanced.pt \
  --model2 weights/model2_ripeness_mobilenetv3large.pt \
  --device cpu
```

## Repository structure

```
.
├── server.py               FastAPI app — loads both models once at startup
├── pipeline.py             End-to-end inference (also a CLI entry point)
├── recommend.py            Rule engine — the locked B-stage × ripeness table
├── checkpoint_adapter.py   Loads Model 2 checkpoints in either checkpoint format
├── models.py               Model 2 backbone factory (mobilenet_v2 / efficientnet_b0 / mobilenet_v3_large)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── configs/
│   ├── model1_config.yaml              Model 1 full training
│   ├── model1_config_balanced.yaml     Model 1 balanced retrain
│   ├── model2_config.yaml              Model 2 full training
│   ├── eval_model1_config.yaml         Model 1 evaluation
│   ├── eval_model2_config.yaml         Model 2 evaluation
│   ├── test_model1_config.yaml         Model 1 smoke test (1 epoch)
│   └── test_model2_config.yaml         Model 2 smoke test (1 epoch)
├── evals/
│   └── model2/
│       ├── classification_report.txt
│       └── confusion_matrix.png
├── pretrained/
│   ├── yolov8n.pt          Base YOLO weights for training Model 1
│   └── yolo26n.pt
├── runs/
│   └── model1/
│       └── b1_b4_detector_balanced/    Training artifacts (curves, confusion matrix, weights/)
├── scripts/
│   ├── train_model1.py
│   ├── train_model2.py
│   ├── evaluate_model1.py
│   ├── evaluate_model2.py
│   ├── augmentation.py     Shared augmentation spec → YOLO kwargs + torchvision transforms
│   └── dataset_kamal.py    Model 2 dataset class + official/fallback split logic
├── static/
│   └── index.html          Web frontend
└── weights/
    ├── model1_stage_detector_yolov8n_balanced.pt   ← default Model 1 weights
    ├── model1_stage_detector_yolo11s.pt
    ├── model1_test_metrics.json
    ├── model2_ripeness_mobilenetv3large.pt          ← default Model 2 weights
    └── model2_test_metrics.json
```

## Model performance (test set)

| Model | Metric | Value |
|---|---|---|
| Model 1 (YOLO, B1–B4 stage, balanced retrain) | mAP50 | 0.527 |
| Model 1 (YOLO, B1–B4 stage, balanced retrain) | mAP50-95 | 0.250 |
| Model 2 (MobileNetV3-Large, 5-class ripeness) | Test accuracy | 96.5% |

**Model 1 per-class mAP50:** B1 0.747 · B2 0.419 · B3 0.599 · B4 0.342 (source: `evals/model1/eval/`)

**Model 2 per-class recall:** unripe 98.3% · partially ripe 93.9% · ripe 96.9% · overripe 97.1% · decayed 100% (source: `evals/model2/classification_report.txt`, 718 test samples)

**Framing note:** Model 1's B-stage output represents agreement with expert Black Bunch Census (BBC) classification, not a validated prediction of actual harvest date — BBC itself carries known real-world error against realized production. The product's claim is that it automates BBC-standard assessment for users without access to a trained surveyor, not that it exceeds the accuracy of the method it replicates.

## Scope note (qualifying round)

Per competition rules, this MVP is single-input/single-output, fully synchronous, with no persistent storage or background jobs. The product's full vision — per-tree tracking across repeat visits and block-level revisit scheduling — is documented in the proposal but intentionally not part of this build, since it requires persistence the qualifying round doesn't allow.
