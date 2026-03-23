# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

BioScan AI — a Python-based visual weight estimator. A user submits an image plus their height (cm) and age via a REST API; the system returns an estimated body weight with a confidence interval.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --reload

# Train the MLP
python training/train.py

# Evaluate a checkpoint
python training/evaluate.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_pipeline.py -v
```

## Architecture

The system is a three-stage pipeline executed on every `POST /analyze` request:

1. **Stage 1 — Pose detection** (`pipeline/pose_detector.py`): MediaPipe Pose extracts 33 body landmarks (x, y, z) from the uploaded image. Face blurring (OpenCV Haar cascade, with landmark-based upper-head fallback) is applied before any processing.

2. **Stage 2 — Feature engineering** (`pipeline/feature_engineer.py`): Derives 10–15 anthropometric features from the landmarks — shoulder width, shoulder-to-hip ratio, silhouette area (normalised by torso height ratio to reduce perspective distortion), limb ratios, etc. User-supplied `height_cm` and `age` are appended here; they are not estimated from the image.

3. **Stage 3 — Weight estimation** (`pipeline/weight_estimator.py`): Loads the trained MLP checkpoint and runs Monte Carlo Dropout inference (multiple stochastic forward passes with dropout active) to produce `estimated_weight_kg` and a `confidence_interval`.

### Model (`model/`)

- `mlp.py` — PyTorch MLP with dropout layers that remain active at inference for Monte Carlo uncertainty estimation.
- `dataset.py` — handles both ANSUR II tabular data (measurements → weight) and SMPL synthetic image-derived features.
- `trainer.py` — training loop with checkpoint saving.
- Scaler (StandardScaler or similar) must be saved alongside the `.pth` checkpoint and loaded identically at inference.

### Training (`training/`)

- Primary dataset: **ANSUR II** — real anthropometric measurements mapped to body weight.
- Secondary dataset: **SMPL synthetic renders** — generated body images with known parameters, used for end-to-end pipeline pretraining.
- All hyperparameters, dataset paths, feature list, and MC dropout sample count live in `training/config.py`.
- Raw data goes in `training/data/raw/` (gitignored); pre-extracted feature vectors go in `training/data/processed/`.

### API (`api/`)

- `main.py` — FastAPI app with lifespan context (model loaded once at startup).
- `routes/analyze.py` — single `POST /analyze` endpoint; accepts multipart `image`, form fields `height_cm` and `age`.
- `schemas.py` — Pydantic models; response includes `estimated_weight_kg`, `confidence_interval`, `input_height_cm` (passthrough of user input, not computed), and `processing_time_ms`.

### Frontend (`frontend/`)

- `index.html` + `static/js/webcam.js` — desktop browser UI using `MediaDevices.getUserMedia`.
- `mobile.html` + `static/js/mobile.js` — mobile capture page opened via QR code; requests rear camera, POSTs to `/analyze`.
- `qr_generator.py` — generates a QR code pointing to the ngrok/production host's `/mobile` route.
- Mobile camera requires **HTTPS**; use ngrok (`ngrok http 8000`) for local development.

## Key Design Decisions

> Full rationale for each decision is documented in `DECISIONS.md`.

- `input_height_cm` in the response is a passthrough of user-supplied height — it is never estimated from the image.
- Monte Carlo Dropout is the chosen uncertainty method — dropout must be active (`model.train()` mode, or a custom `eval_with_dropout` call) during inference passes.
- Silhouette area is normalised by the torso height ratio (landmark-derived) before being added to the feature vector to reduce sensitivity to subject-to-camera distance.
- The feature scaler fitted on training data must be serialised (e.g., `scaler.pkl`) and loaded at API startup alongside the model checkpoint.
