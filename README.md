# BioScan AI

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=flat-square&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

---

## Overview

BioScan AI estimates a person's body weight from a single photograph, along with a calibrated 95% confidence interval that honestly communicates how certain the model is about its prediction. A user submits an image through a browser — either a desktop webcam or a phone camera opened via QR code — together with their height and age, and the system returns an estimated weight in kilograms within roughly two seconds. What makes the project technically interesting is the pipeline design: rather than feeding raw pixels into a convolutional network, the system uses MediaPipe Pose to extract 33 body landmarks from the image, converts those landmarks into perspective-invariant anthropometric ratios, and passes the resulting 10-element feature vector into a compact MLP. This design means the model trains on the ANSUR II dataset — a large, high-quality collection of real human body measurements — without needing any labelled photographs at all, because MediaPipe acts as the bridge between image space and ANSUR II's measurement space. Uncertainty is quantified through Monte Carlo Dropout: 50 stochastic forward passes through the network produce a distribution of outputs whose spread is a direct measure of how far the input lies from the training distribution.

---

## Architecture

```
  ┌───────────────────────────────────┐   ┌──────────────────────────────────┐
  │        Desktop Browser            │   │         Mobile Browser           │
  │                                   │   │                                  │
  │  webcam preview (getUserMedia)    │   │  rear camera (facingMode:ideal)  │
  │  height + age inputs              │   │  height + age inputs             │
  │  canvas capture + un-mirror       │   │  canvas capture → preview        │
  └─────────────────┬─────────────────┘   └─────────────────┬────────────────┘
                    │                                        │
                    │        POST /analyze                   │
                    │  multipart: image + height_cm + age    │
                    └──────────────────┬─────────────────────┘
                                       │
                                       ▼
              ┌────────────────────────────────────────────┐
              │            FastAPI  (api/main.py)           │
              │                                            │
              │  ┌─────────────────────────────────────┐  │
              │  │      POST /analyze                  │  │
              │  │                                     │  │
              │  │  ① Face blur (Haar cascade          │  │
              │  │              + top-20% fallback)    │  │
              │  │              │                      │  │
              │  │              ▼                      │  │
              │  │  ② Stage 1 — PoseDetector           │  │
              │  │    MediaPipe Pose → 33 landmarks    │  │
              │  │              │                      │  │
              │  │              ▼                      │  │
              │  │  ③ Stage 2 — compute_features       │  │
              │  │    landmarks → FeatureVector(10)    │  │
              │  │              │                      │  │
              │  │              ▼                      │  │
              │  │  ④ Stage 3 — WeightEstimator        │  │
              │  │    scaler → BioScanMLP              │  │
              │  │    50 MC Dropout forward passes     │  │
              │  │              │                      │  │
              │  │              ▼                      │  │
              │  │    AnalyzeResponse                  │  │
              │  │    (weight · CI · low_confidence)   │  │
              │  └─────────────────────────────────────┘  │
              └────────────────────────────────────────────┘
                                       │
                                       │  JSON response
                                       ▼
                        ┌──────────────────────────┐
                        │  Result display           │
                        │  weight (large) · CI bar  │
                        │  σ · time · warning?      │
                        └──────────────────────────┘
```

---

## How It Works

**Stage 1 — Pose Detection.** Every request begins with face blurring: an OpenCV Haar cascade locates the subject's face in the uploaded image and applies a Gaussian blur proportional to the detected face size, so that a close-up face and a distant one are both rendered unrecognisable. If the cascade finds nothing — a profile view, for instance — the top 20% of the frame is blurred as a safe fallback. The blurred image is then passed to `PoseDetector`, which resizes it to at most 1280 pixels on the longest side and runs it through MediaPipe Pose. MediaPipe returns 33 `NormalizedLandmark` objects covering the full skeleton; each carries x/y/z coordinates in the range [0, 1] relative to the image dimensions, plus a visibility confidence score. If no person is detected, or if any of the eight critical landmarks — shoulders, hips, wrists, ankles — score below 0.5 visibility, the request fails with a clear 422 error rather than producing a silently unreliable estimate.

**Stage 2 — Feature Engineering.** Raw landmark coordinates are not good direct inputs to an MLP because the same person photographed at different distances from the camera produces completely different coordinate values. `compute_features` converts the 33 landmarks into eight body-shape ratios that are stable regardless of camera distance: shoulder width, hip width, the shoulder-to-hip ratio, torso height, silhouette area normalised by torso height, left arm length, right arm length, and leg length. All distances are computed in 2D from the x/y coordinates only; the z value, which is MediaPipe's relative depth estimate rather than a metric distance, is excluded because it introduces noise when mixed with the normalised plane distances. The user's height in centimetres and their age are appended as the ninth and tenth elements of the feature vector — they come directly from the API request and are never estimated from the image.

**Stage 3 — Weight Estimation.** The 10-element feature vector is standardised using the same `StandardScaler` that was fitted on the training data and saved alongside the model checkpoint, then passed into `BioScanMLP`. Rather than running a single deterministic forward pass, the estimator runs 50 passes with dropout active on every one. Each pass uses a different random dropout mask, so the 50 outputs form a small distribution; their mean is the point estimate, and their standard deviation drives the 95% confidence interval (`mean ± 1.96 × std`). A high standard deviation is not just a wider error bar — it signals that the input feature vector lies in a region of the training distribution the model has seen infrequently, which is qualitatively different from ordinary measurement noise. When the standard deviation exceeds 5 kg, the response is flagged `low_confidence` and the frontend renders a clearly visible warning alongside the result.

---

## Key Engineering Decisions

**Monte Carlo Dropout with an unconditional subclass.** The system uses Monte Carlo Dropout to produce calibrated uncertainty estimates rather than deep ensembles, which would require training and storing multiple full models. The interesting implementation detail is `_MCDropout`, a two-line subclass of `nn.Dropout` that overrides `forward()` to always call `F.dropout(..., training=True)`, bypassing PyTorch's mode flag entirely. This means calling `model.eval()` — something any reasonable engineer might do reflexively before inference — cannot accidentally disable the uncertainty estimation. Correct behaviour is enforced by the architecture rather than by convention.

**Using ANSUR II without any labelled photographs.** Training a weight estimator from images normally requires a large dataset of photographs with known weights, which is difficult and expensive to collect. This project sidesteps that problem by training on ANSUR II, a US Army dataset of over 6,000 real body measurements paired with weights. MediaPipe acts as the bridge: it extracts the same shoulder widths, hip widths, and torso heights from images that ANSUR II measured with calipers, so the model learns from real human data without needing a single labelled photo. SMPL synthetic renders are used to validate the full image-to-prediction pipeline end-to-end.

**Perspective-invariant features through ratio engineering.** A 1.8-metre person photographed 2 metres away and the same person photographed 5 metres away produce dramatically different raw landmark values in MediaPipe's normalised coordinate space. Using ratios — shoulder width divided by hip width, for example — cancels perspective distortion because both the numerator and denominator scale identically with distance. This makes the feature vector genuinely stable across different shooting conditions, which is important for a system that will be used by real people in unpredictable environments.

**MLP over CNN.** Given that the input to the model is a 10-element vector of body-shape ratios rather than raw image pixels, a convolutional network's ability to detect local spatial patterns adds nothing. Feature engineering in Stage 2 already extracts all the spatial information the system needs, in a form that an MLP can consume directly and efficiently. The result is a model that is fast to train, cheap to run, and straightforward to reason about, without any loss in the quality of information fed to it.

**Self-documenting checkpoints and co-located scalers.** The `.pth` checkpoint saved after training stores not just the model weights but also the full config snapshot — `in_features`, `hidden_sizes`, `dropout_p`, `feature_columns` — and the best validation loss. Any checkpoint is therefore self-describing: you can inspect exactly what hyperparameters and feature set produced it without needing the original config file. The fitted scaler is always saved as `bioscan_scaler.pkl` in the same directory, and `WeightEstimator` verifies both files exist at startup rather than at request time. A missing scaler crashes the server with a clear message rather than serving silently wrong predictions to the first user.

---

## Getting Started

**Install dependencies.**

```bash
pip install -r requirements.txt
```

**Obtain the ANSUR II dataset.** The dataset is publicly available from Penn State's OpenLab. Download both the male and female measurement CSVs from [https://www.openlab.psu.edu/ansur2](https://www.openlab.psu.edu/ansur2) and place them in `training/data/raw/`. Rename or adjust `training/config.py`'s `ansur_csv_path` to match the filenames you downloaded. SMPL synthetic renders are optional for an initial training run; set `smpl_csv_path` to point to an empty CSV with the correct column headers if you want to train on ANSUR II alone.

**Run training.**

```bash
python training/train.py
```

Training writes `model/checkpoints/bioscan_model.pth` and `model/checkpoints/bioscan_scaler.pkl` when it finishes. To evaluate a checkpoint against a held-out test CSV:

```bash
python training/evaluate.py \
    --checkpoint model/checkpoints/bioscan_model.pth \
    --data training/data/raw/test_data.csv
```

The evaluation script exits with code 1 if MAE exceeds 6 kg, making it suitable as a CI/CD quality gate.

**Start the API server.**

```bash
uvicorn api.main:app --reload
```

The server starts on `http://localhost:8000`. The desktop UI is available at `http://localhost:8000/static/index.html`.

**Generate the QR code for mobile.** Mobile browsers require HTTPS for camera access. Use ngrok to expose the local server over a public HTTPS tunnel, then generate a QR code pointing to the mobile page:

```bash
ngrok http 8000
python frontend/qr_generator.py https://<your-ngrok-subdomain>.ngrok.io
```

The QR code is saved to `frontend/static/qr_mobile.png` and displayed automatically in the desktop UI's "Switch to Mobile Camera" section. Re-run `qr_generator.py` whenever ngrok assigns a new URL.

---

## Project Structure

```
visual-weight-estimator/
│
├── api/                          # FastAPI application
│   ├── main.py                   # App factory, lifespan, middleware, mounts
│   ├── schemas.py                # Pydantic request/response models
│   └── routes/
│       └── analyze.py            # POST /analyze — full pipeline orchestration
│
├── pipeline/                     # Three-stage inference pipeline
│   ├── pose_detector.py          # Stage 1 — MediaPipe landmark extraction
│   ├── feature_engineer.py       # Stage 2 — anthropometric feature computation
│   └── weight_estimator.py       # Stage 3 — MC Dropout inference wrapper
│
├── model/                        # Neural network and training artefacts
│   ├── mlp.py                    # BioScanMLP architecture + _MCDropout
│   ├── dataset.py                # PyTorch Dataset for training data
│   ├── trainer.py                # Training loop, early stopping, checkpointing
│   └── checkpoints/              # bioscan_model.pth + bioscan_scaler.pkl (gitignored)
│
├── training/                     # Training and evaluation scripts
│   ├── config.py                 # Single source of truth for all hyperparameters
│   ├── train.py                  # Entry point: load data → train → save checkpoint
│   ├── evaluate.py               # MAE / RMSE / CI calibration + quality gate
│   └── data/
│       ├── raw/                  # ANSUR II CSVs + SMPL renders (gitignored)
│       └── processed/            # Pre-extracted feature vectors (gitignored)
│
├── frontend/                     # Browser UI (served as static files at /static)
│   ├── index.html                # Desktop page — webcam capture + results
│   ├── mobile.html               # Mobile page — rear camera capture + results
│   ├── qr_generator.py           # CLI tool: generate QR code for mobile URL
│   └── static/
│       ├── js/
│       │   ├── webcam.js         # Desktop camera handler + result rendering
│       │   └── mobile.js         # Mobile camera handler + result rendering
│       └── css/                  # Shared stylesheets (reserved)
│
├── tests/
│   ├── test_pipeline.py          # Stage 1 + 2 unit tests (MediaPipe patched)
│   ├── test_model.py             # MLP shape + MC Dropout + WeightEstimator tests
│   └── test_api.py               # /analyze + /health integration tests
│
├── requirements.txt              # Pinned dependencies for Python 3.11
├── DECISIONS.md                  # Architectural decision log (32 entries)
├── ARCHITECTURE.md               # Full codebase walkthrough for new engineers
├── CLAUDE.md                     # Context file for AI-assisted development
└── .gitignore
```

---

## Running the Tests

```bash
pytest
```

To run a single file with verbose output:

```bash
pytest tests/test_pipeline.py -v
```
