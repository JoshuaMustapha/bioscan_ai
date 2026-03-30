"""POST /analyze route.

Receives a multipart form containing an image, height_cm, and age.
Applies face blurring, then drives the three-stage pipeline
(PoseDetector → compute_features → WeightEstimator) and returns an
AnalyzeResponse with the weight estimate and a 95% confidence interval.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.schemas import AnalyzeResponse
from pipeline.feature_engineer import compute_features

_log = logging.getLogger(__name__)

router = APIRouter()

# Loaded once at module import.  If the XML file is missing from the OpenCV
# data directory the cascade will be empty and detectMultiScale will return
# an empty list, causing the code to fall through to the top-20% fallback
# without raising an error.
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_HEIGHT_MIN: float = 50.0
_HEIGHT_MAX: float = 300.0
_AGE_MIN: int = 5
_AGE_MAX: int = 120
_VALID_GENDERS: frozenset[int] = frozenset({0, 1})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _blur_face(bgr: np.ndarray) -> np.ndarray:
    """Blur all detected faces in a BGR image.

    Primary strategy: detect faces with the Haar cascade and apply a
    Gaussian blur whose sigma scales with the face size.

    Fallback: if the cascade detects no faces, blur the top 20% of the
    image.  In a full-body standing photo, the head occupies roughly the
    top 20% of the frame — this heuristic is derived from the typical
    distribution of MediaPipe pose landmarks and provides a safe default
    when frontal face detection fails (e.g. profile view, poor lighting).

    Args:
        bgr: A BGR uint8 numpy array as returned by ``cv2.imdecode``.

    Returns:
        A new BGR array with the face region blurred.  The original array
        is never modified in-place.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    result = bgr.copy()

    if len(faces) > 0:
        for x, y, w, h in faces:
            roi = result[y : y + h, x : x + w]
            # sigmaX proportional to face size — small faces still become
            # unrecognisable while large faces are fully obscured.
            result[y : y + h, x : x + w] = cv2.GaussianBlur(
                roi, (0, 0), sigmaX=max(w, h) / 4
            )
    else:
        # Fallback: blur the top 20% of the frame.
        top = int(result.shape[0] * 0.20)
        if top > 0:
            result[:top] = cv2.GaussianBlur(
                result[:top], (0, 0), sigmaX=result.shape[1] / 10
            )

    return result


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    request: Request,
    image: UploadFile = File(..., description="Full-body JPEG or PNG image."),
    height_cm: float = Form(..., description="Standing height in centimetres."),
    age: int = Form(..., description="Age in years."),
    gender: int = Form(..., description="Biological sex: 0 for female, 1 for male."),
) -> AnalyzeResponse:
    """Estimate body weight from an uploaded image plus height, age, and gender.

    The image is processed through three stages:

    1. **Pose detection** — MediaPipe extracts 33 body landmarks.
    2. **Feature engineering** — 8 anthropometric ratios plus height, age,
       and gender are derived from the landmarks and user inputs.
    3. **Weight estimation** — 50 Monte Carlo Dropout forward passes through
       the BioScanMLP yield a point estimate and a 95% confidence interval.

    Face blurring is applied to the raw image **before** any landmark
    extraction so that no identifiable facial data enters the pipeline.

    Raises:
        HTTPException 422: If ``height_cm``, ``age``, or ``gender`` are out
            of range, if the uploaded file cannot be decoded as an image, or
            if MediaPipe cannot detect a person in the frame.
        HTTPException 500: For any unexpected internal error.  The response
            body intentionally omits internal details to avoid leaking stack
            traces or model internals to callers.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate form fields (manual, so we control the error message)   #
    # ------------------------------------------------------------------ #
    if not (_HEIGHT_MIN <= height_cm <= _HEIGHT_MAX):
        raise HTTPException(
            status_code=422,
            detail=(
                f"height_cm must be between {_HEIGHT_MIN:.0f} and "
                f"{_HEIGHT_MAX:.0f} cm; received {height_cm}."
            ),
        )
    if not (_AGE_MIN <= age <= _AGE_MAX):
        raise HTTPException(
            status_code=422,
            detail=(
                f"age must be between {_AGE_MIN} and {_AGE_MAX} years; "
                f"received {age}."
            ),
        )
    if gender not in _VALID_GENDERS:
        raise HTTPException(
            status_code=422,
            detail=f"gender must be 0 (female) or 1 (male); received {gender}.",
        )

    # ------------------------------------------------------------------ #
    # 2. Decode image                                                      #
    # ------------------------------------------------------------------ #
    img_bytes = await image.read()

    # Processing time is measured from the moment we begin working on the
    # decoded pixel data, not from the network receive time.
    t0 = time.perf_counter()

    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not decode the uploaded file as an image. "
                "Ensure the file is a valid JPEG or PNG."
            ),
        )

    # ------------------------------------------------------------------ #
    # 3. Face blur — must happen before pose detection                     #
    # ------------------------------------------------------------------ #
    bgr = _blur_face(bgr)

    # ------------------------------------------------------------------ #
    # 4. Three-stage pipeline                                              #
    # ------------------------------------------------------------------ #
    try:
        # Stage 1 — pose detection
        pose_result = request.app.state.pose_detector.detect(bgr)

        # Stage 2 — feature engineering
        features = compute_features(pose_result, height_cm=height_cm, age=age, gender=gender)

        # Stage 3 — weight estimation (MC Dropout)
        estimation = request.app.state.weight_estimator.predict(features)

    except ValueError as exc:
        # PoseDetector raises ValueError when no person is detected, and
        # compute_features raises it for low-visibility or coincident landmarks.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception:
        # Catch-all for unexpected failures.  Log the full traceback server-
        # side but return a safe message that does not leak internals.
        _log.exception(
            "Unexpected error during /analyze | height_cm=%s | age=%s | gender=%s",
            height_cm,
            age,
            gender,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while processing the image. "
                "Please try again."
            ),
        )

    processing_time_ms = (time.perf_counter() - t0) * 1000.0

    return AnalyzeResponse(
        estimated_weight_kg=estimation.estimated_weight_kg,
        confidence_interval_low=estimation.confidence_interval_low,
        confidence_interval_high=estimation.confidence_interval_high,
        prediction_std=estimation.prediction_std,
        low_confidence=estimation.low_confidence,
        input_height_cm=height_cm,
        processing_time_ms=processing_time_ms,
    )
