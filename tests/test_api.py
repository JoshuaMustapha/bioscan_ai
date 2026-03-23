"""Integration tests for POST /analyze and GET /health.

PoseDetector and WeightEstimator are patched at the api.main import level so
that no model checkpoints need to exist on disk.  The patched constructors
store MagicMock instances in app.state, which the route accesses via
request.app.state as normal.

compute_features runs for real between the two mocked stages, so the
pose_result returned by mock_detector.detect() must contain valid landmark
geometry (all critical landmarks visible, non-zero hip width).
"""

from __future__ import annotations

import io
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import AnalyzeResponse
from pipeline.pose_detector import PoseDetectionResult
from pipeline.weight_estimator import WeightEstimationResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_test_image_bytes() -> io.BytesIO:
    """Minimal valid JPEG image as an in-memory byte stream."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return io.BytesIO(buf.tobytes())


def _make_pose_result() -> PoseDetectionResult:
    """Synthetic PoseDetectionResult accepted by compute_features.

    All critical landmarks have visibility=0.9 and are positioned so that
    shoulder_width, hip_width, etc. are all well-defined and non-zero.
    """
    import mediapipe as mp

    _PL = mp.solutions.pose.PoseLandmark

    lm = [
        types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9)
        for _ in range(33)
    ]
    lm[_PL.LEFT_SHOULDER]  = types.SimpleNamespace(x=0.35, y=0.30, z=0.0, visibility=0.9)
    lm[_PL.RIGHT_SHOULDER] = types.SimpleNamespace(x=0.65, y=0.30, z=0.0, visibility=0.9)
    lm[_PL.LEFT_HIP]       = types.SimpleNamespace(x=0.40, y=0.60, z=0.0, visibility=0.9)
    lm[_PL.RIGHT_HIP]      = types.SimpleNamespace(x=0.60, y=0.60, z=0.0, visibility=0.9)
    lm[_PL.LEFT_WRIST]     = types.SimpleNamespace(x=0.25, y=0.55, z=0.0, visibility=0.9)
    lm[_PL.RIGHT_WRIST]    = types.SimpleNamespace(x=0.75, y=0.55, z=0.0, visibility=0.9)
    lm[_PL.LEFT_ANKLE]     = types.SimpleNamespace(x=0.43, y=0.90, z=0.0, visibility=0.9)
    lm[_PL.RIGHT_ANKLE]    = types.SimpleNamespace(x=0.57, y=0.90, z=0.0, visibility=0.9)

    return PoseDetectionResult(
        landmarks=lm,
        confidence=0.9,
        timestamp=datetime.now(tz=timezone.utc),
    )


_MOCK_ESTIMATION = WeightEstimationResult(
    estimated_weight_kg=75.3,
    confidence_interval_low=69.9,
    confidence_interval_high=80.7,
    prediction_std=2.75,
    low_confidence=False,
    mc_passes_used=50,
)


# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """TestClient with PoseDetector and WeightEstimator replaced by MagicMocks.

    Patching both constructors in api.main prevents the lifespan from
    attempting to load model files from disk and stores the mock instances
    in app.state instead.
    """
    mock_detector = MagicMock()
    mock_detector.detect.return_value = _make_pose_result()

    mock_estimator = MagicMock()
    mock_estimator.predict.return_value = _MOCK_ESTIMATION

    with patch("api.main.PoseDetector", return_value=mock_detector), \
         patch("api.main.WeightEstimator", return_value=mock_estimator):
        with TestClient(app) as c:
            yield c


def _post_analyze(client: TestClient, height_cm=175, age=30, image=None):
    """Helper that posts to /analyze with sensible defaults."""
    if image is None:
        image = _make_test_image_bytes()
    return client.post(
        "/analyze",
        files={"image": ("photo.jpg", image, "image/jpeg")},
        data={"height_cm": str(height_cm), "age": str(age)},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_valid_request_returns_200_with_valid_response(client: TestClient):
    resp = _post_analyze(client, height_cm=175, age=30)

    assert resp.status_code == 200

    body = resp.json()
    # All AnalyzeResponse fields must be present.
    required_fields = {
        "estimated_weight_kg",
        "confidence_interval_low",
        "confidence_interval_high",
        "prediction_std",
        "low_confidence",
        "input_height_cm",
        "processing_time_ms",
    }
    assert required_fields.issubset(body.keys())

    assert body["estimated_weight_kg"] == pytest.approx(75.3)
    assert body["input_height_cm"] == pytest.approx(175.0)
    assert body["low_confidence"] is False
    assert body["processing_time_ms"] >= 0


def test_input_height_cm_is_passed_through_unchanged(client: TestClient):
    """input_height_cm in the response must equal the value sent in the request."""
    resp = _post_analyze(client, height_cm=162.5, age=25)
    assert resp.status_code == 200
    assert resp.json()["input_height_cm"] == pytest.approx(162.5)


def test_missing_height_cm_returns_422(client: TestClient):
    resp = client.post(
        "/analyze",
        files={"image": ("photo.jpg", _make_test_image_bytes(), "image/jpeg")},
        data={"age": "30"},  # height_cm omitted
    )
    assert resp.status_code == 422


def test_height_cm_out_of_range_returns_422_with_clear_message(client: TestClient):
    resp = _post_analyze(client, height_cm=400, age=30)

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "height_cm" in detail.lower()
    assert "400" in detail


def test_age_out_of_range_returns_422(client: TestClient):
    resp = _post_analyze(client, height_cm=175, age=200)

    assert resp.status_code == 422
    assert "age" in resp.json()["detail"].lower()


def test_no_person_detected_returns_422(client: TestClient):
    """PoseDetector raising ValueError must surface as a 422 with the original message."""
    client.app.state.pose_detector.detect.side_effect = ValueError(
        "No person detected in the image."
    )
    try:
        resp = _post_analyze(client, height_cm=175, age=30)
        assert resp.status_code == 422
        assert "No person" in resp.json()["detail"]
    finally:
        # Reset so other tests are unaffected.
        client.app.state.pose_detector.detect.side_effect = None


def test_health_endpoint_returns_ok(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
