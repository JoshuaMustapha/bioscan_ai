"""Tests for Stage 1 (PoseDetector) and Stage 2 (compute_features).

Synthetic landmark data (types.SimpleNamespace objects) is used throughout to
avoid loading real images or running MediaPipe inference in unit tests.
MediaPipe's Pose graph is patched at the constructor level so that each
PoseDetector fixture is instantiated without loading TFLite models.
"""

from __future__ import annotations

import threading
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.feature_engineer import FeatureVector, compute_features
from pipeline.pose_detector import PoseDetectionResult, PoseDetector


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_raw_landmark(x: float = 0.5, y: float = 0.5, visibility: float = 0.9):
    """A SimpleNamespace that quacks like a MediaPipe NormalizedLandmark."""
    return types.SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility)


def _make_mp_result(visible: bool = True, visibility: float = 0.9):
    """Fake return value for mp.solutions.pose.Pose.process()."""
    result = MagicMock()
    if visible:
        result.pose_landmarks = MagicMock()
        result.pose_landmarks.landmark = [
            MagicMock(x=0.5, y=0.5, z=0.0, visibility=visibility)
            for _ in range(33)
        ]
    else:
        result.pose_landmarks = None
    return result


def _make_pose_result(visibility: float = 0.9) -> PoseDetectionResult:
    """Synthetic PoseDetectionResult with plausible landmark positions.

    Used by compute_features tests — no MediaPipe processing involved.
    Landmark indices follow mp.solutions.pose.PoseLandmark (IntEnum).
    """
    import mediapipe as mp

    _PL = mp.solutions.pose.PoseLandmark

    lm = [_make_raw_landmark(visibility=visibility) for _ in range(33)]

    # Place key landmarks at distinct, geometrically sensible positions.
    lm[_PL.LEFT_SHOULDER]  = _make_raw_landmark(x=0.35, y=0.30, visibility=visibility)
    lm[_PL.RIGHT_SHOULDER] = _make_raw_landmark(x=0.65, y=0.30, visibility=visibility)
    lm[_PL.LEFT_HIP]       = _make_raw_landmark(x=0.40, y=0.60, visibility=visibility)
    lm[_PL.RIGHT_HIP]      = _make_raw_landmark(x=0.60, y=0.60, visibility=visibility)
    lm[_PL.LEFT_WRIST]     = _make_raw_landmark(x=0.25, y=0.55, visibility=visibility)
    lm[_PL.RIGHT_WRIST]    = _make_raw_landmark(x=0.75, y=0.55, visibility=visibility)
    lm[_PL.LEFT_ANKLE]     = _make_raw_landmark(x=0.43, y=0.90, visibility=visibility)
    lm[_PL.RIGHT_ANKLE]    = _make_raw_landmark(x=0.57, y=0.90, visibility=visibility)

    return PoseDetectionResult(
        landmarks=lm,
        confidence=visibility,
        timestamp=datetime.now(tz=timezone.utc),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pose_detector():
    """PoseDetector with MediaPipe Pose constructor patched out.

    Yields a ``(PoseDetector, mock_pose_instance)`` tuple so that individual
    tests can configure ``mock_pose_instance.process.return_value`` and
    ``mock_pose_instance.process.side_effect`` freely.
    """
    with patch("mediapipe.solutions.pose.Pose") as mock_pose_cls:
        mock_pose_instance = MagicMock()
        mock_pose_cls.return_value = mock_pose_instance
        # Default: valid result with 33 landmarks.
        mock_pose_instance.process.return_value = _make_mp_result(visible=True)
        detector = PoseDetector()
        yield detector, mock_pose_instance
    # patch exits; detector.close() calls mock_pose_instance.close() safely.


# ── PoseDetector tests ────────────────────────────────────────────────────────


def test_detect_returns_33_landmarks_and_positive_confidence(pose_detector):
    detector, mock_pose = pose_detector
    mock_pose.process.return_value = _make_mp_result(visible=True, visibility=0.85)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(img)

    assert isinstance(result, PoseDetectionResult)
    assert len(result.landmarks) == 33
    assert 0 < result.confidence <= 1.0
    assert result.timestamp is not None


def test_detect_no_person_raises_value_error(pose_detector):
    detector, mock_pose = pose_detector
    mock_pose.process.return_value = _make_mp_result(visible=False)

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No person detected"):
        detector.detect(img)


def test_detect_oversized_image_is_resized_before_processing(pose_detector):
    """A 4000×4000 image must be downscaled to max_side_px before inference."""
    detector, mock_pose = pose_detector
    mock_pose.process.return_value = _make_mp_result(visible=True)

    big_img = np.zeros((4000, 4000, 3), dtype=np.uint8)
    detector.detect(big_img)

    # Inspect the RGB array that was actually passed to process().
    called_rgb = mock_pose.process.call_args[0][0]
    assert max(called_rgb.shape[:2]) <= detector._max_side_px


def test_detect_accepts_numpy_array(pose_detector):
    """BGR numpy array is accepted and converted to RGB before process()."""
    detector, mock_pose = pose_detector
    mock_pose.process.return_value = _make_mp_result(visible=True)

    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(bgr)

    assert len(result.landmarks) == 33
    # Verify RGB conversion: process() must NOT receive a BGR array.
    called_rgb = mock_pose.process.call_args[0][0]
    assert called_rgb.shape == (480, 640, 3)


def test_detect_accepts_file_path(pose_detector, tmp_path):
    """A file path string/Path is loaded, BGR→RGB converted, and processed."""
    import cv2

    detector, mock_pose = pose_detector
    mock_pose.process.return_value = _make_mp_result(visible=True)

    img_file = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_file), np.zeros((240, 320, 3), dtype=np.uint8))

    result = detector.detect(img_file)
    assert len(result.landmarks) == 33

    result_from_str = detector.detect(str(img_file))
    assert len(result_from_str.landmarks) == 33


def test_detect_concurrent_calls_do_not_corrupt_results(pose_detector):
    """Two threads calling detect() simultaneously both receive valid results."""
    detector, mock_pose = pose_detector
    mock_pose.process.return_value = _make_mp_result(visible=True)

    results: list[PoseDetectionResult] = []
    errors: list[Exception] = []

    def run():
        try:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            results.append(detector.detect(img))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=run)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == [], f"Concurrent detect() raised: {errors}"
    assert len(results) == 2
    assert all(len(r.landmarks) == 33 for r in results)


# ── compute_features tests ────────────────────────────────────────────────────


def test_compute_features_returns_feature_vector_with_10_float_fields():
    pose_result = _make_pose_result()
    fv = compute_features(pose_result, height_cm=175.0, age=30)

    assert isinstance(fv, FeatureVector)
    fields = fv.to_list()
    assert len(fields) == 10
    assert all(isinstance(v, float) for v in fields), (
        f"Non-float field found: {[(i, type(v)) for i, v in enumerate(fields) if not isinstance(v, float)]}"
    )


def test_compute_features_shoulder_to_hip_ratio_equals_width_division():
    """shoulder_to_hip_ratio must equal shoulder_width / hip_width exactly."""
    pose_result = _make_pose_result()
    fv = compute_features(pose_result, height_cm=175.0, age=30)

    assert fv.shoulder_to_hip_ratio == pytest.approx(
        fv.shoulder_width / fv.hip_width, rel=1e-6
    )


def test_compute_features_low_visibility_raises_naming_landmark():
    """ValueError must identify the specific landmark that failed visibility."""
    import mediapipe as mp

    pose_result = _make_pose_result(visibility=0.9)
    # Degrade one critical landmark below the 0.5 threshold.
    pose_result.landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].visibility = 0.3

    with pytest.raises(ValueError, match="LEFT_SHOULDER"):
        compute_features(pose_result, height_cm=175.0, age=30)


def test_compute_features_height_and_age_passed_through_unchanged():
    """height_cm and age must appear verbatim in the FeatureVector."""
    pose_result = _make_pose_result()
    fv = compute_features(pose_result, height_cm=182.5, age=47)

    assert fv.height_cm == pytest.approx(182.5)
    assert fv.age == pytest.approx(47.0)
