"""Stage 2: Anthropometric feature engineering from pose landmarks.

Receives a PoseDetectionResult from Stage 1 and computes a fixed-length
vector of body-shape features for consumption by the MLP in Stage 3.
All distances are computed in MediaPipe's normalised coordinate space
([0, 1] relative to image dimensions), so they are scale-independent and
unaffected by the subject's absolute distance from the camera.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

from pipeline.pose_detector import PoseDetectionResult

# Convenience alias so call-sites read as plain English.
_PL = PoseLandmark

# Landmarks that must be sufficiently visible for features to be reliable.
# Any landmark below the threshold triggers a ValueError before computation.
_CRITICAL_LANDMARKS: tuple[PoseLandmark, ...] = (
    _PL.LEFT_SHOULDER,
    _PL.RIGHT_SHOULDER,
    _PL.LEFT_HIP,
    _PL.RIGHT_HIP,
    _PL.LEFT_WRIST,
    _PL.RIGHT_WRIST,
    _PL.LEFT_ANKLE,
    _PL.RIGHT_ANKLE,
)

_VISIBILITY_THRESHOLD: float = 0.5


@dataclass
class FeatureVector:
    """The engineered feature set passed to the MLP for weight estimation.

    All distance-based features are computed in MediaPipe's normalised
    coordinate space and are therefore dimensionless ratios.  ``height_cm``
    and ``age`` are the only features with real-world units; they are
    passed through directly from user input and are not estimated from
    the image.

    Attributes:
        shoulder_width: Euclidean distance between the left and right
            shoulder landmarks.
        hip_width: Euclidean distance between the left and right hip
            landmarks.
        shoulder_to_hip_ratio: ``shoulder_width`` divided by
            ``hip_width``.  Captures body-shape taper (e.g. inverted
            triangle vs. pear).
        torso_height: Euclidean distance from the shoulder midpoint to
            the hip midpoint.  Used as an implicit normalisation base for
            ``silhouette_area``.
        silhouette_area: ``shoulder_width * torso_height``.  A normalised
            proxy for body cross-sectional area; using torso height as one
            factor reduces perspective distortion compared to a raw pixel
            count.
        left_arm_length: Euclidean distance from the left shoulder to the
            left wrist.
        right_arm_length: Euclidean distance from the right shoulder to
            the right wrist.
        leg_length: Euclidean distance from the hip midpoint to the ankle
            midpoint.
        height_cm: User-supplied standing height in centimetres.  Not
            estimated from the image.
        age: User-supplied age in years, stored as a float for uniformity
            with the rest of the feature vector.
        gender: User-supplied biological sex encoded as a float: 1.0 for
            male, 0.0 for female.  Not estimated from the image.
    """

    shoulder_width: float
    hip_width: float
    shoulder_to_hip_ratio: float
    torso_height: float
    silhouette_area: float
    left_arm_length: float
    right_arm_length: float
    leg_length: float
    height_cm: float
    age: float
    gender: float

    def to_list(self) -> list[float]:
        """Return all features as an ordered list of floats.

        The order matches the feature list in ``training/config.py`` and
        is the format expected by the scaler and MLP at inference time.
        Call this method when constructing a PyTorch input tensor.

        Returns:
            A list of 11 floats in the canonical feature order.
        """
        return [
            self.shoulder_width,
            self.hip_width,
            self.shoulder_to_hip_ratio,
            self.torso_height,
            self.silhouette_area,
            self.left_arm_length,
            self.right_arm_length,
            self.leg_length,
            self.height_cm,
            self.age,
            self.gender,
        ]


def compute_features(
    result: PoseDetectionResult,
    height_cm: float,
    age: int,
    gender: int,
) -> FeatureVector:
    """Compute the anthropometric feature vector from a pose detection result.

    Validates that all critical landmarks are sufficiently visible, then
    derives each feature from the named landmark positions using MediaPipe's
    ``PoseLandmark`` enum.  User-supplied ``height_cm``, ``age``, and
    ``gender`` are appended as-is; they are not derived from the image.

    Args:
        result: The :class:`~pipeline.pose_detector.PoseDetectionResult`
            returned by Stage 1.  Must contain all 33 MediaPipe landmarks.
        height_cm: The user's standing height in centimetres, as provided
            in the API request.
        age: The user's age in years, as provided in the API request.
        gender: Biological sex encoded as an integer: 1 for male, 0 for
            female, as provided in the API request.

    Returns:
        A :class:`FeatureVector` containing 11 features ready for scaling
        and MLP inference.

    Raises:
        ValueError: If any critical landmark has a visibility score below
            ``_VISIBILITY_THRESHOLD`` (0.5), or if the hip landmarks are
            coincident (indicating corrupt landmark data).
    """
    _validate_visibility(result)

    lm = result.landmarks

    # Named landmark references — no magic index numbers.
    ls = lm[_PL.LEFT_SHOULDER]
    rs = lm[_PL.RIGHT_SHOULDER]
    lh = lm[_PL.LEFT_HIP]
    rh = lm[_PL.RIGHT_HIP]
    lw = lm[_PL.LEFT_WRIST]
    rw = lm[_PL.RIGHT_WRIST]
    la = lm[_PL.LEFT_ANKLE]
    ra = lm[_PL.RIGHT_ANKLE]

    shoulder_mid = _midpoint(ls, rs)
    hip_mid = _midpoint(lh, rh)
    ankle_mid = _midpoint(la, ra)

    shoulder_width = _euclidean(ls, rs)
    hip_width = _euclidean(lh, rh)

    if hip_width == 0.0:
        raise ValueError(
            "Left and right hip landmarks are coincident (hip_width = 0). "
            "Landmark data may be corrupt or the subject may be sideways-on."
        )

    shoulder_to_hip_ratio = shoulder_width / hip_width
    torso_height = _euclidean(shoulder_mid, hip_mid)
    silhouette_area = shoulder_width * torso_height
    left_arm_length = _euclidean(ls, lw)
    right_arm_length = _euclidean(rs, rw)
    leg_length = _euclidean(hip_mid, ankle_mid)

    return FeatureVector(
        shoulder_width=shoulder_width,
        hip_width=hip_width,
        shoulder_to_hip_ratio=shoulder_to_hip_ratio,
        torso_height=torso_height,
        silhouette_area=silhouette_area,
        left_arm_length=left_arm_length,
        right_arm_length=right_arm_length,
        leg_length=leg_length,
        height_cm=float(height_cm),
        age=float(age),
        gender=float(gender),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_visibility(result: PoseDetectionResult) -> None:
    """Raise a ValueError if any critical landmark is below the visibility threshold.

    Called at the start of :func:`compute_features` before any distance
    calculations begin.  Failing fast here gives a clear error message
    pinpointing which landmark is problematic, rather than silently
    producing unreliable features.

    Args:
        result: The pose detection result whose landmarks will be inspected.

    Raises:
        ValueError: If any landmark in ``_CRITICAL_LANDMARKS`` has a
            ``visibility`` score strictly below ``_VISIBILITY_THRESHOLD``.
    """
    for landmark_enum in _CRITICAL_LANDMARKS:
        lm = result.landmarks[landmark_enum]
        if lm.visibility < _VISIBILITY_THRESHOLD:
            raise ValueError(
                f"Landmark {landmark_enum.name} has visibility "
                f"{lm.visibility:.2f}, below the required threshold of "
                f"{_VISIBILITY_THRESHOLD:.2f}. Ensure the full body is "
                "unobstructed, facing the camera, and well-lit."
            )


def _euclidean(a: types.SimpleNamespace, b: types.SimpleNamespace) -> float:
    """Return the Euclidean distance between two 2-D points.

    Operates on the ``x`` and ``y`` attributes only.  The ``z`` coordinate
    (MediaPipe's relative depth estimate) is intentionally excluded because
    it is not metric and introduces noise when mixed with the normalised
    x/y plane distances.

    Args:
        a: Any object exposing ``.x`` and ``.y`` float attributes
           (a MediaPipe ``NormalizedLandmark`` or a :func:`_midpoint` result).
        b: Same type as ``a``.

    Returns:
        The 2-D Euclidean distance as a Python float.
    """
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _midpoint(a: types.SimpleNamespace, b: types.SimpleNamespace) -> types.SimpleNamespace:
    """Return a point at the geometric midpoint of two 2-D points.

    The returned object exposes ``.x`` and ``.y`` attributes and can be
    passed directly to :func:`_euclidean` as if it were a landmark.

    Args:
        a: Any object exposing ``.x`` and ``.y`` float attributes.
        b: Same type as ``a``.

    Returns:
        A :class:`types.SimpleNamespace` with ``.x`` and ``.y`` set to
        the midpoint coordinates.
    """
    return types.SimpleNamespace(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0)
