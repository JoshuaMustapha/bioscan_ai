"""Stage 1: Human pose detection using MediaPipe Pose.

Wraps MediaPipe PoseLandmarker (Tasks API) to extract 33 body landmark
coordinates from a single image. Designed to be instantiated once (e.g. at
API startup) and reused across requests to amortise initialisation cost.

Requires a MediaPipe pose landmarker ``.task`` model file.  Download
``pose_landmarker_full.task`` from the MediaPipe model repository and place it
in ``model/`` before starting the server::

    https://storage.googleapis.com/mediapipe-models/pose_landmarker/
    pose_landmarker_full/float16/latest/pose_landmarker_full.task
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)


@dataclass
class PoseDetectionResult:
    """The output of a single pose detection pass.

    Attributes:
        landmarks: Ordered list of 33 ``NormalizedLandmark`` objects as
            returned by MediaPipe PoseLandmarker.  Each landmark exposes
            ``.x``, ``.y``, ``.z`` (all normalised to [0, 1] relative to
            image dimensions, with ``z`` representing relative depth), and
            ``.visibility`` (confidence that the landmark is visible,
            in [0, 1]).  Landmark ordering follows the
            ``PoseLandmark`` enum.
        confidence: Mean visibility score across all 33 landmarks,
            in [0, 1].  Provides a single scalar summary of how
            reliably the pose was detected.
        timestamp: UTC timestamp recorded immediately before the
            MediaPipe inference call.
    """

    landmarks: list  # list[mediapipe.tasks.python.components.containers.landmark.NormalizedLandmark]
    confidence: float
    timestamp: datetime


class PoseDetector:
    """Extracts 33 body landmarks from a still image using MediaPipe PoseLandmarker.

    Instantiate once and call :meth:`detect` for each image.  The
    underlying MediaPipe graph is kept alive between calls, which avoids
    repeated initialisation overhead.  Call :meth:`close` (or use the
    instance as a context manager) when the detector is no longer needed.

    Args:
        model_path: Path to the MediaPipe pose landmarker ``.task`` model
            file.  Defaults to ``model/pose_landmarker_full.task`` relative
            to the repository root.  Raises :exc:`FileNotFoundError` at
            construction time if the file does not exist.
        min_detection_confidence: Minimum confidence value [0.0, 1.0]
            required for pose detection to be considered successful.
            Defaults to 0.5.
        min_tracking_confidence: Minimum confidence value [0.0, 1.0]
            for landmark tracking between frames.  Has limited effect
            for still-image inference, but kept for API consistency.
            Defaults to 0.5.
        max_side_px: Images whose longest side exceeds this value are
            resized (preserving aspect ratio) before inference.
            Defaults to 1280.

    Example::

        detector = PoseDetector()
        result = detector.detect("photo.jpg")
        print(result.confidence, len(result.landmarks))

        # Or as a context manager:
        with PoseDetector(model_path="model/pose_landmarker_full.task") as detector:
            result = detector.detect(numpy_bgr_array)
    """

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_side_px: int = 1280,
    ) -> None:
        """Initialise the MediaPipe PoseLandmarker graph with the given configuration."""
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "model"
                / "pose_landmarker_full.task"
            )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe pose landmarker model not found: {model_path}\n"
                "Download pose_landmarker_full.task and place it in model/:\n"
                "  https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = PoseLandmarker.create_from_options(options)
        self._max_side_px = max_side_px
        # A single lock serialises concurrent detect() calls on this instance.
        # For production scale, replace with a PoseDetector pool (one instance
        # per worker thread) or offload to a ProcessPoolExecutor to eliminate
        # lock contention entirely.
        self._lock = threading.Lock()

    def detect(self, image: Union[np.ndarray, str, Path]) -> PoseDetectionResult:
        """Run MediaPipe PoseLandmarker on an image and return the 33 body landmarks.

        Args:
            image: The source image, provided as one of:

                * A **numpy array** — assumed to be in BGR channel order
                  (as produced by ``cv2.imread``).  Shape must be
                  ``(H, W, 3)``, dtype ``uint8``.
                * A **file path** (``str`` or ``pathlib.Path``) pointing
                  to a JPEG or PNG image on disk.

        Returns:
            :class:`PoseDetectionResult` containing the 33 raw
            ``NormalizedLandmark`` objects, a mean-visibility confidence
            score, and a UTC timestamp.

        Raises:
            FileNotFoundError: If a file path is given but the file does
                not exist on disk.
            ValueError: If the file at the given path cannot be decoded
                as an image, or if MediaPipe cannot detect a person in
                the image.
            TypeError: If ``image`` is neither a numpy array nor a path.
        """
        rgb = self._load_as_rgb(image)
        rgb = self._cap_size(rgb)
        mp_image = MpImage(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp = datetime.now(tz=timezone.utc)
        with self._lock:
            results = self._pose.detect(mp_image)

        if not results.pose_landmarks:
            raise ValueError(
                "No person detected in the image. "
                "Ensure the full body is visible in the frame, the image is "
                "well-lit, and the subject is not heavily occluded or cropped."
            )

        landmark_list = results.pose_landmarks[0]
        confidence = float(np.mean([lm.visibility or 0.0 for lm in landmark_list]))

        return PoseDetectionResult(
            landmarks=landmark_list,
            confidence=confidence,
            timestamp=timestamp,
        )

    def _load_as_rgb(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load and convert an image to an RGB numpy array.

        MediaPipe expects RGB input.  OpenCV loads images as BGR, so
        conversion is applied automatically when reading from disk or
        when a numpy array with three channels is provided (BGR assumed).

        Args:
            image: A numpy array or a path to an image file.

        Returns:
            An RGB numpy array of shape ``(H, W, 3)``, dtype ``uint8``.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file exists but cannot be decoded.
            TypeError: If the input type is not supported.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(
                    f"Could not decode image at path: {path}. "
                    "Ensure the file is a valid JPEG or PNG."
                )
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # Assume BGR (OpenCV convention) and convert to RGB.
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Single-channel or already-RGB array — pass through unchanged.
            return image

        raise TypeError(
            f"Expected a numpy array or a file path (str/Path), "
            f"got {type(image).__name__}."
        )

    def _cap_size(self, rgb: np.ndarray) -> np.ndarray:
        """Resize an RGB image so its longest side does not exceed ``max_side_px``.

        Aspect ratio is preserved.  Images already within the limit are
        returned unchanged (no copy is made).

        Args:
            rgb: An RGB numpy array of shape ``(H, W, 3)``.

        Returns:
            The original array if no resize is needed, otherwise a new
            array resized with ``cv2.INTER_AREA`` (best quality for
            downscaling).
        """
        h, w = rgb.shape[:2]
        longest = max(h, w)
        if longest <= self._max_side_px:
            return rgb
        scale = self._max_side_px / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        """Release the underlying MediaPipe PoseLandmarker resources.

        Call this when the detector is no longer needed, or prefer using
        the detector as a context manager to have resources released
        automatically.
        """
        self._pose.close()

    def __enter__(self) -> "PoseDetector":
        """Enable use as a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release resources on context manager exit."""
        self.close()
