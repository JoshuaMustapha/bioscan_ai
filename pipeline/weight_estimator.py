"""Stage 3: Monte Carlo Dropout weight estimation.

Loads a trained BioScanMLP checkpoint and a fitted feature scaler once at
startup, then runs 50 stochastic forward passes per request to produce a
point estimate and a 95% confidence interval for the subject's body weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from model.mlp import BioScanMLP, IN_FEATURES
from pipeline.feature_engineer import FeatureVector

# Number of Monte Carlo forward passes per inference call.
# Higher values reduce variance in the confidence interval at the cost of
# latency; 50 is a practical default for single-request API use.
_DEFAULT_MC_PASSES: int = 50

# Z-score for a 95% confidence interval under the normal approximation.
_CI_Z_95: float = 1.96


@dataclass
class WeightEstimationResult:
    """The output of a single Monte Carlo Dropout inference run.

    Attributes:
        estimated_weight_kg: Mean of all MC forward pass outputs, in
            kilograms.  This is the primary point estimate returned to
            the user.
        confidence_interval_low: Lower bound of the 95% confidence
            interval: ``mean - 1.96 * std``.
        confidence_interval_high: Upper bound of the 95% confidence
            interval: ``mean + 1.96 * std``.
        prediction_std: Raw standard deviation of the MC pass outputs, in
            kilograms.  A low value means the 50 passes agreed closely; a
            high value indicates **epistemic uncertainty** — the input
            feature vector is in a region of the training distribution that
            the model has seen infrequently, and the passes are pulling in
            inconsistent directions.  This is qualitatively different from
            generic imprecision: it signals that the estimate may be
            unreliable regardless of the CI width.
        low_confidence: ``True`` when ``prediction_std`` exceeds the
            configured threshold (default 5.0 kg).  When set, the point
            estimate should be presented with an explicit warning rather
            than displayed as a normal result — the CI is wide enough that
            even the mean may not be a meaningful central tendency.
        mc_passes_used: Number of stochastic forward passes that produced
            this result.  Recorded so callers can distinguish results
            generated with different pass counts.
    """

    estimated_weight_kg: float
    confidence_interval_low: float
    confidence_interval_high: float
    prediction_std: float
    low_confidence: bool
    mc_passes_used: int


class WeightEstimator:
    """Loads a BioScanMLP checkpoint and runs Monte Carlo Dropout inference.

    Designed to be instantiated once at API startup (inside the FastAPI
    lifespan context) and reused across all requests.  Loading the checkpoint
    and scaler is expensive; prediction is cheap.

    The scaler saved alongside the checkpoint must be the exact ``sklearn``
    scaler fitted on the training data.  Applying a different scaler, or no
    scaler at all, will shift all predictions by an unpredictable amount.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint produced by
            ``model/trainer.py``.  Expected to be a dict with keys
            ``model_state_dict``, ``config``, and ``best_val_loss``.
        scaler_path: Path to the ``bioscan_scaler.pkl`` file saved by the
            training pipeline alongside the checkpoint.
        mc_passes: Number of stochastic forward passes to run per call to
            :meth:`predict`.  Defaults to 50.
        std_threshold_kg: Standard deviation threshold in kilograms above
            which a result is flagged as low confidence.  A high MC std
            indicates **epistemic uncertainty** — the input is out of the
            training distribution — rather than generic imprecision.
            Defaults to 5.0 kg, corresponding to a CI width of ~20 kg.
            Tune this against :mod:`training.evaluate` calibration metrics
            once a trained checkpoint is available.

    Raises:
        RuntimeError: If either ``checkpoint_path`` or ``scaler_path`` does
            not exist at construction time, or if the checkpoint's recorded
            ``in_features`` does not match :data:`~model.mlp.IN_FEATURES`.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        scaler_path: str | Path,
        mc_passes: int = _DEFAULT_MC_PASSES,
        std_threshold_kg: float = 5.0,
    ) -> None:
        """Load the model checkpoint and feature scaler from disk."""
        self._log = logging.getLogger(__name__)

        checkpoint_path = Path(checkpoint_path)
        scaler_path = Path(scaler_path)

        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Model checkpoint not found: {checkpoint_path}. "
                "Run training/train.py to produce a checkpoint before "
                "starting the API server."
            )
        if not scaler_path.exists():
            raise RuntimeError(
                f"Feature scaler not found: {scaler_path}. "
                "The scaler is saved alongside the checkpoint during training. "
                "Ensure both files are present in the same checkpoint directory."
            )

        # Load the full checkpoint dict saved by model/trainer.py.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,  # Prevents arbitrary code execution from malicious .pth files.
        )

        # Log and verify config metadata recorded at training time.
        checkpoint_cfg = checkpoint.get("config", {})
        self._log.info(
            "Loading checkpoint | in_features=%s | hidden_sizes=%s | "
            "dropout_p=%s | best_val_loss=%s | feature_columns=%s",
            checkpoint_cfg.get("in_features"),
            checkpoint_cfg.get("hidden_sizes"),
            checkpoint_cfg.get("dropout_p"),
            checkpoint.get("best_val_loss"),
            checkpoint_cfg.get("feature_columns"),
        )

        ckpt_in_features = checkpoint_cfg.get("in_features")
        if ckpt_in_features is not None and ckpt_in_features != IN_FEATURES:
            raise RuntimeError(
                f"Checkpoint in_features={ckpt_in_features} does not match "
                f"the current model IN_FEATURES={IN_FEATURES}. The checkpoint "
                "was trained with a different feature set — retrain or use a "
                "matching checkpoint."
            )

        self._model = BioScanMLP()
        self._model.load_state_dict(checkpoint["model_state_dict"])
        # Do NOT call self._model.eval(). _MCDropout ignores model mode and
        # stays active unconditionally, but keeping the model in train() mode
        # makes the intent explicit: this model is intentionally stochastic
        # at inference time.
        self._model.train()

        self._scaler = joblib.load(scaler_path)
        self._mc_passes = mc_passes
        self._std_threshold_kg = std_threshold_kg

    def predict(self, features: FeatureVector) -> WeightEstimationResult:
        """Run Monte Carlo Dropout inference and return a weight estimate with CI.

        Scales the feature vector, runs ``mc_passes`` stochastic forward passes
        through the MLP with dropout active, then summarises the resulting
        distribution as a mean point estimate, a 95% confidence interval, and
        a standard deviation that signals epistemic uncertainty.

        A high ``prediction_std`` means the 50 passes disagreed substantially,
        which indicates the input feature vector is in a region of the training
        distribution the model has seen infrequently — not merely that the
        measurement is hard.  When ``low_confidence`` is ``True``, the point
        estimate should be surfaced with an explicit user-facing warning rather
        than presented as a normal prediction, because even the mean may not be
        a reliable central tendency if the distribution is multimodal.

        Args:
            features: The :class:`~pipeline.feature_engineer.FeatureVector`
                produced by Stage 2.  Must contain exactly
                :data:`~model.mlp.IN_FEATURES` features in the order defined
                by :meth:`~pipeline.feature_engineer.FeatureVector.to_list`.

        Returns:
            A :class:`WeightEstimationResult` containing the point estimate,
            confidence interval bounds, raw standard deviation,
            low-confidence flag, and the number of MC passes used.
        """
        scaled_tensor = self._prepare_input(features)

        with torch.no_grad():
            # torch.no_grad() suppresses gradient tracking for efficiency.
            # It does not affect dropout — _MCDropout is unconditionally active.
            passes = [
                self._model(scaled_tensor).item()
                for _ in range(self._mc_passes)
            ]

        predictions = np.array(passes, dtype=np.float64)
        mean = float(np.mean(predictions))
        std = float(np.std(predictions))

        return WeightEstimationResult(
            estimated_weight_kg=mean,
            confidence_interval_low=mean - _CI_Z_95 * std,
            confidence_interval_high=mean + _CI_Z_95 * std,
            prediction_std=std,
            low_confidence=std > self._std_threshold_kg,
            mc_passes_used=self._mc_passes,
        )

    def _prepare_input(self, features: FeatureVector) -> torch.Tensor:
        """Convert a FeatureVector to a scaled float tensor ready for the MLP.

        Applies the same ``sklearn`` scaler that was fitted on the training data
        so that each feature has zero mean and unit variance from the model's
        perspective.  Skipping this step, or applying a different scaler, will
        produce incorrect predictions.

        Args:
            features: The feature vector from Stage 2.

        Returns:
            A float32 tensor of shape ``(1, in_features)`` on CPU.
        """
        raw = np.array(features.to_list(), dtype=np.float32).reshape(1, -1)
        scaled = self._scaler.transform(raw).astype(np.float32)
        return torch.from_numpy(scaled)
