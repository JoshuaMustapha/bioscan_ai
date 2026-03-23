"""Unit tests for BioScanMLP and WeightEstimator.

BioScanMLP tests verify output shape, the MC Dropout contract (dropout must
remain active in eval mode), and the IN_FEATURES / FeatureVector contract.

WeightEstimator tests use a real saved checkpoint written to tmp_path so
that the full load → scale → MC inference path is exercised, but without
depending on any pre-trained weights on disk.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from model.mlp import IN_FEATURES, BioScanMLP
from pipeline.feature_engineer import FeatureVector
from pipeline.weight_estimator import WeightEstimationResult, WeightEstimator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_feature_vector() -> FeatureVector:
    """A plausible FeatureVector for a 175 cm, 30-year-old subject."""
    shoulder_width = 0.30
    hip_width = 0.22
    return FeatureVector(
        shoulder_width=shoulder_width,
        hip_width=hip_width,
        shoulder_to_hip_ratio=shoulder_width / hip_width,
        torso_height=0.28,
        silhouette_area=shoulder_width * 0.28,
        left_arm_length=0.31,
        right_arm_length=0.30,
        leg_length=0.42,
        height_cm=175.0,
        age=30.0,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mlp() -> BioScanMLP:
    """Fresh BioScanMLP with default hyperparameters."""
    return BioScanMLP()


@pytest.fixture
def saved_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Write a real BioScanMLP checkpoint and a fitted StandardScaler to tmp_path.

    Returns ``(checkpoint_path, scaler_path)`` so tests can pass them directly
    to WeightEstimator.  Weights are random (untrained) — these tests only
    verify the inference contract, not prediction quality.
    """
    # ── Checkpoint ──────────────────────────────────────────────────────────
    model = BioScanMLP()
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "in_features": IN_FEATURES,
            "hidden_sizes": [128, 64, 32],
            "dropout_p": 0.3,
        },
        "best_val_loss": 9.99,
    }
    ckpt_path = tmp_path / "bioscan_model.pth"
    torch.save(ckpt, ckpt_path)

    # ── Scaler ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, IN_FEATURES))
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler_path = tmp_path / "bioscan_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    return ckpt_path, scaler_path


@pytest.fixture
def estimator(saved_checkpoint: tuple[Path, Path]) -> WeightEstimator:
    """WeightEstimator loaded from the tmp_path checkpoint."""
    ckpt_path, scaler_path = saved_checkpoint
    return WeightEstimator(
        checkpoint_path=ckpt_path,
        scaler_path=scaler_path,
        mc_passes=50,
    )


# ── BioScanMLP tests ──────────────────────────────────────────────────────────


def test_mlp_forward_batch_size_1_returns_shape_1(mlp: BioScanMLP):
    x = torch.randn(1, IN_FEATURES)
    out = mlp(x)
    assert out.shape == (1,)


def test_mlp_forward_batch_size_8_returns_shape_8(mlp: BioScanMLP):
    x = torch.randn(8, IN_FEATURES)
    out = mlp(x)
    assert out.shape == (8,)


def test_mc_dropout_remains_active_in_eval_mode(mlp: BioScanMLP):
    """_MCDropout must fire regardless of model.eval() — this is the core MC contract."""
    mlp.eval()
    x = torch.randn(1, IN_FEATURES)

    with torch.no_grad():
        outputs = [mlp(x).item() for _ in range(100)]

    std = float(np.std(outputs))
    assert std > 0.0, (
        "All 100 forward passes in eval() mode returned identical values — "
        "_MCDropout appears to be disabled.  Check that _MCDropout passes "
        "training=True to F.dropout unconditionally."
    )


def test_in_features_constant_matches_feature_vector_length():
    """IN_FEATURES must equal the length of FeatureVector.to_list() exactly."""
    fv = _make_feature_vector()
    assert len(fv.to_list()) == IN_FEATURES


# ── WeightEstimator tests ─────────────────────────────────────────────────────


def test_estimator_predict_returns_all_6_result_fields(estimator: WeightEstimator):
    result = estimator.predict(_make_feature_vector())

    assert isinstance(result, WeightEstimationResult)
    assert isinstance(result.estimated_weight_kg, float)
    assert isinstance(result.confidence_interval_low, float)
    assert isinstance(result.confidence_interval_high, float)
    assert isinstance(result.prediction_std, float)
    assert isinstance(result.low_confidence, bool)
    assert isinstance(result.mc_passes_used, int)
    assert result.mc_passes_used == 50


def test_estimator_ci_bounds_contain_point_estimate(estimator: WeightEstimator):
    """The 95% CI must strictly bracket the mean estimate when std > 0."""
    result = estimator.predict(_make_feature_vector())

    assert result.confidence_interval_low < result.estimated_weight_kg, (
        "CI lower bound must be less than the point estimate."
    )
    assert result.estimated_weight_kg < result.confidence_interval_high, (
        "Point estimate must be less than CI upper bound."
    )


def test_low_confidence_flag_set_when_std_exceeds_threshold(
    saved_checkpoint: tuple[Path, Path],
):
    """Any MC variance must trigger low_confidence when threshold is 0."""
    ckpt_path, scaler_path = saved_checkpoint
    # std_threshold_kg=0.0 means any non-zero MC std sets low_confidence=True.
    # _MCDropout guarantees std > 0 across 50 stochastic passes.
    strict_estimator = WeightEstimator(
        checkpoint_path=ckpt_path,
        scaler_path=scaler_path,
        mc_passes=50,
        std_threshold_kg=0.0,
    )
    result = strict_estimator.predict(_make_feature_vector())
    assert result.low_confidence is True


def test_missing_checkpoint_raises_runtime_error_at_construction(
    tmp_path: Path,
    saved_checkpoint: tuple[Path, Path],
):
    """Construction must fail immediately if the checkpoint file is absent."""
    _, scaler_path = saved_checkpoint
    missing_ckpt = tmp_path / "does_not_exist.pth"

    with pytest.raises(RuntimeError, match="checkpoint"):
        WeightEstimator(checkpoint_path=missing_ckpt, scaler_path=scaler_path)
