"""Evaluation script for a trained BioScan AI checkpoint.

Runs inference through the production WeightEstimator (not by importing
BioScanMLP directly) so the exact artifacts used in production are validated.
Computes MAE, RMSE, and 95% CI calibration, and exits with code 1 if MAE
exceeds 6 kg so a bad model cannot be silently deployed.

Usage:
    python training/evaluate.py \\
        --checkpoint model/checkpoints/bioscan_model.pth \\
        --data training/data/raw/test_data.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.feature_engineer import FeatureVector
from pipeline.weight_estimator import WeightEstimator
from training.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# MAE threshold above which the model is considered too inaccurate to deploy.
_MAE_QUALITY_GATE_KG: float = 6.0


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        Namespace with ``checkpoint`` (Path) and ``data`` (Path) attributes.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BioScan AI checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to bioscan_model.pth produced by training/train.py.",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Path to a CSV containing feature columns and the target column "
            "(weight_kg). Must match the column schema defined in Config."
        ),
    )
    return parser.parse_args()


def _compute_metrics(
    true_weights: list[float],
    results: list,
) -> tuple[float, float, float]:
    """Compute MAE, RMSE, and 95% CI calibration from paired predictions.

    CI calibration is the fraction of true weights that fall within the
    predicted 95% confidence interval.  A well-calibrated model should
    produce a value close to 0.95.  A significantly lower value means the
    confidence intervals are too narrow (overconfident); higher means they
    are too wide.

    Args:
        true_weights: Ground-truth weight in kg for each sample.
        results: List of :class:`~pipeline.weight_estimator.WeightEstimationResult`
            objects, one per sample, in the same order as ``true_weights``.

    Returns:
        A tuple of ``(mae, rmse, ci_calibration)`` as floats.
    """
    errors = [
        abs(true - result.estimated_weight_kg)
        for true, result in zip(true_weights, results)
    ]
    squared_errors = [e ** 2 for e in errors]
    in_ci = [
        result.confidence_interval_low <= true <= result.confidence_interval_high
        for true, result in zip(true_weights, results)
    ]

    mae = float(np.mean(errors))
    rmse = float(math.sqrt(np.mean(squared_errors)))
    ci_calibration = float(np.mean(in_ci))

    return mae, rmse, ci_calibration


def main() -> None:
    """Run evaluation and exit with code 1 if the quality gate fails."""
    args = _parse_args()
    cfg = Config()

    # -------------------------------------------------------------------------
    # Resolve scaler path from checkpoint directory — they are always co-located
    # -------------------------------------------------------------------------
    scaler_path = args.checkpoint.parent / "bioscan_scaler.pkl"

    # -------------------------------------------------------------------------
    # Load WeightEstimator — this validates checkpoint format and in_features
    # -------------------------------------------------------------------------
    log.info("Loading WeightEstimator from %s", args.checkpoint)
    try:
        estimator = WeightEstimator(
            checkpoint_path=args.checkpoint,
            scaler_path=scaler_path,
            std_threshold_kg=cfg.std_threshold_kg,
        )
    except RuntimeError as exc:
        log.error("Failed to load checkpoint: %s", exc)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Load evaluation data
    # -------------------------------------------------------------------------
    log.info("Loading evaluation data from %s", args.data)
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        log.error("Evaluation data not found: %s", args.data)
        sys.exit(1)

    required_columns = cfg.feature_columns + [cfg.target_column]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        log.error(
            "Evaluation CSV is missing required columns: %s. "
            "Available: %s",
            missing,
            list(df.columns),
        )
        sys.exit(1)

    log.info("%d evaluation samples loaded", len(df))
    log.info(
        "Running %d MC passes per sample — this may take a moment for large sets",
        50,
    )

    # -------------------------------------------------------------------------
    # Run inference through WeightEstimator (not BioScanMLP directly)
    # -------------------------------------------------------------------------
    true_weights: list[float] = []
    results = []

    for row_idx, row in df.iterrows():
        # Construct FeatureVector from raw (unscaled) row values.
        # Config.feature_columns names match FeatureVector field names exactly.
        feature_vector = FeatureVector(
            **{col: float(row[col]) for col in cfg.feature_columns}
        )
        result = estimator.predict(feature_vector)
        true_weight = float(row[cfg.target_column])

        true_weights.append(true_weight)
        results.append(result)

        log.debug(
            "Sample %s | true=%.2f kg | predicted=%.2f kg | "
            "ci=[%.2f, %.2f] | std=%.2f | low_confidence=%s",
            row_idx,
            true_weight,
            result.estimated_weight_kg,
            result.confidence_interval_low,
            result.confidence_interval_high,
            result.prediction_std,
            result.low_confidence,
        )

    # -------------------------------------------------------------------------
    # Compute and log metrics
    # -------------------------------------------------------------------------
    mae, rmse, ci_calibration = _compute_metrics(true_weights, results)
    low_confidence_count = sum(r.low_confidence for r in results)

    log.info("--- Evaluation Results ---")
    log.info("Samples evaluated    : %d", len(results))
    log.info("MAE                  : %.3f kg", mae)
    log.info("RMSE                 : %.3f kg", rmse)
    log.info(
        "CI calibration (95%%) : %.3f  (target ~0.95)", ci_calibration
    )
    log.info(
        "Low-confidence flags : %d / %d  (%.1f%%)",
        low_confidence_count,
        len(results),
        100.0 * low_confidence_count / len(results) if results else 0.0,
    )

    # -------------------------------------------------------------------------
    # Quality gate
    # -------------------------------------------------------------------------
    if mae > _MAE_QUALITY_GATE_KG:
        log.error(
            "QUALITY GATE FAILED: MAE=%.3f kg exceeds threshold of %.1f kg. "
            "This model should not be deployed. Retrain with more data or "
            "adjusted hyperparameters.",
            mae,
            _MAE_QUALITY_GATE_KG,
        )
        sys.exit(1)

    log.info(
        "Quality gate passed: MAE=%.3f kg is within the %.1f kg threshold.",
        mae,
        _MAE_QUALITY_GATE_KG,
    )


if __name__ == "__main__":
    main()
