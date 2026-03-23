"""Entry point for BioScan AI model training.

Loads and merges ANSUR II and SMPL data, performs the train/val split,
fits the feature scaler on the training split only, and delegates the
training loop to Trainer.

Usage:
    python training/train.py
"""

from __future__ import annotations

import logging
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from model.dataset import BioScanDataset
from model.mlp import BioScanMLP
from model.trainer import Trainer
from training.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# BioScanMLP has fixed hidden sizes in mlp.py. Config.hidden_sizes documents
# the intended architecture and is stored in the checkpoint, but is not a
# constructor parameter. We validate alignment here so a config mismatch is
# caught at the start of a training run rather than silently ignored.
_MLP_HIDDEN_SIZES: list[int] = [128, 64, 32]


def _validate_columns(df: pd.DataFrame, path: object, required: list[str]) -> None:
    """Raise ValueError if any required column is absent from the DataFrame.

    Args:
        df: The loaded DataFrame to inspect.
        path: The file path the DataFrame was loaded from (used in the error
            message only).
        required: List of column names that must be present.

    Raises:
        ValueError: With a list of missing columns and the available columns.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV at '{path}' is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def main() -> None:
    """Run the full training pipeline."""
    t_start = time.perf_counter()
    cfg = Config()

    # -------------------------------------------------------------------------
    # Validate architecture alignment
    # -------------------------------------------------------------------------
    if cfg.hidden_sizes != _MLP_HIDDEN_SIZES:
        log.warning(
            "config.hidden_sizes=%s does not match the hardcoded BioScanMLP "
            "architecture %s. Update model/mlp.py or training/config.py to "
            "align them before training.",
            cfg.hidden_sizes,
            _MLP_HIDDEN_SIZES,
        )

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------
    required_columns = cfg.feature_columns + [cfg.target_column]

    log.info("Loading ANSUR II data from %s", cfg.ansur_csv_path)
    try:
        ansur_df = pd.read_csv(cfg.ansur_csv_path)
    except FileNotFoundError:
        log.error("ANSUR II CSV not found: %s", cfg.ansur_csv_path)
        sys.exit(1)
    _validate_columns(ansur_df, cfg.ansur_csv_path, required_columns)
    log.info("ANSUR II: %d rows loaded", len(ansur_df))

    log.info("Loading SMPL synthetic data from %s", cfg.smpl_csv_path)
    try:
        smpl_df = pd.read_csv(cfg.smpl_csv_path)
    except FileNotFoundError:
        log.error("SMPL CSV not found: %s", cfg.smpl_csv_path)
        sys.exit(1)
    _validate_columns(smpl_df, cfg.smpl_csv_path, required_columns)
    log.info("SMPL: %d rows loaded", len(smpl_df))

    # -------------------------------------------------------------------------
    # Merge with source column so origin is traceable in the combined DataFrame
    # -------------------------------------------------------------------------
    ansur_df = ansur_df.copy()
    smpl_df = smpl_df.copy()
    ansur_df["source"] = "ansur"
    smpl_df["source"] = "smpl"
    combined_df = pd.concat([ansur_df, smpl_df], ignore_index=True)
    log.info(
        "Combined dataset: %d rows (ansur=%d, smpl=%d)",
        len(combined_df),
        len(ansur_df),
        len(smpl_df),
    )

    # -------------------------------------------------------------------------
    # Train / validation split
    # -------------------------------------------------------------------------
    train_df, val_df = train_test_split(
        combined_df,
        test_size=0.2,
        random_state=42,
    )
    log.info("Split: %d train rows, %d val rows", len(train_df), len(val_df))

    # -------------------------------------------------------------------------
    # Fit scaler on X_train only — never on the full dataset or X_val
    # -------------------------------------------------------------------------
    X_train = train_df[cfg.feature_columns].values
    scaler = StandardScaler()
    scaler.fit(X_train)
    log.info(
        "StandardScaler fitted on %d training samples across %d features",
        len(X_train),
        cfg.in_features,
    )

    # -------------------------------------------------------------------------
    # Build datasets and DataLoaders
    # -------------------------------------------------------------------------
    train_dataset = BioScanDataset(train_df, scaler, cfg.feature_columns, cfg.target_column)
    val_dataset = BioScanDataset(val_df, scaler, cfg.feature_columns, cfg.target_column)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    log.info(
        "DataLoaders ready: %d train batches, %d val batches (batch_size=%d)",
        len(train_loader),
        len(val_loader),
        cfg.batch_size,
    )

    # -------------------------------------------------------------------------
    # Instantiate model
    # BioScanMLP does not accept hidden_sizes — architecture is fixed in mlp.py.
    # in_features and dropout_p are passed explicitly from config.
    # -------------------------------------------------------------------------
    model = BioScanMLP(in_features=cfg.in_features, dropout_p=cfg.dropout_p)
    log.info(
        "BioScanMLP instantiated (in_features=%d, dropout_p=%s)",
        cfg.in_features,
        cfg.dropout_p,
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    trainer = Trainer(cfg)
    best_model = trainer.train(model, train_loader, val_loader, scaler)  # noqa: F841

    elapsed = time.perf_counter() - t_start
    log.info("Training complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
