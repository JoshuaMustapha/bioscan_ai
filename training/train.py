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

# Raw ANSUR II columns consumed by _map_ansur_to_features.  All linear
# measurements are in millimetres; stature is used as the normalisation
# denominator.
_ANSUR_RAW_COLUMNS: list[str] = [
    "biacromialbreadth",
    "hipbreadth",
    "acromialheight",
    "iliocristaleheight",
    "acromionradialelength",
    "crotchheight",
    "stature",
    "Age",
    "weightkg",
]


def _map_ansur_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map ANSUR II raw measurement columns to the BioScan AI feature vector.

    All linear measurements in ANSUR II are in millimetres.  Dividing by
    stature produces dimensionless ratios that mirror what the MediaPipe
    pipeline produces at inference time (landmark coordinates normalised by
    image dimensions / torso height).

    ``df`` must already contain an injected ``gender`` column (1 = male,
    0 = female) before this function is called.

    Args:
        df: ANSUR II DataFrame with the columns in ``_ANSUR_RAW_COLUMNS``
            plus an already-injected ``gender`` column.

    Returns:
        A new DataFrame whose columns are exactly ``Config.feature_columns``
        plus ``weight_kg`` (the target).
    """
    stature = df["stature"]  # mm

    shoulder_width = df["biacromialbreadth"] / stature
    hip_width = df["hipbreadth"] / stature
    torso_height = (df["acromialheight"] - df["iliocristaleheight"]) / stature
    arm_length = df["acromionradialelength"] / stature
    leg_length = df["crotchheight"] / stature

    return pd.DataFrame(
        {
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "shoulder_to_hip_ratio": df["biacromialbreadth"] / df["hipbreadth"],
            "torso_height": torso_height,
            "silhouette_area": shoulder_width * torso_height,
            "left_arm_length": arm_length,
            "right_arm_length": arm_length,
            "leg_length": leg_length,
            "height_cm": stature / 10.0,
            "age": df["Age"],
            "gender": df["gender"],
            "weight_kg": df["weightkg"],
        },
        index=df.index,
    )


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
    # ANSUR II CSVs contain raw anthropometric measurements (mm), not the
    # derived feature vector.  Validate against _ANSUR_RAW_COLUMNS, then call
    # _map_ansur_to_features to produce the canonical feature DataFrame.
    # SMPL CSVs already use feature column names, so they are validated against
    # the feature list directly (gender excluded — it may be absent or added).
    smpl_required = [c for c in cfg.feature_columns if c != "gender"] + [cfg.target_column]

    log.info("Loading ANSUR II male data from %s", cfg.ansur_male_csv_path)
    try:
        ansur_male_df = pd.read_csv(cfg.ansur_male_csv_path, encoding="latin-1")
    except FileNotFoundError:
        log.error("ANSUR II male CSV not found: %s", cfg.ansur_male_csv_path)
        sys.exit(1)
    _validate_columns(ansur_male_df, cfg.ansur_male_csv_path, _ANSUR_RAW_COLUMNS)
    ansur_male_df = ansur_male_df.copy()
    ansur_male_df["gender"] = 1
    ansur_male_df = _map_ansur_to_features(ansur_male_df)
    log.info("ANSUR II male: %d rows loaded and mapped", len(ansur_male_df))

    log.info("Loading ANSUR II female data from %s", cfg.ansur_female_csv_path)
    try:
        ansur_female_df = pd.read_csv(cfg.ansur_female_csv_path, encoding="latin-1")
    except FileNotFoundError:
        log.error("ANSUR II female CSV not found: %s", cfg.ansur_female_csv_path)
        sys.exit(1)
    _validate_columns(ansur_female_df, cfg.ansur_female_csv_path, _ANSUR_RAW_COLUMNS)
    ansur_female_df = ansur_female_df.copy()
    ansur_female_df["gender"] = 0
    ansur_female_df = _map_ansur_to_features(ansur_female_df)
    log.info("ANSUR II female: %d rows loaded and mapped", len(ansur_female_df))

    log.info("Loading SMPL synthetic data from %s", cfg.smpl_csv_path)
    smpl_df: pd.DataFrame | None = None
    try:
        smpl_df = pd.read_csv(cfg.smpl_csv_path)
        _validate_columns(smpl_df, cfg.smpl_csv_path, smpl_required)
        smpl_df = smpl_df.copy()
        # SMPL synthetic data does not carry a definitive biological sex label;
        # default to 0 so the column is present. Replace with real labels if the
        # SMPL dataset includes them.
        smpl_df.setdefault("gender", 0)
        log.info("SMPL: %d rows loaded", len(smpl_df))
    except FileNotFoundError:
        log.warning(
            "SMPL CSV not found: %s — continuing with ANSUR II data only.",
            cfg.smpl_csv_path,
        )

    # -------------------------------------------------------------------------
    # Merge with source tag
    # -------------------------------------------------------------------------
    ansur_df = pd.concat([ansur_male_df, ansur_female_df], ignore_index=True)
    ansur_df["source"] = "ansur"

    if smpl_df is not None:
        smpl_df["source"] = "smpl"
        combined_df = pd.concat([ansur_df, smpl_df], ignore_index=True)
        log.info(
            "Combined dataset: %d rows (ansur_male=%d, ansur_female=%d, smpl=%d)",
            len(combined_df),
            len(ansur_male_df),
            len(ansur_female_df),
            len(smpl_df),
        )
    else:
        combined_df = ansur_df
        log.info(
            "Combined dataset: %d rows (ansur_male=%d, ansur_female=%d, smpl=0)",
            len(combined_df),
            len(ansur_male_df),
            len(ansur_female_df),
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
