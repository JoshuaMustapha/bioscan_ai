"""Central configuration for BioScan AI training and inference.

Import this module wherever hyperparameters, paths, or the feature list are
needed.  Keeping all settings in one dataclass prevents magic numbers from
spreading across files and makes the training-to-inference contract explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """All hyperparameters, paths, and structural constants for BioScan AI.

    Instantiate with defaults for a standard training run, or override
    individual fields for experiments::

        cfg = Config(learning_rate=0.0005, epochs=200)

    **Feature order contract**

    ``feature_columns`` defines the canonical order of the feature vector.
    It must match ``FeatureVector.to_list()`` in
    ``pipeline/feature_engineer.py`` exactly.  Changing the order in one
    place without changing the other will silently misalign training and
    inference features.

    **Architecture contract**

    ``hidden_sizes`` documents the intended MLP architecture.  It is not
    consumed directly by ``BioScanMLP`` (whose layer sizes are fixed in
    ``model/mlp.py``) but is recorded in the saved checkpoint so that the
    architecture used for a given checkpoint is always recoverable.
    ``training/train.py`` is responsible for ensuring the model instantiated
    there matches these values.

    Attributes:
        in_features: Number of input features.  Must equal
            ``len(feature_columns)`` and match ``model.mlp.IN_FEATURES``.
        hidden_sizes: MLP hidden layer widths in order.  Documents the
            architecture — see note above.
        dropout_p: Dropout probability passed to ``BioScanMLP``.  Also
            controls ``_MCDropout`` intensity at inference time.
        learning_rate: Initial learning rate for the Adam optimiser.
        batch_size: Number of samples per training mini-batch.
        epochs: Maximum number of training epochs before stopping.
        std_threshold_kg: MC Dropout standard deviation threshold above
            which a prediction is flagged ``low_confidence``.  Passed to
            ``WeightEstimator`` at API startup.
        feature_columns: Ordered list of column names to select from the
            training DataFrame.  Order is the contract with inference.
        target_column: Name of the weight label column in the DataFrame.
        ansur_csv_path: Path to the ANSUR II measurements CSV.
        smpl_csv_path: Path to the SMPL synthetic dataset CSV (feature
            vectors pre-extracted from rendered images).
        checkpoint_dir: Directory where ``bioscan_model.pth`` and
            ``bioscan_scaler.pkl`` are written after training.
        scaler_path: Full path for the serialised scaler file.  Must be
            inside or alongside ``checkpoint_dir`` so both artefacts are
            deployed together.
    """

    # ---- Model architecture -----------------------------------------------
    in_features: int = 10
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_p: float = 0.3

    # ---- Training ------------------------------------------------------------
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

    # ---- Uncertainty ---------------------------------------------------------
    std_threshold_kg: float = 5.0

    # ---- Feature definition --------------------------------------------------
    # CRITICAL: order must match FeatureVector.to_list() in
    # pipeline/feature_engineer.py. Do not reorder without updating both.
    feature_columns: list[str] = field(default_factory=lambda: [
        "shoulder_width",
        "hip_width",
        "shoulder_to_hip_ratio",
        "torso_height",
        "silhouette_area",
        "left_arm_length",
        "right_arm_length",
        "leg_length",
        "height_cm",
        "age",
    ])
    target_column: str = "weight_kg"

    # ---- Dataset paths -------------------------------------------------------
    ansur_csv_path: Path = field(
        default_factory=lambda: Path("training/data/raw/ansur.csv")
    )
    smpl_csv_path: Path = field(
        default_factory=lambda: Path("training/data/raw/smpl.csv")
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("model/checkpoints")
    )
    scaler_path: Path = field(
        default_factory=lambda: Path("model/checkpoints/bioscan_scaler.pkl")
    )

    def __post_init__(self) -> None:
        """Validate internal consistency at construction time."""
        if len(self.feature_columns) != self.in_features:
            raise ValueError(
                f"len(feature_columns)={len(self.feature_columns)} does not "
                f"match in_features={self.in_features}. Update both together."
            )
