"""PyTorch Dataset for BioScan AI training data.

Wraps a pandas DataFrame of anthropometric features and weight labels,
applying a pre-fitted scaler so that the DataLoader yields scaled tensors
ready for the MLP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BioScanDataset(Dataset):
    """Maps rows of a feature DataFrame to scaled (features, weight) tensor pairs.

    .. warning::
        **The scaler must be fitted on training data only before being passed
        in.**  This class calls ``scaler.transform()`` — never ``fit()`` or
        ``fit_transform()``.  Fitting the scaler inside this class, or passing
        a scaler that was fitted after the train/validation split, leaks
        validation statistics into the scaling transform and invalidates all
        evaluation metrics.  Always fit the scaler on the training split
        exclusively, then pass the same fitted instance to the training and
        validation dataset constructors.

    Args:
        df: DataFrame containing at least ``feature_columns`` and
            ``target_column``.  Rows should already be filtered and cleaned
            (no NaN values in the selected columns).
        scaler: A fitted ``sklearn`` scaler (e.g. ``StandardScaler``) that
            exposes a ``transform(X)`` method.  Must have been fitted on
            training-split feature values only.
        feature_columns: Ordered list of column names to select as the
            feature vector.  Order must match ``FeatureVector.to_list()``
            and ``Config.feature_columns``.
        target_column: Name of the column containing ground-truth weight
            labels in kilograms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        """Pre-scale all features and convert to tensors at construction time.

        Pre-scaling in ``__init__`` rather than in ``__getitem__`` means the
        transform is applied once per dataset construction rather than once
        per sample per epoch, which is significantly faster for large datasets.
        """
        raw_features = df[feature_columns].values.astype(np.float32)
        scaled_features = scaler.transform(raw_features).astype(np.float32)

        self._X = torch.from_numpy(scaled_features)
        self._y = torch.from_numpy(
            df[target_column].values.astype(np.float32)
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the scaled feature tensor and weight label for one sample.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A tuple of ``(feature_tensor, weight_tensor)`` where
            ``feature_tensor`` has shape ``(in_features,)`` and
            ``weight_tensor`` is a scalar float32 tensor containing the
            ground-truth weight in kilograms.
        """
        return self._X[idx], self._y[idx]
