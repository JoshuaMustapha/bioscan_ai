"""PyTorch MLP architecture for BioScan AI body weight estimation.

The model uses a custom MCDropout layer rather than standard nn.Dropout.
Standard dropout is disabled when the model is in eval() mode, which would
make every inference pass identical and defeat uncertainty estimation entirely.
MCDropout overrides that behaviour so dropout is always active regardless of
model mode, enabling Monte Carlo Dropout: running N stochastic forward passes
and treating their spread as a proxy for prediction uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match the length of FeatureVector.to_list() exactly.
# If features are added or removed from Stage 2, this constant must be updated.
IN_FEATURES: int = 10


class _MCDropout(nn.Dropout):
    """Dropout that remains active during inference for Monte Carlo sampling.

    Standard ``nn.Dropout`` checks the module's ``training`` flag and becomes
    a no-op when the model is in ``eval()`` mode.  This subclass bypasses that
    check by always passing ``training=True`` to ``F.dropout``, ensuring that
    neurons are randomly dropped on every forward pass regardless of the model's
    mode.

    This is the mechanism that makes Monte Carlo Dropout work: repeated forward
    passes through the same input produce different outputs, and the distribution
    of those outputs encodes the model's uncertainty about its prediction.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply dropout unconditionally, ignoring the model's training flag.

        Args:
            input: Input tensor of any shape.

        Returns:
            Tensor of the same shape with random elements zeroed and the
            remainder scaled by ``1 / (1 - p)`` to preserve expected values.
        """
        return F.dropout(input, self.p, training=True, inplace=self.inplace)


class BioScanMLP(nn.Module):
    """Multi-Layer Perceptron that estimates body weight from anthropometric features.

    Takes a scaled 10-feature vector produced by ``pipeline/feature_engineer.py``
    and outputs a single scalar: the estimated weight in kilograms.

    Dropout layers use :class:`_MCDropout` so they remain active at inference
    time, enabling Monte Carlo uncertainty estimation without any changes to
    calling code.

    Architecture::

        Input  (10)  →  Linear(128) → ReLU → MCDropout(p)
                      →  Linear(64)  → ReLU → MCDropout(p)
                      →  Linear(32)  → ReLU → MCDropout(p)
                      →  Linear(1)

    Args:
        in_features: Number of input features.  Must match
            ``FeatureVector.to_list()`` length.  Defaults to
            :data:`IN_FEATURES` (10).
        dropout_p: Dropout probability applied after each hidden layer.
            Defaults to 0.3.
    """

    def __init__(
        self,
        in_features: int = IN_FEATURES,
        dropout_p: float = 0.3,
    ) -> None:
        """Build the network layers."""
        super().__init__()

        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(in_features, 128),
            nn.ReLU(),
            _MCDropout(p=dropout_p),
            # Hidden layer 2
            nn.Linear(128, 64),
            nn.ReLU(),
            _MCDropout(p=dropout_p),
            # Hidden layer 3
            nn.Linear(64, 32),
            nn.ReLU(),
            _MCDropout(p=dropout_p),
            # Output: single weight estimate in kg
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass.

        Args:
            x: Float tensor of shape ``(batch_size, in_features)``.

        Returns:
            Float tensor of shape ``(batch_size,)`` containing one weight
            estimate per sample.
        """
        return self.network(x).squeeze(-1)
