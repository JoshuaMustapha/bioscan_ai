"""Training loop for BioScanMLP.

Handles forward passes, loss computation, backpropagation, learning rate
scheduling, early stopping, and checkpoint saving.  Accepts a Config object
so all hyperparameters come from a single source of truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.mlp import BioScanMLP
from training.config import Config


class Trainer:
    """Runs the BioScanMLP training loop with early stopping and LR scheduling.

    Instantiate once per training run, then call :meth:`train`.  The Trainer
    does not instantiate the model or the DataLoaders — those are constructed
    in ``training/train.py`` and passed in, keeping this class focused purely
    on the training loop.

    Args:
        config: The :class:`~training.config.Config` instance that governs all
            hyperparameters, paths, and structural constants for this run.
    """

    def __init__(self, config: Config) -> None:
        """Store config and initialise the module logger."""
        self._cfg = config
        self._log = logging.getLogger(__name__)

    def train(
        self,
        model: BioScanMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scaler,
    ) -> BioScanMLP:
        """Run the full training loop and return the best model.

        Trains for up to ``config.epochs`` epochs.  Applies
        ``ReduceLROnPlateau`` scheduling (factor 0.5, patience 10 epochs) and
        stops early if validation loss does not improve for 20 consecutive
        epochs.  The best model weights (lowest validation loss) are restored
        before returning, then saved to disk alongside the scaler.

        .. note::
            Because ``_MCDropout`` is always active regardless of model mode,
            validation loss computed here is stochastic — each validation pass
            uses a different dropout mask.  This adds minor variance to the
            reported val loss per epoch but does not bias the trend, so early
            stopping and LR scheduling still converge correctly over many epochs.

        Args:
            model: An untrained (or partially trained) :class:`~model.mlp.BioScanMLP`
                instance.  Architecture must match ``config.in_features`` and
                ``config.dropout_p``.
            train_loader: DataLoader wrapping a :class:`~model.dataset.BioScanDataset`
                built from the training split.
            val_loader: DataLoader wrapping a :class:`~model.dataset.BioScanDataset`
                built from the validation split.
            scaler: The fitted ``sklearn`` scaler used to construct both
                DataLoaders.  Saved to disk at the end of training so that
                inference can apply the identical transform.

        Returns:
            The ``model`` with its weights restored to the epoch that achieved
            the lowest validation loss.
        """
        cfg = self._cfg

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )

        best_val_loss: float = float("inf")
        epochs_without_improvement: int = 0
        # Deep-copy the state dict so in-place weight updates don't corrupt it.
        best_state: dict = _clone_state(model.state_dict())

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._val_epoch(model, val_loader, criterion)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self._log.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                epoch,
                cfg.epochs,
                train_loss,
                val_loss,
                current_lr,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = _clone_state(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 20:
                    self._log.info(
                        "Early stopping: val_loss did not improve for 20 "
                        "consecutive epochs. Stopping at epoch %d.",
                        epoch,
                    )
                    break

        # Restore the best weights before saving and returning.
        model.load_state_dict(best_state)
        self._save_checkpoint(model, scaler, best_val_loss)
        return model

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _train_epoch(
        self,
        model: BioScanMLP,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch and return mean MSE loss over all samples.

        Gradients are computed and weights updated for every mini-batch.
        Loss is accumulated as a sample-weighted mean so batches of different
        sizes contribute proportionally.

        Args:
            model: The model to train.
            loader: DataLoader for the training split.
            criterion: Loss function (MSELoss).
            optimizer: Adam optimiser, already initialised with model parameters.

        Returns:
            Mean training MSE loss across all samples in the epoch.
        """
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)

        return total_loss / total_samples

    def _val_epoch(
        self,
        model: BioScanMLP,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Run one validation epoch and return mean MSE loss over all samples.

        Gradient computation is disabled for memory and speed efficiency.
        Dropout remains active (``_MCDropout`` ignores model mode), so the
        validation loss has minor stochastic variance — see :meth:`train`
        for discussion of why this is acceptable.

        Args:
            model: The model to evaluate.
            loader: DataLoader for the validation split.
            criterion: Loss function (MSELoss).

        Returns:
            Mean validation MSE loss across all samples in the epoch.
        """
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                total_loss += loss.item() * len(X_batch)
                total_samples += len(X_batch)

        return total_loss / total_samples

    def _save_checkpoint(
        self,
        model: BioScanMLP,
        scaler,
        best_val_loss: float,
    ) -> None:
        """Save the model checkpoint and fitted scaler to disk.

        Writes two files to ``config.checkpoint_dir``:

        * ``bioscan_model.pth`` — PyTorch state dict plus config metadata.
          Config metadata is stored so that the architecture and feature
          contract used for this checkpoint are always recoverable.

        .. warning::
            ``pipeline/weight_estimator.py`` currently calls
            ``model.load_state_dict(torch.load(path))`` and expects the
            ``.pth`` file to be a plain state dict.  Because this checkpoint
            includes config metadata, ``weight_estimator.py`` must be updated
            to extract the state dict with::

                checkpoint = torch.load(path, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])

        * ``bioscan_scaler.pkl`` — the fitted sklearn scaler, serialised with
          joblib.  Must be deployed alongside ``bioscan_model.pth``.

        Args:
            model: The model with best weights already restored.
            scaler: The fitted scaler used during training.
            best_val_loss: Best validation MSE loss achieved, stored in the
                checkpoint for reference.
        """
        cfg = self._cfg
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "bioscan_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "in_features": cfg.in_features,
                    "hidden_sizes": cfg.hidden_sizes,
                    "dropout_p": cfg.dropout_p,
                    "feature_columns": cfg.feature_columns,
                    "target_column": cfg.target_column,
                    "std_threshold_kg": cfg.std_threshold_kg,
                },
                "best_val_loss": best_val_loss,
            },
            checkpoint_path,
        )
        self._log.info("Saved model checkpoint → %s", checkpoint_path)

        scaler_path = Path(cfg.scaler_path)
        joblib.dump(scaler, scaler_path)
        self._log.info("Saved feature scaler → %s", scaler_path)


# -----------------------------------------------------------------------------
# Module-level helper
# -----------------------------------------------------------------------------


def _clone_state(state_dict: dict) -> dict:
    """Return a deep copy of a model state dict.

    ``model.state_dict()`` returns references to the live parameter tensors.
    Storing that reference directly as ``best_state`` means every subsequent
    weight update would silently overwrite what we think we saved.  Cloning
    each tensor ensures the snapshot is independent of future updates.

    Args:
        state_dict: The state dict returned by ``model.state_dict()``.

    Returns:
        A new dict with cloned tensors for every parameter.
    """
    return {k: v.clone() for k, v in state_dict.items()}
