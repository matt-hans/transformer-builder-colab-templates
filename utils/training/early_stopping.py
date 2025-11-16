"""
Early stopping monitoring and W&B logging utilities.

Provides a lightweight monitor for validation metrics and a Lightning
callback that logs early stopping events to W&B while delegating the
actual stopping to PyTorch Lightning's EarlyStopping callback.
"""

from typing import Optional, Literal

try:
    import pytorch_lightning as pl  # noqa: F401
    from pytorch_lightning.callbacks import Callback  # type: ignore
except Exception:
    class Callback:  # type: ignore
        """Fallback Callback stub when Lightning not installed."""
        pass


class EarlyStoppingMonitor:
    """
    Track validation metric improvements with patience/min_delta.

    This class does not stop training itself; it just tracks state.
    """

    def __init__(self,
                 patience: int = 5,
                 min_delta: float = 0.0,
                 mode: Literal['min', 'max'] = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        self.triggered = False

    def update(self, current_metric: float) -> tuple[bool, bool]:
        improved = False
        if self.mode == 'min':
            improved = current_metric < (self.best_metric - self.min_delta)
        else:
            improved = current_metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.triggered = True

        return improved, self.triggered


class EarlyStoppingWandbCallback(Callback):
    """
    Lightning callback that logs early stopping progress to stdout and W&B.

    Does not perform stopping; use PyTorch Lightning EarlyStopping for that.
    """

    def __init__(self,
                 patience: int = 5,
                 min_delta: float = 0.0,
                 mode: Literal['min', 'max'] = 'min'):
        super().__init__()
        self.monitor = EarlyStoppingMonitor(patience=patience, min_delta=min_delta, mode=mode)
        self._logged_event = False

    def _maybe_log_to_wandb(self, epoch: int):
        try:
            import wandb  # type: ignore
            if getattr(wandb, 'run', None):
                wandb.log({
                    'events/early_stopping_epoch': epoch,
                    'events/early_stopping_triggered': 1,
                    'metrics/best_val_loss': self.monitor.best_metric
                }, step=epoch)
        except Exception:
            # W&B not available or not initialized; ignore
            pass

    def on_validation_end(self, trainer, pl_module):  # type: ignore[override]
        # Extract validation loss from callback_metrics if available
        metrics = getattr(trainer, 'callback_metrics', {}) or {}
        val_loss = metrics.get('val_loss', None)
        train_loss = metrics.get('train_loss', metrics.get('train_loss_epoch', None))
        if val_loss is None:
            return
        try:
            # Tensor-like to float
            current = float(getattr(val_loss, 'item', lambda: val_loss)()) if hasattr(val_loss, 'item') else float(val_loss)
        except Exception:
            return

        improved, triggered = self.monitor.update(current)

        epoch = getattr(trainer, 'current_epoch', 0)
        if improved:
            print(f"✅ EarlyStopping: val_loss improved to {current:.4f} at epoch {epoch}")
        else:
            print(f"⚠️ EarlyStopping: no improvement ({self.monitor.epochs_without_improvement}/{self.monitor.patience}) — best={self.monitor.best_metric:.4f}")

        # Intentionally avoid per-epoch W&B logging here to keep this
        # callback focused on early-stop events only. Tests expect no
        # wandb.log calls until the early stopping condition triggers.

        if triggered and not self._logged_event:
            print(f"🛑 EarlyStopping: patience exceeded — logging event at epoch {epoch}")
            self._maybe_log_to_wandb(epoch)
            self._logged_event = True
