"""
Simple progress hooks with tqdm for training visibility.
"""

import logging
from typing import Dict, Optional

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProgressBarHooks:
    """
    Training hooks with tqdm progress bars.

    Shows:
    - Epoch progress with ETA
    - Loss, learning rate, accuracy every N batches
    - Validation metrics after each epoch

    Falls back to text logging if tqdm unavailable.
    """

    def __init__(self, update_freq: int = 10):
        """
        Args:
            update_freq: Update progress every N batches (default 10)
        """
        self.update_freq = update_freq
        self.use_tqdm = TQDM_AVAILABLE
        self.epoch_pbar = None
        self.batch_count = 0
        self.nan_count = 0  # Track nan losses

    def on_training_start(self) -> None:
        """Called at start of training."""
        if not self.use_tqdm:
            logger.warning("tqdm not available - using text progress")

    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of each epoch."""
        self.batch_count = 0
        if not self.use_tqdm:
            print(f"\n→ Epoch {epoch + 1} started")

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """
        Called after each training batch.

        Args:
            batch_idx: Current batch index
            loss: Batch loss value
        """
        self.batch_count += 1

        # Detect nan loss
        if loss != loss:  # nan check (nan != nan is True)
            self.nan_count += 1
            logger.warning(f"⚠️  Batch {batch_idx}: loss is NAN (nan count: {self.nan_count})")

        # Update every N batches
        if batch_idx % self.update_freq == 0:
            if self.use_tqdm and self.epoch_pbar:
                self.epoch_pbar.set_postfix({'loss': f'{loss:.4f}'})
                self.epoch_pbar.update(min(self.update_freq, self.epoch_pbar.total - self.epoch_pbar.n))
            else:
                print(f"  Batch {batch_idx}: loss={loss:.4f}")

    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        """
        Called after validation completes.

        Args:
            metrics: Validation metrics (prefixed with 'val_')
        """
        # Close epoch progress bar
        if self.use_tqdm and self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None

        # Display validation metrics
        print()
        val_loss = metrics.get('val_loss', metrics.get('loss', float('nan')))
        val_acc = metrics.get('val_accuracy', metrics.get('accuracy', float('nan')))

        print(f"  Val Loss: {val_loss:.4f}", end="")
        if not (val_acc != val_acc):  # Check if not nan
            print(f"  |  Val Acc: {val_acc:.2%}", end="")
        print()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Called after epoch completes.

        Args:
            epoch: Current epoch number
            metrics: Combined train + val metrics
        """
        if self.nan_count > 0:
            logger.warning(f"Epoch {epoch}: {self.nan_count} batches had NAN loss!")
            self.nan_count = 0  # Reset for next epoch

    def on_training_end(self) -> None:
        """Called when training completes."""
        if self.epoch_pbar:
            self.epoch_pbar.close()
