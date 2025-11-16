"""
Comprehensive metrics tracking for transformer training with W&B integration.

This module provides the MetricsTracker class for tracking and logging:
- Loss (train/validation)
- Perplexity (exp(loss))
- Accuracy (next-token prediction)
- Learning rate
- Gradient norms
- Epoch duration
- System metrics (GPU memory/utilization)

Supports both online (W&B) and offline (local storage) modes with error resilience.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Literal


class MetricsTracker:
    """
    Comprehensive metrics tracking for transformer training.

    Tracks and logs training metrics to W&B and/or local storage. Handles
    perplexity computation with overflow protection, accuracy calculation
    with padding support, and system metrics collection.

    Args:
        use_wandb: Whether to log metrics to W&B (default: True)

    Attributes:
        use_wandb: Whether W&B logging is enabled
        metrics_history: List of metric dicts for all logged epochs

    Examples:
        >>> tracker = MetricsTracker(use_wandb=True)
        >>> tracker.log_epoch(
        ...     epoch=0,
        ...     train_metrics={'loss': 2.5, 'accuracy': 0.75},
        ...     val_metrics={'loss': 2.7, 'accuracy': 0.72},
        ...     learning_rate=5e-5,
        ...     gradient_norm=0.85,
        ...     epoch_duration=120.5
        ... )
        >>> df = tracker.get_summary()
        >>> best_epoch = tracker.get_best_epoch('val/loss', 'min')
    """

    def __init__(self, use_wandb: bool = True):
        """
        Initialize metrics tracker.

        Args:
            use_wandb: Whether to enable W&B logging (default: True)
        """
        self.use_wandb = use_wandb
        self.metrics_history = []

    def compute_perplexity(self, loss: float) -> float:
        """
        Compute perplexity from cross-entropy loss.

        Perplexity = exp(loss). To prevent overflow with very high losses,
        loss is clipped at 100.0 before exponentiation. This is appropriate
        because exp(100) = 2.7e43 is already meaningless and indicates severe
        numerical instability.

        Args:
            loss: Cross-entropy loss value

        Returns:
            Perplexity value (exp of clipped loss)

        Examples:
            >>> tracker = MetricsTracker()
            >>> ppl = tracker.compute_perplexity(2.3026)  # ln(10)
            >>> print(f"{ppl:.1f}")  # 10.0
        """
        # Clip loss to prevent overflow (exp(100) = 2.7e43)
        # Losses > 100 indicate severe numerical instability anyway
        clipped_loss = min(loss, 100.0)
        return np.exp(clipped_loss)

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> float:
        """
        Compute next-token prediction accuracy.

        Computes the fraction of tokens where argmax(logits) matches the
        target label, excluding padding tokens (ignore_index).

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size] or
                    [batch_size * seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len] or [batch_size * seq_len]
            ignore_index: Label value to ignore (default: -100 for padding)

        Returns:
            Accuracy as float in [0.0, 1.0]

        Raises:
            ZeroDivisionError: If all labels are ignore_index (no valid tokens)

        Examples:
            >>> logits = torch.tensor([[[10, 1], [1, 10]]])  # pred=[0, 1]
            >>> labels = torch.tensor([[0, 1]])
            >>> tracker = MetricsTracker()
            >>> acc = tracker.compute_accuracy(logits, labels)
            >>> print(f"{acc:.1f}")  # 1.0 (100%)
        """
        # Get predictions (argmax over vocabulary dimension)
        predictions = logits.argmax(dim=-1)

        # Create mask for non-ignored positions
        mask = (labels != ignore_index)

        # Compute accuracy only on non-ignored tokens
        correct = (predictions == labels) & mask
        total_valid = mask.sum().item()

        if total_valid == 0:
            raise ZeroDivisionError(
                "All labels are ignore_index - no valid tokens to compute accuracy"
            )

        accuracy = correct.sum().item() / total_valid
        return accuracy

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        gradient_norm: float,
        epoch_duration: float
    ):
        """
        Log metrics for a single epoch to W&B and local storage.

        Computes derived metrics (perplexity), collects system metrics
        (GPU memory/utilization), logs to W&B with error handling, and
        stores to local history for offline analysis.

        Args:
            epoch: Current epoch number (0-indexed)
            train_metrics: Dict with 'loss' and 'accuracy' keys
            val_metrics: Dict with 'loss' and 'accuracy' keys
            learning_rate: Current learning rate from scheduler
            gradient_norm: Maximum gradient norm this epoch
            epoch_duration: Time taken for epoch (seconds)

        Examples:
            >>> tracker = MetricsTracker(use_wandb=True)
            >>> tracker.log_epoch(
            ...     epoch=0,
            ...     train_metrics={'loss': 2.5, 'accuracy': 0.75},
            ...     val_metrics={'loss': 2.7, 'accuracy': 0.72},
            ...     learning_rate=5e-5,
            ...     gradient_norm=0.85,
            ...     epoch_duration=120.5
            ... )
            Epoch 0: train_loss=2.5000 val_loss=2.7000 val_ppl=14.88 val_acc=0.7200
        """
        # Compute derived metrics (perplexity from loss)
        train_ppl = self.compute_perplexity(train_metrics['loss'])
        val_ppl = self.compute_perplexity(val_metrics['loss'])

        # Compile all metrics with namespace prefixes
        metrics_dict = {
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/perplexity': train_ppl,
            'train/accuracy': train_metrics['accuracy'],
            'val/loss': val_metrics['loss'],
            'val/perplexity': val_ppl,
            'val/accuracy': val_metrics['accuracy'],
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            'epoch_duration': epoch_duration,
        }

        # Add system metrics if GPU available
        if torch.cuda.is_available():
            # GPU memory in MB
            gpu_memory_bytes = torch.cuda.max_memory_allocated()
            metrics_dict['system/gpu_memory_mb'] = gpu_memory_bytes / (1024**2)

            # GPU utilization percentage
            metrics_dict['system/gpu_utilization'] = self._get_gpu_utilization()

        # Log to W&B with error handling (don't crash training if W&B fails)
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics_dict, step=epoch)
            except Exception as e:
                print(f"⚠️ W&B logging failed for epoch {epoch}: {e}")

        # Store locally for offline analysis
        self.metrics_history.append(metrics_dict)

        # Print summary to console
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ppl={val_ppl:.2f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    def _get_gpu_utilization(self) -> float:
        """
        Get current GPU utilization percentage via nvidia-smi.

        Runs nvidia-smi subprocess to query GPU utilization. Returns 0.0
        if nvidia-smi is unavailable (Mac, Windows, Docker without GPU)
        or if the query fails.

        Returns:
            GPU utilization percentage (0.0-100.0), or 0.0 on failure

        Examples:
            >>> tracker = MetricsTracker()
            >>> util = tracker._get_gpu_utilization()
            >>> print(f"GPU: {util:.1f}%")  # e.g., "GPU: 75.0%"
        """
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit
            )
            return float(result.stdout.strip())
        except Exception:
            # nvidia-smi not available or query failed
            return 0.0

    def get_summary(self) -> pd.DataFrame:
        """
        Get all metrics as DataFrame for analysis.

        Returns:
            DataFrame with one row per epoch, all metric columns

        Examples:
            >>> tracker = MetricsTracker()
            >>> # ... log some epochs ...
            >>> df = tracker.get_summary()
            >>> print(df[['epoch', 'train/loss', 'val/loss']])
               epoch  train/loss  val/loss
            0      0        2.50      2.70
            1      1        2.30      2.55
        """
        return pd.DataFrame(self.metrics_history)

    def get_best_epoch(
        self,
        metric: str = 'val/loss',
        mode: Literal['min', 'max'] = 'min'
    ) -> int:
        """
        Find epoch with best metric value for model selection.

        Args:
            metric: Metric name to optimize (default: 'val/loss')
            mode: 'min' to minimize, 'max' to maximize (default: 'min')

        Returns:
            Epoch number with best metric value

        Examples:
            >>> tracker = MetricsTracker()
            >>> # ... log epochs with varying val_loss ...
            >>> best_epoch = tracker.get_best_epoch('val/loss', 'min')
            >>> print(f"Best model at epoch {best_epoch}")
        """
        df = self.get_summary()

        if mode == 'min':
            best_idx = df[metric].idxmin()
        else:  # mode == 'max'
            best_idx = df[metric].idxmax()

        return int(df.loc[best_idx, 'epoch'])
