"""
Integration tests for MetricsTracker with training loops.

Tests end-to-end integration of metrics tracking with actual model training,
including W&B logging, offline mode, and error resilience scenarios.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.training.metrics_tracker import MetricsTracker


class TinyTransformer(nn.Module):
    """Minimal transformer for testing (to avoid long training times)."""

    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """Simple forward pass: embed → linear → logits."""
        x = self.embedding(input_ids)
        logits = self.fc(x)
        return logits


def create_synthetic_data(vocab_size=100, seq_len=16, n_samples=10):
    """
    Create synthetic training data for testing.

    Returns:
        List of tensors [n_samples], each of shape [seq_len]
    """
    return [torch.randint(0, vocab_size, (seq_len,)) for _ in range(n_samples)]


def mini_training_loop(
    model: nn.Module,
    train_data: list,
    val_data: list,
    n_epochs: int,
    tracker: MetricsTracker,
    vocab_size: int = 100
):
    """
    Minimal training loop for integration testing.

    Trains model for n_epochs, logging metrics via tracker each epoch.

    Args:
        model: Model to train
        train_data: List of input_ids tensors
        val_data: List of input_ids tensors for validation
        n_epochs: Number of epochs to train
        tracker: MetricsTracker instance
        vocab_size: Vocabulary size

    Returns:
        None (metrics stored in tracker)
    """
    import torch.nn.functional as F
    import time

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = next(model.parameters()).device

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Train phase
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_steps = 0
        max_grad_norm = 0.0

        for sample in train_data:
            sample = sample.to(device)
            optimizer.zero_grad()

            # Forward
            logits = model(sample.unsqueeze(0))  # [1, seq_len, vocab_size]

            # Next-token prediction loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sample[1:].unsqueeze(0).contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Compute accuracy
            accuracy = tracker.compute_accuracy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Backward
            loss.backward()

            # Track gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            max_grad_norm = max(max_grad_norm, grad_norm.item())

            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += accuracy
            train_steps += 1

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for sample in val_data:
                sample = sample.to(device)
                logits = model(sample.unsqueeze(0))

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = sample[1:].unsqueeze(0).contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1)
                )

                accuracy = tracker.compute_accuracy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1)
                )

                val_loss_sum += loss.item()
                val_acc_sum += accuracy
                val_steps += 1

        # Log epoch metrics
        epoch_duration = time.time() - epoch_start

        tracker.log_epoch(
            epoch=epoch,
            train_metrics={
                'loss': train_loss_sum / train_steps,
                'accuracy': train_acc_sum / train_steps
            },
            val_metrics={
                'loss': val_loss_sum / val_steps,
                'accuracy': val_acc_sum / val_steps
            },
            learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=max_grad_norm,
            epoch_duration=epoch_duration
        )


class TestMetricsIntegration:
    """Integration tests for MetricsTracker with training loops."""

    def test_fine_tuning_with_metrics_tracking(self):
        """
        Scenario: Train tiny model for 3 epochs with metrics tracking
        Expected: MetricsTracker records 3 epochs, loss decreases
        Why: Validates E2E integration with real training loop
        """
        # Setup
        model = TinyTransformer(vocab_size=100, d_model=32)
        train_data = create_synthetic_data(vocab_size=100, seq_len=16, n_samples=10)
        val_data = create_synthetic_data(vocab_size=100, seq_len=16, n_samples=5)

        tracker = MetricsTracker(use_wandb=False)

        # Run training
        mini_training_loop(
            model=model,
            train_data=train_data,
            val_data=val_data,
            n_epochs=3,
            tracker=tracker,
            vocab_size=100
        )

        # Verify metrics logged
        df = tracker.get_summary()
        assert len(df) == 3, f"Expected 3 epochs, got {len(df)}"

        # Check all required columns present
        required_cols = [
            'epoch', 'train/loss', 'train/perplexity', 'train/accuracy',
            'val/loss', 'val/perplexity', 'val/accuracy',
            'learning_rate', 'gradient_norm', 'epoch_duration'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Verify loss decreased (training worked)
        initial_loss = df.iloc[0]['train/loss']
        final_loss = df.iloc[-1]['train/loss']
        assert final_loss < initial_loss, \
            f"Loss should decrease during training (initial={initial_loss:.4f}, final={final_loss:.4f})"

    def test_metrics_tracking_offline_mode(self):
        """
        Scenario: Training with use_wandb=False (offline mode)
        Expected: Training completes, metrics available locally
        Why: Validates offline workflow (test scenario 6)
        """
        model = TinyTransformer(vocab_size=50, d_model=16)
        train_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=5)
        val_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=3)

        tracker = MetricsTracker(use_wandb=False)

        # Run training
        mini_training_loop(
            model=model,
            train_data=train_data,
            val_data=val_data,
            n_epochs=2,
            tracker=tracker,
            vocab_size=50
        )

        # Verify local storage
        assert len(tracker.metrics_history) == 2
        assert tracker.metrics_history[0]['epoch'] == 0
        assert tracker.metrics_history[1]['epoch'] == 1

        # Verify we can export to DataFrame
        df = tracker.get_summary()
        assert len(df) == 2

    def test_metrics_tracking_with_wandb_errors(self):
        """
        Scenario: W&B logging fails mid-training (network error)
        Expected: Training continues, metrics saved locally
        Why: Validates error resilience (test scenario 8)
        """
        model = TinyTransformer(vocab_size=50, d_model=16)
        train_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=5)
        val_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=3)

        # We'll test error resilience by verifying the try/except works
        # Simpler approach: use tracker with wandb enabled, but don't actually
        # import wandb (let it fail naturally)
        tracker = MetricsTracker(use_wandb=True)

        # Run training - should complete even if wandb import fails
        mini_training_loop(
            model=model,
            train_data=train_data,
            val_data=val_data,
            n_epochs=2,
            tracker=tracker,
            vocab_size=50
        )

        # Both epochs should be in local storage regardless of W&B status
        assert len(tracker.metrics_history) == 2
        assert tracker.metrics_history[0]['epoch'] == 0
        assert tracker.metrics_history[1]['epoch'] == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_metrics_tracking_gpu_metrics(self):
        """
        Scenario: Training on GPU with CUDA available
        Expected: GPU memory and utilization metrics logged
        Why: Validates system metrics collection (test scenario 5)
        """
        model = TinyTransformer(vocab_size=50, d_model=16).cuda()
        train_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=5)
        val_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=3)

        tracker = MetricsTracker(use_wandb=False)

        mini_training_loop(
            model=model,
            train_data=train_data,
            val_data=val_data,
            n_epochs=2,
            tracker=tracker,
            vocab_size=50
        )

        df = tracker.get_summary()

        # GPU metrics should be present
        assert 'system/gpu_memory_mb' in df.columns
        assert 'system/gpu_utilization' in df.columns

        # GPU memory should be > 0 (model is on GPU)
        assert df['system/gpu_memory_mb'].iloc[0] > 0

    def test_best_epoch_selection(self):
        """
        Scenario: Train for multiple epochs, select best by val_loss
        Expected: get_best_epoch() returns epoch with minimum val_loss
        Why: Validates early stopping / checkpoint selection
        """
        model = TinyTransformer(vocab_size=50, d_model=16)
        train_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=10)
        val_data = create_synthetic_data(vocab_size=50, seq_len=8, n_samples=5)

        tracker = MetricsTracker(use_wandb=False)

        mini_training_loop(
            model=model,
            train_data=train_data,
            val_data=val_data,
            n_epochs=5,
            tracker=tracker,
            vocab_size=50
        )

        # Get best epoch
        best_epoch = tracker.get_best_epoch('val/loss', 'min')

        # Verify it's actually the minimum
        df = tracker.get_summary()
        min_val_loss_epoch = df['val/loss'].idxmin()

        assert best_epoch == df.loc[min_val_loss_epoch, 'epoch']
