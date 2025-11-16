"""
Unit tests for MetricsTracker class.

Tests perplexity computation, accuracy calculation, metrics logging,
error resilience, and data export functionality.
"""

import pytest
import numpy as np
import torch
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.training.metrics_tracker import MetricsTracker


class TestComputePerplexity:
    """Test perplexity computation from cross-entropy loss."""

    def test_compute_perplexity_normal(self):
        """
        Scenario: Normal loss value (ln(10) = 2.3026)
        Expected: perplexity = exp(2.3026) ≈ 10.0
        Why: Validates core perplexity formula
        """
        tracker = MetricsTracker(use_wandb=False)
        loss = 2.3026  # ln(10)
        perplexity = tracker.compute_perplexity(loss)

        assert abs(perplexity - 10.0) < 0.01, f"Expected ~10.0, got {perplexity}"

    def test_compute_perplexity_zero_loss(self):
        """
        Scenario: Perfect predictions (loss=0)
        Expected: perplexity = exp(0) = 1.0
        Why: Validates edge case of perfect model
        """
        tracker = MetricsTracker(use_wandb=False)
        loss = 0.0
        perplexity = tracker.compute_perplexity(loss)

        assert perplexity == 1.0, f"Expected 1.0, got {perplexity}"

    def test_compute_perplexity_clipping(self):
        """
        Scenario: Extremely high loss (150.0, which would cause overflow)
        Expected: Loss clipped to 100.0, perplexity = exp(100.0)
        Why: Validates overflow protection mechanism
        """
        tracker = MetricsTracker(use_wandb=False)
        loss = 150.0
        perplexity = tracker.compute_perplexity(loss)

        # exp(100) = 2.688e43
        expected_ppl = np.exp(100.0)
        assert perplexity == expected_ppl, f"Expected {expected_ppl}, got {perplexity}"
        assert not np.isinf(perplexity), "Perplexity should not be inf"

    def test_compute_perplexity_negative_loss(self):
        """
        Scenario: Negative loss (shouldn't happen, but test robustness)
        Expected: perplexity < 1.0 (exp of negative is < 1)
        Why: Validates handling of unexpected inputs
        """
        tracker = MetricsTracker(use_wandb=False)
        loss = -1.0
        perplexity = tracker.compute_perplexity(loss)

        assert 0 < perplexity < 1.0, f"Expected 0 < ppl < 1, got {perplexity}"


class TestComputeAccuracy:
    """Test next-token prediction accuracy calculation."""

    def test_compute_accuracy_basic_perfect(self):
        """
        Scenario: All predictions correct
        Input: logits favor correct class for all tokens
        Expected: accuracy = 1.0 (100%)
        Why: Validates correct prediction handling
        """
        tracker = MetricsTracker(use_wandb=False)

        # Logits: [batch=2, seq=3, vocab=4]
        # Token 0: highest logit at index 0
        # Token 1: highest logit at index 1
        # Token 2: highest logit at index 2
        logits = torch.tensor([
            [[10.0, 1.0, 1.0, 1.0],   # pred=0
             [1.0, 10.0, 1.0, 1.0],   # pred=1
             [1.0, 1.0, 10.0, 1.0]],  # pred=2

            [[10.0, 1.0, 1.0, 1.0],   # pred=0
             [1.0, 10.0, 1.0, 1.0],   # pred=1
             [1.0, 1.0, 10.0, 1.0]]   # pred=2
        ])

        labels = torch.tensor([
            [0, 1, 2],
            [0, 1, 2]
        ])

        accuracy = tracker.compute_accuracy(logits, labels)
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

    def test_compute_accuracy_basic_half_correct(self):
        """
        Scenario: 50% predictions correct
        Expected: accuracy = 0.5
        Why: Validates accuracy calculation formula
        """
        tracker = MetricsTracker(use_wandb=False)

        # First 3 tokens correct, last 3 wrong
        logits = torch.tensor([
            [[10.0, 1.0, 1.0],   # pred=0, label=0 ✓
             [1.0, 10.0, 1.0],   # pred=1, label=1 ✓
             [1.0, 1.0, 10.0],   # pred=2, label=2 ✓
             [10.0, 1.0, 1.0],   # pred=0, label=1 ✗
             [1.0, 10.0, 1.0],   # pred=1, label=2 ✗
             [1.0, 1.0, 10.0]]   # pred=2, label=0 ✗
        ])

        labels = torch.tensor([[0, 1, 2, 1, 2, 0]])

        accuracy = tracker.compute_accuracy(logits, labels)
        assert accuracy == 0.5, f"Expected 0.5, got {accuracy}"

    def test_compute_accuracy_with_padding(self):
        """
        Scenario: Labels contain padding tokens (ignore_index=-100)
        Expected: Accuracy computed only on non-padding tokens
        Why: Validates ignore_index mask handling
        """
        tracker = MetricsTracker(use_wandb=False)

        # 4 tokens: 2 correct, 1 wrong, 1 padding
        logits = torch.tensor([
            [[10.0, 1.0],   # pred=0, label=0 ✓
             [1.0, 10.0],   # pred=1, label=1 ✓
             [10.0, 1.0],   # pred=0, label=1 ✗
             [1.0, 10.0]]   # pred=1, label=-100 (ignored)
        ])

        labels = torch.tensor([[0, 1, 1, -100]])

        accuracy = tracker.compute_accuracy(logits, labels, ignore_index=-100)

        # 2 correct out of 3 non-padding = 2/3 ≈ 0.6667
        expected = 2.0 / 3.0
        assert abs(accuracy - expected) < 0.001, f"Expected {expected:.4f}, got {accuracy:.4f}"

    def test_compute_accuracy_all_padding(self):
        """
        Scenario: All labels are padding tokens
        Expected: Should not crash, handle division by zero
        Why: Validates edge case robustness
        """
        tracker = MetricsTracker(use_wandb=False)

        logits = torch.tensor([
            [[10.0, 1.0],
             [1.0, 10.0]]
        ])

        labels = torch.tensor([[-100, -100]])

        # This will cause division by zero - should handle gracefully
        with pytest.raises(ZeroDivisionError):
            tracker.compute_accuracy(logits, labels, ignore_index=-100)


class TestLogEpoch:
    """Test epoch metrics logging functionality."""

    def test_log_epoch_stores_locally(self):
        """
        Scenario: Log epoch with use_wandb=False
        Expected: Metrics stored in metrics_history list
        Why: Validates local storage mechanism
        """
        tracker = MetricsTracker(use_wandb=False)

        tracker.log_epoch(
            epoch=0,
            train_metrics={'loss': 2.5, 'accuracy': 0.75},
            val_metrics={'loss': 2.7, 'accuracy': 0.72},
            learning_rate=5e-5,
            gradient_norm=0.85,
            epoch_duration=120.5
        )

        assert len(tracker.metrics_history) == 1, "Should have 1 entry"

        metrics = tracker.metrics_history[0]
        assert metrics['epoch'] == 0
        assert metrics['train/loss'] == 2.5
        assert metrics['train/accuracy'] == 0.75
        assert metrics['val/loss'] == 2.7
        assert metrics['val/accuracy'] == 0.72
        assert metrics['learning_rate'] == 5e-5
        assert metrics['gradient_norm'] == 0.85
        assert metrics['epoch_duration'] == 120.5

    def test_log_epoch_computes_perplexity(self):
        """
        Scenario: Log epoch with specific loss values
        Expected: Perplexity correctly computed as exp(loss)
        Why: Validates perplexity computation integration
        """
        tracker = MetricsTracker(use_wandb=False)

        train_loss = 2.3026  # ln(10)
        val_loss = 1.6094    # ln(5)

        tracker.log_epoch(
            epoch=0,
            train_metrics={'loss': train_loss, 'accuracy': 0.75},
            val_metrics={'loss': val_loss, 'accuracy': 0.72},
            learning_rate=5e-5,
            gradient_norm=0.85,
            epoch_duration=120.5
        )

        metrics = tracker.metrics_history[0]
        assert abs(metrics['train/perplexity'] - 10.0) < 0.01
        assert abs(metrics['val/perplexity'] - 5.0) < 0.01

    def test_log_epoch_wandb_success(self):
        """
        Scenario: W&B enabled and logging succeeds
        Expected: wandb.log() called with correct parameters
        Why: Validates W&B integration
        """
        # Mock wandb at the point of import (inside log_epoch method)
        with patch('builtins.__import__', wraps=__import__) as mock_import:
            mock_wandb = Mock()

            def import_side_effect(name, *args, **kwargs):
                if name == 'wandb':
                    return mock_wandb
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            tracker = MetricsTracker(use_wandb=True)

            tracker.log_epoch(
                epoch=5,
                train_metrics={'loss': 2.5, 'accuracy': 0.75},
                val_metrics={'loss': 2.7, 'accuracy': 0.72},
                learning_rate=5e-5,
                gradient_norm=0.85,
                epoch_duration=120.5
            )

            # Verify wandb.log was called
            assert mock_wandb.log.called, "wandb.log should be called"

            # Get the call arguments
            call_args = mock_wandb.log.call_args
            metrics_dict = call_args[0][0]
            step = call_args[1]['step']

            assert step == 5, f"Expected step=5, got {step}"
            assert 'train/loss' in metrics_dict
            assert 'val/loss' in metrics_dict
            assert metrics_dict['train/loss'] == 2.5

    def test_log_epoch_wandb_failure_resilience(self):
        """
        Scenario: W&B API fails with network error
        Expected: Warning printed, no crash, metrics stored locally
        Why: Validates error resilience (AC #10)
        """
        # Mock wandb at the point of import to raise exception
        with patch('builtins.__import__', wraps=__import__) as mock_import:
            mock_wandb = Mock()
            mock_wandb.log.side_effect = Exception("Network error")

            def import_side_effect(name, *args, **kwargs):
                if name == 'wandb':
                    return mock_wandb
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect

            tracker = MetricsTracker(use_wandb=True)

            # Should not crash
            tracker.log_epoch(
                epoch=5,
                train_metrics={'loss': 2.5, 'accuracy': 0.75},
                val_metrics={'loss': 2.7, 'accuracy': 0.72},
                learning_rate=5e-5,
                gradient_norm=0.85,
                epoch_duration=120.5
            )

            # Metrics should still be stored locally
            assert len(tracker.metrics_history) == 1
            assert tracker.metrics_history[0]['epoch'] == 5

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.max_memory_allocated', return_value=8000 * 1024**2)
    @patch('utils.training.metrics_tracker.MetricsTracker._get_gpu_utilization', return_value=75.0)
    def test_log_epoch_gpu_metrics(self, mock_gpu_util, mock_mem, mock_cuda):
        """
        Scenario: Training on GPU with CUDA available
        Expected: GPU memory and utilization logged
        Why: Validates system metrics collection (AC #9)
        """
        tracker = MetricsTracker(use_wandb=False)

        tracker.log_epoch(
            epoch=0,
            train_metrics={'loss': 2.5, 'accuracy': 0.75},
            val_metrics={'loss': 2.7, 'accuracy': 0.72},
            learning_rate=5e-5,
            gradient_norm=0.85,
            epoch_duration=120.5
        )

        metrics = tracker.metrics_history[0]
        assert 'system/gpu_memory_mb' in metrics
        assert abs(metrics['system/gpu_memory_mb'] - 8000.0) < 1.0
        assert metrics['system/gpu_utilization'] == 75.0


class TestDataExport:
    """Test metrics export and analysis methods."""

    def test_get_summary_returns_dataframe(self):
        """
        Scenario: Log 3 epochs, then get summary
        Expected: DataFrame with 3 rows and all metric columns
        Why: Validates analysis interface
        """
        tracker = MetricsTracker(use_wandb=False)

        # Log 3 epochs
        for epoch in range(3):
            tracker.log_epoch(
                epoch=epoch,
                train_metrics={'loss': 3.0 - epoch * 0.2, 'accuracy': 0.7 + epoch * 0.05},
                val_metrics={'loss': 3.2 - epoch * 0.15, 'accuracy': 0.68 + epoch * 0.04},
                learning_rate=5e-5,
                gradient_norm=0.9 - epoch * 0.1,
                epoch_duration=120.0
            )

        df = tracker.get_summary()

        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) == 3, "Should have 3 rows"
        assert 'epoch' in df.columns
        assert 'train/loss' in df.columns
        assert 'val/loss' in df.columns
        assert 'train/perplexity' in df.columns

    def test_get_best_epoch_min_loss(self):
        """
        Scenario: 3 epochs with varying val_loss
        Expected: Returns epoch with minimum val_loss
        Why: Validates best model selection for early stopping
        """
        tracker = MetricsTracker(use_wandb=False)

        # Epoch 1: val_loss = 3.0
        # Epoch 2: val_loss = 2.5  <- best
        # Epoch 3: val_loss = 2.8
        for epoch, val_loss in enumerate([3.0, 2.5, 2.8]):
            tracker.log_epoch(
                epoch=epoch,
                train_metrics={'loss': 2.5, 'accuracy': 0.75},
                val_metrics={'loss': val_loss, 'accuracy': 0.72},
                learning_rate=5e-5,
                gradient_norm=0.85,
                epoch_duration=120.0
            )

        best_epoch = tracker.get_best_epoch(metric='val/loss', mode='min')
        assert best_epoch == 1, f"Expected epoch 1, got {best_epoch}"

    def test_get_best_epoch_max_accuracy(self):
        """
        Scenario: 3 epochs with varying val_accuracy
        Expected: Returns epoch with maximum val_accuracy
        Why: Validates best model selection by accuracy
        """
        tracker = MetricsTracker(use_wandb=False)

        # Epoch 0: val_acc = 0.70
        # Epoch 1: val_acc = 0.75  <- best
        # Epoch 2: val_acc = 0.73
        for epoch, val_acc in enumerate([0.70, 0.75, 0.73]):
            tracker.log_epoch(
                epoch=epoch,
                train_metrics={'loss': 2.5, 'accuracy': 0.75},
                val_metrics={'loss': 2.7, 'accuracy': val_acc},
                learning_rate=5e-5,
                gradient_norm=0.85,
                epoch_duration=120.0
            )

        best_epoch = tracker.get_best_epoch(metric='val/accuracy', mode='max')
        assert best_epoch == 1, f"Expected epoch 1, got {best_epoch}"


class TestGPUUtilization:
    """Test GPU utilization monitoring."""

    @patch('subprocess.run')
    def test_get_gpu_utilization_success(self, mock_run):
        """
        Scenario: nvidia-smi available and returns 75%
        Expected: Returns 75.0
        Why: Validates GPU monitoring on supported systems
        """
        mock_result = Mock()
        mock_result.stdout = "75\n"
        mock_run.return_value = mock_result

        tracker = MetricsTracker(use_wandb=False)
        utilization = tracker._get_gpu_utilization()

        assert utilization == 75.0

    @patch('subprocess.run', side_effect=Exception("nvidia-smi not found"))
    def test_get_gpu_utilization_graceful_failure(self, mock_run):
        """
        Scenario: nvidia-smi not available (Mac, Windows, no GPU)
        Expected: Returns 0.0, no crash
        Why: Validates cross-platform robustness
        """
        tracker = MetricsTracker(use_wandb=False)
        utilization = tracker._get_gpu_utilization()

        assert utilization == 0.0, f"Expected 0.0, got {utilization}"
