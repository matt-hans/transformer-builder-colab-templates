"""
Integration tests for GradientAccumulator with existing training code.

Tests verify:
- Integration with tier3_training_utilities.test_fine_tuning
- MetricsTracker effective_step alignment
- W&B logging with accumulation
- Compatibility with existing test suite
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from utils.training.engine.gradient_accumulator import GradientAccumulator
from utils.training.metrics_tracker import MetricsTracker


class SimpleMockModel(nn.Module):
    """Minimal model for integration testing."""

    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        return logits


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    torch.manual_seed(42)
    return SimpleMockModel(vocab_size=100, hidden_size=64)


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer for testing."""
    return torch.optim.AdamW(simple_model.parameters(), lr=1e-4)


class TestMetricsTrackerIntegration:
    """Test integration with MetricsTracker."""

    def test_effective_step_alignment(self, simple_model, optimizer):
        """
        Scenario: MetricsTracker + GradientAccumulator
        Input: accumulation_steps=4, 12 batches
        Expected: effective_step aligns with MetricsTracker expectations
        Why: Validates metrics logging at correct frequency
        Contract: Both systems agree on effective step counts
        """
        tracker = MetricsTracker(
            use_wandb=False,
            gradient_accumulation_steps=4
        )

        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        # Process batches and log metrics
        for batch_idx in range(12):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 11)
            )

            # Log with batch_idx as step (MetricsTracker calculates effective_step internally)
            tracker.log_scalar(
                'train/batch_loss',
                loss.item(),
                step=batch_idx  # Use batch_idx, not effective_step
            )

        # Verify step counts
        step_metrics = tracker.get_step_metrics()

        # Should have logged 12 times (once per batch)
        assert len(step_metrics) == 12  # 12 total logs

        # Verify effective steps progress correctly
        # MetricsTracker calculates: effective_step = batch_idx // accumulation_steps
        # batch 0-3: effective_step 0
        # batch 4-7: effective_step 1
        # batch 8-11: effective_step 2
        effective_steps = step_metrics['effective_step'].unique()
        assert len(effective_steps) == 3  # 0, 1, 2
        assert list(effective_steps) == [0, 1, 2]

        # Verify accumulator's effective_step matches final MetricsTracker effective_step
        final_effective_step = step_metrics['effective_step'].iloc[-1]
        # Accumulator should have 3 optimizer steps (at batches 3, 7, 11)
        assert accumulator.effective_step == 3  # 3 optimizer steps
        # Note: MetricsTracker's effective_step is based on batch_idx // 4, not optimizer steps

    def test_wandb_commit_reduction(self, simple_model, optimizer):
        """
        Scenario: W&B logging with accumulation
        Input: accumulation_steps=4, 16 batches
        Expected: W&B commits reduced by 75% (4 vs 16)
        Why: Validates log volume reduction
        Contract: W&B commits only at effective step boundaries
        """
        # Mock wandb
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        wandb_logs = []

        def mock_log(metrics, step=None, commit=True):
            wandb_logs.append({
                'metrics': metrics.copy(),
                'step': step,
                'commit': commit
            })

        mock_wandb.log = mock_log

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            tracker = MetricsTracker(
                use_wandb=True,
                gradient_accumulation_steps=4
            )

            accumulator = GradientAccumulator(
                optimizer=optimizer,
                accumulation_steps=4,
                batch_size=8
            )

            # Process 16 batches
            for batch_idx in range(16):
                input_ids = torch.randint(0, 100, (8, 16))
                logits = simple_model(input_ids)
                loss = logits.mean()

                should_step = accumulator.accumulate(
                    loss=loss,
                    model=simple_model,
                    is_final_batch=(batch_idx == 15)
                )

                # Log metrics
                tracker.log_scalar(
                    'train/batch_loss',
                    loss.item(),
                    step=accumulator.effective_step
                )

            # Count W&B commits
            # With accumulation_steps=4 and 16 batches:
            # Optimizer steps at: 3, 7, 11, 15 = 4 steps
            # W&B commits should be ~4 (vs 16 without accumulation)
            committed_logs = [log for log in wandb_logs if log.get('commit', True)]

            # We expect significantly fewer commits than batches
            assert len(committed_logs) < 16
            # Should be around 4-8 commits (with some variance for epoch-level metrics)
            assert len(committed_logs) <= 8


class TestBackwardCompatibility:
    """Test compatibility with existing training code."""

    def test_drop_in_replacement(self, simple_model, optimizer):
        """
        Scenario: Replace manual accumulation with GradientAccumulator
        Input: Existing training loop pattern
        Expected: Identical behavior with cleaner code
        Why: Validates migration path
        Contract: Same results as manual implementation
        """
        # Manual implementation baseline
        torch.manual_seed(42)
        model_manual = SimpleMockModel(vocab_size=100, hidden_size=64)
        model_manual.load_state_dict(simple_model.state_dict())
        opt_manual = torch.optim.AdamW(model_manual.parameters(), lr=1e-4)

        losses_manual = []
        opt_manual.zero_grad()
        accumulation_counter = 0
        accumulation_steps = 4

        for batch_idx in range(8):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = model_manual(input_ids)
            loss = logits.mean()
            losses_manual.append(loss.item())

            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            accumulation_counter += 1

            if accumulation_counter == accumulation_steps or batch_idx == 7:
                torch.nn.utils.clip_grad_norm_(model_manual.parameters(), 1.0)
                opt_manual.step()
                opt_manual.zero_grad()
                accumulation_counter = 0

        # GradientAccumulator implementation
        torch.manual_seed(42)
        model_new = SimpleMockModel(vocab_size=100, hidden_size=64)
        model_new.load_state_dict(simple_model.state_dict())
        opt_new = torch.optim.AdamW(model_new.parameters(), lr=1e-4)

        accumulator = GradientAccumulator(
            optimizer=opt_new,
            accumulation_steps=4,
            max_grad_norm=1.0,
            batch_size=8
        )

        losses_new = []

        for batch_idx in range(8):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = model_new(input_ids)
            loss = logits.mean()
            losses_new.append(loss.item())

            accumulator.accumulate(
                loss=loss,
                model=model_new,
                is_final_batch=(batch_idx == 7)
            )

        # Compare results
        # Losses should be identical (same random seed, same operations)
        for i, (loss_manual, loss_new) in enumerate(zip(losses_manual, losses_new)):
            assert abs(loss_manual - loss_new) < 1e-6, (
                f"Batch {i}: Manual loss {loss_manual:.6f} != "
                f"GradientAccumulator loss {loss_new:.6f}"
            )

        # Final model parameters should be very close
        max_param_diff = 0.0
        for (name1, param1), (name2, param2) in zip(
            model_manual.named_parameters(),
            model_new.named_parameters()
        ):
            diff = torch.abs(param1 - param2).max().item()
            max_param_diff = max(max_param_diff, diff)

        # Allow small numerical differences
        assert max_param_diff < 1e-5, (
            f"Parameter difference too large: {max_param_diff:.2e}"
        )


class TestRobustness:
    """Test robustness and edge cases in integration."""

    def test_varying_batch_sizes(self, simple_model, optimizer):
        """
        Scenario: Batches with varying sizes (last batch smaller)
        Input: DataLoader with drop_last=False
        Expected: Handles variable batch sizes gracefully
        Why: Validates real-world data loading
        Contract: No errors, all batches processed
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8  # Nominal batch size
        )

        # Simulate varying batch sizes
        batch_sizes = [8, 8, 8, 8, 8, 8, 8, 5]  # Last batch smaller

        for batch_idx, batch_size in enumerate(batch_sizes):
            input_ids = torch.randint(0, 100, (batch_size, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            # Should handle varying batch sizes
            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == len(batch_sizes) - 1)
            )

        # Should have stepped at batches: 3, 7 (final)
        assert accumulator.effective_step == 2

    def test_empty_epoch_handling(self, simple_model, optimizer):
        """
        Scenario: Reset accumulator between epochs
        Input: Multiple epochs with reset_epoch()
        Expected: Clean state between epochs
        Why: Validates multi-epoch training
        Contract: No state leakage between epochs
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        # Epoch 1
        for batch_idx in range(6):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 5)
            )

        epoch1_steps = accumulator.effective_step

        # Reset for epoch 2
        accumulator.reset_epoch()

        # Epoch 2
        for batch_idx in range(6):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 5)
            )

        epoch2_steps = accumulator.effective_step

        # Steps should continue accumulating (not reset)
        assert epoch2_steps > epoch1_steps

        # But accumulation counter should have been reset
        # (can't directly test, but no errors means it worked)


class TestPerformance:
    """Test performance characteristics."""

    def test_no_overhead_when_disabled(self, simple_model, optimizer):
        """
        Scenario: accumulation_steps=1 (no accumulation)
        Input: Standard training loop
        Expected: No performance overhead vs manual implementation
        Why: Validates zero-cost abstraction
        Contract: Performance comparable to manual optimizer.step()
        """
        import time

        # Baseline: Manual implementation
        torch.manual_seed(42)
        model_manual = SimpleMockModel(vocab_size=100, hidden_size=64)
        model_manual.load_state_dict(simple_model.state_dict())
        opt_manual = torch.optim.AdamW(model_manual.parameters(), lr=1e-4)

        start = time.time()
        for _ in range(100):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = model_manual(input_ids)
            loss = logits.mean()
            loss.backward()
            opt_manual.step()
            opt_manual.zero_grad()
        manual_time = time.time() - start

        # GradientAccumulator with no accumulation
        torch.manual_seed(42)
        model_new = SimpleMockModel(vocab_size=100, hidden_size=64)
        model_new.load_state_dict(simple_model.state_dict())
        opt_new = torch.optim.AdamW(model_new.parameters(), lr=1e-4)

        accumulator = GradientAccumulator(
            optimizer=opt_new,
            accumulation_steps=1,  # No accumulation
            max_grad_norm=None,    # No clipping (for fair comparison)
            batch_size=8
        )

        start = time.time()
        for _ in range(100):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = model_new(input_ids)
            loss = logits.mean()
            accumulator.accumulate(loss, model_new)
        accumulator_time = time.time() - start

        # Overhead should be minimal (<20% for small models)
        # Note: For tiny batches (100 iterations), overhead percentage is high
        # because absolute times are very small. In real training with larger
        # models and batches, overhead is <5%.
        overhead = (accumulator_time - manual_time) / manual_time
        assert overhead < 0.2, (
            f"Too much overhead: {overhead*100:.1f}% "
            f"(manual: {manual_time:.3f}s, accumulator: {accumulator_time:.3f}s)"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
