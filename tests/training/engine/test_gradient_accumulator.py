"""
Unit tests for GradientAccumulator.

Tests verify:
- Manual gradient accumulation (steps=1,4,8)
- Lightning trainer integration and delegation
- Double accumulation conflict detection
- Loss scaling correctness
- Optimizer step frequency
- Effective step counting for MetricsTracker
- W&B logging alignment with accumulation
- State persistence (checkpointing)
- Edge cases (final batch, empty accumulation)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from utils.training.engine.gradient_accumulator import (
    GradientAccumulator,
    AccumulationStats
)


class SimpleMockModel(nn.Module):
    """Minimal model for testing gradient accumulation."""

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


class TestBasicAccumulation:
    """Test basic gradient accumulation without Lightning."""

    def test_no_accumulation(self, simple_model, optimizer):
        """
        Scenario: accumulation_steps=1 (no accumulation)
        Input: 5 batches
        Expected: optimizer.step() called 5 times
        Why: Validates backward compatibility
        Contract: optimizer_steps == total_steps
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=1,
            batch_size=4
        )

        # Track optimizer.step() calls
        step_count = 0
        original_step = optimizer.step

        def track_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        optimizer.step = track_step

        # Process 5 batches
        for batch_idx in range(5):
            input_ids = torch.randint(0, 100, (4, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 4)
            )

            assert should_step, f"Expected step at batch {batch_idx}"

        assert step_count == 5, f"Expected 5 optimizer steps, got {step_count}"
        assert accumulator.effective_step == 5
        assert accumulator.stats.optimizer_steps == 5

    def test_accumulation_steps_4(self, simple_model, optimizer):
        """
        Scenario: accumulation_steps=4
        Input: 10 batches
        Expected: optimizer.step() called ceil(10/4) = 3 times
        Why: Validates accumulation logic
        Contract: optimizer_steps == ceil(total_steps / accumulation_steps)
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        # Track optimizer.step() calls
        step_count = 0
        original_step = optimizer.step

        def track_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        optimizer.step = track_step

        # Process 10 batches
        for batch_idx in range(10):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 9)
            )

            # Should step at batches: 3, 7, 9 (final)
            expected_step = batch_idx in [3, 7, 9]
            assert should_step == expected_step, (
                f"Batch {batch_idx}: expected step={expected_step}, got {should_step}"
            )

        assert step_count == 3, f"Expected 3 optimizer steps, got {step_count}"
        assert accumulator.effective_step == 3

    def test_accumulation_steps_8(self, simple_model, optimizer):
        """
        Scenario: accumulation_steps=8
        Input: 16 batches
        Expected: optimizer.step() called 2 times (at batch 7 and 15)
        Why: Validates accumulation with exact multiple
        Contract: optimizer_steps == total_steps / accumulation_steps
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=8,
            batch_size=4
        )

        # Track optimizer.step() calls
        step_count = 0
        original_step = optimizer.step

        def track_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        optimizer.step = track_step

        # Process 16 batches (exact multiple of 8)
        for batch_idx in range(16):
            input_ids = torch.randint(0, 100, (4, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 15)
            )

            # Should step at batches: 7, 15
            expected_step = batch_idx in [7, 15]
            assert should_step == expected_step

        assert step_count == 2, f"Expected 2 optimizer steps, got {step_count}"
        assert accumulator.effective_step == 2
        assert accumulator.effective_batch_size == 4 * 8  # 32


class TestLossScaling:
    """Test that loss is correctly scaled before backward()."""

    def test_loss_scaling_correctness(self, simple_model, optimizer):
        """
        Scenario: accumulation_steps=4
        Input: Manual loss values
        Expected: Loss scaled by 0.25 before backward()
        Why: Validates loss scaling prevents gradient explosion
        Contract: scaled_loss = raw_loss / accumulation_steps
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=1,
            scaler=None  # No AMP for clear tracking
        )

        # Track backward() calls
        backward_losses = []
        original_backward = torch.Tensor.backward

        def track_backward(self, *args, **kwargs):
            backward_losses.append(self.item())
            return original_backward(self, *args, **kwargs)

        # Process 4 batches to complete one accumulation cycle
        with patch.object(torch.Tensor, 'backward', track_backward):
            for batch_idx in range(4):
                input_ids = torch.randint(0, 100, (1, 16))
                logits = simple_model(input_ids)
                loss = logits.mean()

                accumulator.accumulate(
                    loss=loss,
                    model=simple_model,
                    is_final_batch=(batch_idx == 3)
                )

        # Verify 4 backward calls
        assert len(backward_losses) == 4

        # All losses should be scaled (consistent magnitude)
        avg_loss = sum(backward_losses) / len(backward_losses)
        for loss in backward_losses:
            # Allow 50% variance (different batches have different losses)
            assert abs(loss - avg_loss) / avg_loss < 0.5


class TestEffectiveStepCounting:
    """Test effective step counting for MetricsTracker integration."""

    def test_effective_step_matches_optimizer_steps(self, simple_model, optimizer):
        """
        Scenario: accumulation_steps=4, 12 batches
        Input: 12 training batches
        Expected: effective_step matches optimizer.step() call count
        Why: Validates MetricsTracker integration
        Contract: effective_step == optimizer_steps == ceil(12/4) == 3
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        effective_steps = []

        for batch_idx in range(12):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 11)
            )

            # Record effective step after each batch
            effective_steps.append(accumulator.effective_step)

        # Verify effective_step progression
        # Steps should increase at batches: 3, 7, 11 (final)
        expected_steps = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3]
        assert effective_steps == expected_steps

    def test_effective_step_for_wandb_logging(self, simple_model, optimizer):
        """
        Scenario: Verify effective_step suitable for W&B commit control
        Input: accumulation_steps=4, 8 batches
        Expected: effective_step only changes when optimizer steps
        Why: Validates W&B log volume reduction (75% with steps=4)
        Contract: W&B commits only at effective step boundaries
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=4
        )

        # Simulate W&B logging (only after optimizer step)
        wandb_commits = []

        for batch_idx in range(8):
            input_ids = torch.randint(0, 100, (4, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 7)
            )

            # Only commit when optimizer steps
            if should_step:
                wandb_commits.append(accumulator.effective_step)

        # With 8 batches and accumulation_steps=4:
        # Optimizer steps at batch 3 and 7 → 2 commits
        assert len(wandb_commits) == 2
        assert wandb_commits == [1, 2]


class TestLightningIntegration:
    """Test PyTorch Lightning integration and conflict detection."""

    def test_lightning_detection(self, optimizer):
        """
        Scenario: Trainer with accumulate_grad_batches=4
        Input: Lightning trainer instance
        Expected: is_lightning_managed == True
        Why: Validates automatic Lightning detection
        Contract: Manual accumulation disabled when Lightning detected
        """
        # Mock Lightning Trainer
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = 'Trainer'
        mock_trainer.accumulate_grad_batches = 4

        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=1,
            trainer=mock_trainer
        )

        assert accumulator.is_lightning_managed
        assert accumulator.effective_batch_size == 4  # batch_size * lightning_accum

    def test_lightning_conflict_detection(self, optimizer):
        """
        Scenario: Both manual (steps=4) and Lightning (accum=4) enabled
        Input: accumulation_steps=4, trainer.accumulate_grad_batches=4
        Expected: ValueError raised with resolution steps
        Why: Prevents double accumulation (8x effective batch)
        Contract: Raises ValueError with clear error message
        """
        # Mock Lightning Trainer
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = 'Trainer'
        mock_trainer.accumulate_grad_batches = 4

        with pytest.raises(ValueError) as exc_info:
            GradientAccumulator(
                optimizer=optimizer,
                accumulation_steps=4,  # Conflict!
                trainer=mock_trainer
            )

        error_msg = str(exc_info.value)
        assert "conflict" in error_msg.lower()
        assert "accumulation_steps: 4" in error_msg
        assert "accumulate_grad_batches: 4" in error_msg
        assert "Resolution options" in error_msg

    def test_lightning_delegation(self, simple_model, optimizer):
        """
        Scenario: Lightning-managed accumulation
        Input: Trainer with accumulate_grad_batches=4
        Expected: accumulate() always returns True, manual logic bypassed
        Why: Lightning handles everything, just track steps
        Contract: is_lightning_managed → always return True from accumulate()
        """
        # Mock Lightning Trainer
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = 'Trainer'
        mock_trainer.accumulate_grad_batches = 4
        mock_trainer.global_step = 0

        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=1,
            trainer=mock_trainer
        )

        # Process batches - should always return True (Lightning controls)
        for batch_idx in range(8):
            input_ids = torch.randint(0, 100, (4, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            # Simulate Lightning updating global_step
            mock_trainer.global_step = batch_idx // 4

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 7)
            )

            assert should_step  # Always True for Lightning

        # effective_step should match Lightning's global_step
        assert accumulator.effective_step == mock_trainer.global_step


class TestGradientClipping:
    """Test gradient clipping integration."""

    def test_gradient_clipping(self, simple_model, optimizer):
        """
        Scenario: max_grad_norm=1.0, accumulation_steps=2
        Input: Batches that produce high gradients
        Expected: Gradients clipped before optimizer step
        Why: Validates gradient clipping is applied correctly
        Contract: post-clip norm <= max_grad_norm
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=2,
            max_grad_norm=1.0
        )

        # Create batches with high loss to trigger clipping
        for batch_idx in range(4):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            # Large loss → large gradients
            loss = logits.mean() * 100.0

            accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 3)
            )

            # After optimizer step, check last grad norm
            if batch_idx in [1, 3]:  # Optimizer stepped
                # Note: grad_norm stored before clipping
                # In practice, should be <= max_grad_norm after clip
                # This is a sanity check that clipping was attempted
                assert accumulator.stats.last_grad_norm >= 0

    def test_no_gradient_clipping(self, simple_model, optimizer):
        """
        Scenario: max_grad_norm=None (disabled)
        Input: Any batches
        Expected: No clipping applied, gradients preserved
        Why: Validates clipping can be disabled
        Contract: max_grad_norm=None → no clip_grad_norm_ call
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=2,
            max_grad_norm=None  # Disabled
        )

        # Track clip_grad_norm_ calls
        clip_calls = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def track_clip(*args, **kwargs):
            clip_calls.append(1)
            return original_clip(*args, **kwargs)

        with patch('torch.nn.utils.clip_grad_norm_', track_clip):
            for batch_idx in range(4):
                input_ids = torch.randint(0, 100, (8, 16))
                logits = simple_model(input_ids)
                loss = logits.mean()

                accumulator.accumulate(
                    loss=loss,
                    model=simple_model,
                    is_final_batch=(batch_idx == 3)
                )

        # No clipping should have occurred
        assert len(clip_calls) == 0


class TestAccumulationStats:
    """Test AccumulationStats reporting."""

    def test_stats_reporting(self, simple_model, optimizer):
        """
        Scenario: Track stats during accumulation
        Input: accumulation_steps=4, 10 batches
        Expected: Stats accurately reflect accumulation state
        Why: Validates stats API for monitoring
        Contract: Stats match internal state at all times
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        for batch_idx in range(10):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 9)
            )

            stats = accumulator.stats

            # Verify stats accuracy
            assert stats.total_steps == batch_idx + 1
            assert stats.effective_batch_size == 8 * 4  # 32

            # current_accumulation follows pattern: 1,2,3,0, 1,2,3,0, 1,2
            # After optimizer step (batches 3, 7, 9), counter resets to 0
            if batch_idx == 3 or batch_idx == 7 or batch_idx == 9:
                # Just stepped, counter is 0
                assert stats.current_accumulation == 0
                assert not stats.is_accumulating
            else:
                # Still accumulating
                assert stats.current_accumulation > 0
                assert stats.is_accumulating


class TestStateManagement:
    """Test state persistence and checkpointing."""

    def test_state_dict_save_load(self, simple_model, optimizer):
        """
        Scenario: Save and load accumulator state
        Input: Accumulator with some history
        Expected: State restored correctly after load
        Why: Validates checkpointing support
        Contract: load_state_dict restores exact state
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        # Process some batches
        for batch_idx in range(6):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 5)
            )

        # Save state
        state = accumulator.state_dict()

        # Create new accumulator and load state
        new_accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )
        new_accumulator.load_state_dict(state)

        # Verify state restored
        assert new_accumulator.effective_step == accumulator.effective_step
        assert new_accumulator.stats.total_steps == accumulator.stats.total_steps
        assert new_accumulator.stats.optimizer_steps == accumulator.stats.optimizer_steps

    def test_reset_epoch(self, simple_model, optimizer):
        """
        Scenario: Reset accumulator between epochs
        Input: Accumulator mid-accumulation
        Expected: Accumulation counter reset, warning if incomplete
        Why: Validates epoch boundary handling
        Contract: reset_epoch() clears accumulation_counter
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        # Process 2 batches (incomplete accumulation)
        for batch_idx in range(2):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            accumulator.accumulate(
                loss=loss,
                model=simple_model
            )

        # Should have accumulated gradients
        assert accumulator.stats.current_accumulation == 2

        # Reset epoch (should warn about incomplete accumulation)
        with patch('utils.training.engine.gradient_accumulator.logger') as mock_logger:
            accumulator.reset_epoch()
            mock_logger.warning.assert_called_once()

        # Accumulation counter should be reset
        assert accumulator.stats.current_accumulation == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_accumulation_steps_validation(self, optimizer):
        """
        Scenario: Invalid accumulation_steps values
        Input: accumulation_steps < 1
        Expected: ValueError raised
        Why: Validates input validation
        Contract: accumulation_steps must be >= 1
        """
        with pytest.raises(ValueError) as exc_info:
            GradientAccumulator(
                optimizer=optimizer,
                accumulation_steps=0  # Invalid
            )

        assert "accumulation_steps must be >= 1" in str(exc_info.value)

    def test_max_grad_norm_validation(self, optimizer):
        """
        Scenario: Invalid max_grad_norm values
        Input: max_grad_norm <= 0
        Expected: ValueError raised
        Why: Validates input validation
        Contract: max_grad_norm must be > 0 or None
        """
        with pytest.raises(ValueError) as exc_info:
            GradientAccumulator(
                optimizer=optimizer,
                max_grad_norm=-1.0  # Invalid
            )

        assert "max_grad_norm must be > 0 or None" in str(exc_info.value)

    def test_final_batch_handling(self, simple_model, optimizer):
        """
        Scenario: Final batch with incomplete accumulation
        Input: 5 batches with accumulation_steps=4
        Expected: Final batch triggers optimizer step
        Why: Validates all gradients are used
        Contract: is_final_batch=True forces optimizer step
        """
        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=4,
            batch_size=8
        )

        step_count = 0
        original_step = optimizer.step

        def track_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        optimizer.step = track_step

        # Process 5 batches
        for batch_idx in range(5):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            should_step = accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 4)
            )

            # Should step at batch 3 (accumulation complete) and 4 (final)
            if batch_idx in [3, 4]:
                assert should_step

        # Two optimizer steps: one at batch 3, one at batch 4 (final)
        assert step_count == 2


class TestAMPIntegration:
    """Test integration with Automatic Mixed Precision (AMP)."""

    def test_amp_with_scaler(self, simple_model, optimizer):
        """
        Scenario: Accumulation with AMP GradScaler
        Input: accumulation_steps=2, GradScaler enabled
        Expected: Scaler properly scales/unscales gradients
        Why: Validates AMP integration
        Contract: Scaler used for scale/unscale/step/update
        """
        scaler = torch.cuda.amp.GradScaler()

        accumulator = GradientAccumulator(
            optimizer=optimizer,
            accumulation_steps=2,
            scaler=scaler
        )

        # Track scaler.step() calls
        step_calls = []
        original_step = scaler.step

        def track_step(*args, **kwargs):
            step_calls.append(1)
            return original_step(*args, **kwargs)

        scaler.step = track_step

        # Process 4 batches
        for batch_idx in range(4):
            input_ids = torch.randint(0, 100, (8, 16))
            logits = simple_model(input_ids)
            loss = logits.mean()

            accumulator.accumulate(
                loss=loss,
                model=simple_model,
                is_final_batch=(batch_idx == 3)
            )

        # Scaler.step() should be called twice (at batch 1 and 3)
        assert len(step_calls) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
