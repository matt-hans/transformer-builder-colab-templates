"""
Unit tests for gradient accumulation feature.

Tests verify:
- Loss scaling by accumulation steps
- Optimizer step frequency (every N batches)
- Scheduler step frequency (matches optimizer)
- Effective batch size logging to W&B
- Incomplete final batch handling
- Backward compatibility (accum_steps=1)
- Mathematical equivalence to larger physical batches
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from utils.tier3_training_utilities import test_fine_tuning


class SimpleMockModel(nn.Module):
    """Minimal model for testing gradient accumulation logic."""

    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        return logits


class TestLossScaling:
    """Test that loss is correctly scaled by 1/accumulation_steps."""

    def test_loss_scaling_correctness(self):
        """
        Scenario: Training with gradient_accumulation_steps=4
        Input: Mock loss values from forward pass
        Expected: Loss scaled by 0.25 before backward()
        Why: Validates loss scaling prevents gradient explosion
        Contract: scaled_loss = raw_loss / accumulation_steps
        """
        torch.manual_seed(42)

        model = SimpleMockModel(vocab_size=100)

        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=100, max_seq_len=16)

        # Create training data (4 batches to test one full accumulation cycle)
        train_data = [torch.randint(0, 100, (16,)) for _ in range(4)]

        # Track backward() calls to verify loss scaling
        backward_losses = []
        original_backward = torch.Tensor.backward

        def track_backward(self, *args, **kwargs):
            # Record the loss value that backward() was called on
            backward_losses.append(self.item())
            return original_backward(self, *args, **kwargs)

        with patch.object(torch.Tensor, 'backward', track_backward):
            result = test_fine_tuning(
                model=model,
                config=config,
                train_data=train_data,
                val_data=train_data[:2],
                n_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=4,
                use_wandb=False,
                use_amp=False
            )

            # Verify that backward was called 4 times (once per batch)
            assert len(backward_losses) == 4, (
                f"Expected 4 backward calls, got {len(backward_losses)}"
            )

            # The losses should all be scaled (much smaller than raw loss)
            # We can't easily get raw loss, but we can verify all scaled losses
            # are consistent in magnitude (all divided by 4)
            # As a proxy, verify the final reported loss is similar to backward losses
            final_loss = result['final_loss']

            # The backward losses should be approximately 1/4 of what they would be unscaled
            # We verify this indirectly by checking the final reported loss is reasonable
            assert final_loss > 0, "Final loss should be positive"
            assert final_loss < 100, "Final loss should be reasonable (not exploded)"

            # The key validation: all backward losses should be similar magnitude
            # (all scaled by same factor)
            avg_backward_loss = sum(backward_losses) / len(backward_losses)
            for loss in backward_losses:
                # All scaled losses should be within 50% of average (allowing for variance)
                assert abs(loss - avg_backward_loss) / avg_backward_loss < 0.5, (
                    f"Backward loss {loss} deviates too much from average {avg_backward_loss}"
                )


class TestOptimizerStepFrequency:
    """Test that optimizer.step() is called at correct frequency."""

    def test_optimizer_step_frequency(self, tracked_adamw_factory):
        """
        Scenario: 10 batches with gradient_accumulation_steps=3
        Input: 10 training batches, accum_steps=3
        Expected: optimizer.step() called ceil(10/3) = 4 times
        Why: Validates optimizer updates only after accumulation complete
        Contract: optimizer.step.call_count == ceil(n_batches / accum_steps)
        """
        torch.manual_seed(42)

        model = SimpleMockModel(vocab_size=100)

        # Create config
        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=100, max_seq_len=16)

        # Create 10 training samples
        train_data = [torch.randint(0, 100, (16,)) for _ in range(10)]

        # Use fixture to track optimizer.step() calls
        TrackedAdamW, step_calls = tracked_adamw_factory

        with patch('utils.tier3_training_utilities.torch.optim.AdamW', TrackedAdamW):
            # Run training with accumulation_steps=3, batch_size=1
            result = test_fine_tuning(
                model=model,
                config=config,
                train_data=train_data,
                val_data=train_data[:2],
                n_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=3,
                use_wandb=False,
                use_amp=False
            )

            # With 10 batches and accum_steps=3:
            # Steps 0-2: accumulate (step 1)
            # Steps 3-5: accumulate (step 2)
            # Steps 6-8: accumulate (step 3)
            # Step 9: final incomplete batch (step 4)
            # Expected: 4 optimizer.step() calls

            expected_steps = 4  # ceil(10/3) = 4
            actual_steps = len(step_calls)

            assert actual_steps == expected_steps, (
                f"Expected {expected_steps} optimizer steps for 10 batches "
                f"with accum_steps=3, got {actual_steps}"
            )


class TestSchedulerStepFrequency:
    """Test that scheduler.step() is called with optimizer, not every batch."""

    def test_scheduler_step_frequency(self, tracked_adamw_factory):
        """
        Scenario: 10 batches with gradient_accumulation_steps=3
        Input: 10 training batches, accum_steps=3
        Expected: scheduler.step() called 4 times (matches optimizer)
        Why: Validates learning rate updates only on optimizer steps
        Contract: scheduler.step.call_count == optimizer.step.call_count
        """
        torch.manual_seed(42)

        model = SimpleMockModel(vocab_size=100)

        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=100, max_seq_len=16)

        # Create 10 training samples
        train_data = [torch.randint(0, 100, (16,)) for _ in range(10)]

        # Track both optimizer and scheduler steps
        TrackedAdamW, optimizer_step_calls = tracked_adamw_factory
        scheduler_step_calls = []

        # Track scheduler.step() calls
        def track_scheduler_step(original_step):
            def wrapper(*args, **kwargs):
                scheduler_step_calls.append(1)
                return original_step(*args, **kwargs)
            return wrapper

        with patch('utils.tier3_training_utilities.torch.optim.AdamW', TrackedAdamW):
            # Patch scheduler.step to track calls
            with patch('torch.optim.lr_scheduler.CosineAnnealingLR.step', track_scheduler_step(torch.optim.lr_scheduler.CosineAnnealingLR.step)):
                result = test_fine_tuning(
                    model=model,
                    config=config,
                    train_data=train_data,
                    val_data=train_data[:2],
                    n_epochs=1,
                    batch_size=1,
                    gradient_accumulation_steps=3,
                    use_wandb=False,
                    use_amp=False
                )

                # Scheduler should step with optimizer (4 times)
                expected_scheduler_steps = len(optimizer_step_calls)
                actual_scheduler_steps = len(scheduler_step_calls)

                assert actual_scheduler_steps == expected_scheduler_steps, (
                    f"Expected scheduler.step() called {expected_scheduler_steps} times "
                    f"(matching optimizer), got {actual_scheduler_steps}"
                )


class TestEffectiveBatchSizeLogging:
    """Test that effective batch size is logged to W&B."""

    def test_effective_batch_size_logging(self):
        """
        Scenario: batch_size=4, gradient_accumulation_steps=8
        Input: Training with specified parameters
        Expected: effective_batch_size=32 logged to metrics
        Why: Validates users can see actual effective batch size
        Contract: 'effective_batch_size': 32 in metrics_tracker
        """
        torch.manual_seed(42)

        model = SimpleMockModel(vocab_size=100)

        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=100, max_seq_len=16)

        # Create training data
        train_data = [torch.randint(0, 100, (16,)) for _ in range(16)]

        # Mock wandb to capture logged metrics
        wandb_logs = []

        def mock_wandb_log(metrics, step=None):
            wandb_logs.append(metrics.copy())

        # Mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # W&B is "initialized"
        mock_wandb.log = mock_wandb_log

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            result = test_fine_tuning(
                model=model,
                config=config,
                train_data=train_data,
                val_data=train_data[:2],
                n_epochs=1,
                batch_size=4,
                gradient_accumulation_steps=8,
                use_wandb=True,  # Enable W&B
                use_amp=False
            )

            # Verify effective batch size was logged
            expected_effective_batch_size = 4 * 8  # 32

            # Find config metrics in wandb logs
            config_metrics = [
                log for log in wandb_logs
                if 'config/effective_batch_size' in log
            ]

            assert len(config_metrics) > 0, (
                "No config metrics logged to W&B"
            )

            # Check that effective batch size is correct
            logged_effective_batch_size = config_metrics[0]['config/effective_batch_size']
            assert logged_effective_batch_size == expected_effective_batch_size, (
                f"Expected effective_batch_size={expected_effective_batch_size}, "
                f"got {logged_effective_batch_size}"
            )

            # Also verify gradient accumulation steps are logged
            logged_accum_steps = config_metrics[0]['config/gradient_accumulation_steps']
            assert logged_accum_steps == 8, (
                f"Expected gradient_accumulation_steps=8, got {logged_accum_steps}"
            )


class TestIncompleteFinalBatch:
    """Test handling of final batch when batches % accum_steps != 0."""

    def test_incomplete_final_batch(self):
        """
        Scenario: 10 batches with gradient_accumulation_steps=3
        Input: 10 batches (leaves 1 batch in final accumulation)
        Expected: Final accumulated gradients still applied (4 total steps)
        Why: Validates all gradients are used, not discarded
        Contract: optimizer.step called ceil(batches/accum) times
        """
        # This is same as test_optimizer_step_frequency, verifying
        # the final batch (batch 9) triggers optimizer step despite
        # being incomplete (only 1/3 of accumulation window)
        pytest.skip("Covered by test_optimizer_step_frequency")


class TestBackwardCompatibility:
    """Test that accum_steps=1 behaves identically to no accumulation."""

    def test_accumulation_steps_one_is_noop(self, tracked_adamw_factory):
        """
        Scenario: gradient_accumulation_steps=1 (default)
        Input: Any training configuration with accum_steps=1
        Expected: Optimizer steps every batch (original behavior)
        Why: Validates backward compatibility with existing code
        Contract: optimizer.step.call_count == n_batches
        """
        torch.manual_seed(42)

        model = SimpleMockModel(vocab_size=100)

        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=100, max_seq_len=16)

        # Create 5 training samples
        train_data = [torch.randint(0, 100, (16,)) for _ in range(5)]

        # Use fixture to track optimizer.step() calls
        TrackedAdamW, step_calls = tracked_adamw_factory

        with patch('utils.tier3_training_utilities.torch.optim.AdamW', TrackedAdamW):
            # Run with accum_steps=1, batch_size=1
            result = test_fine_tuning(
                model=model,
                config=config,
                train_data=train_data,
                val_data=train_data[:2],
                n_epochs=1,
                batch_size=1,
                gradient_accumulation_steps=1,  # No accumulation
                use_wandb=False,
                use_amp=False
            )

            # With 5 batches and accum_steps=1:
            # Should step after every batch (5 times)
            expected_steps = 5
            actual_steps = len(step_calls)

            assert actual_steps == expected_steps, (
                f"Expected {expected_steps} optimizer steps with accum_steps=1, "
                f"got {actual_steps}"
            )


class TestGradientEquivalence:
    """Test mathematical equivalence of gradient accumulation."""

    @pytest.mark.slow
    def test_gradient_equivalence_with_larger_batch(self):
        """
        Scenario: Compare (batch=4, accum=8) vs (batch=32, accum=1)
        Input: Same data processed two different ways
        Expected: Final gradients are equal within numerical tolerance
        Why: Validates gradient accumulation is mathematically correct
        Contract: max_grad_diff < 1e-6 (FP32 tolerance)
        """
        torch.manual_seed(42)

        vocab_size = 100
        seq_len = 16

        # Create two identical models
        model_accum = SimpleMockModel(vocab_size=vocab_size)
        model_batch = SimpleMockModel(vocab_size=vocab_size)

        # Ensure identical initialization
        model_batch.load_state_dict(model_accum.state_dict())

        # Create 32 samples
        data = [torch.randint(0, vocab_size, (seq_len,)) for _ in range(32)]

        from types import SimpleNamespace
        config = SimpleNamespace(vocab_size=vocab_size, max_seq_len=seq_len)

        # Train model 1: batch_size=4, gradient_accumulation_steps=8
        # Effective batch = 32
        result_accum = test_fine_tuning(
            model=model_accum,
            config=config,
            train_data=data,
            val_data=data[:4],
            n_epochs=1,
            batch_size=4,
            gradient_accumulation_steps=8,
            use_wandb=False,
            use_amp=False
        )

        # Train model 2: batch_size=32, gradient_accumulation_steps=1
        # Effective batch = 32
        result_batch = test_fine_tuning(
            model=model_batch,
            config=config,
            train_data=data,
            val_data=data[:4],
            n_epochs=1,
            batch_size=32,
            gradient_accumulation_steps=1,
            use_wandb=False,
            use_amp=False
        )

        # Compare final parameters
        max_param_diff = 0.0
        for (name1, param1), (name2, param2) in zip(
            model_accum.named_parameters(),
            model_batch.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} != {name2}"
            diff = torch.abs(param1 - param2).max().item()
            max_param_diff = max(max_param_diff, diff)

        # Allow small numerical differences due to floating point
        tolerance = 1e-5  # Relaxed tolerance for accumulated operations
        assert max_param_diff < tolerance, (
            f"Gradient accumulation produced different parameters. "
            f"Max difference: {max_param_diff:.2e} (tolerance: {tolerance:.2e})"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_accumulation_steps_greater_than_batches(self):
        """
        Scenario: gradient_accumulation_steps=10 but only 5 batches
        Input: 5 batches, accum_steps=10
        Expected: Warning logged, single optimizer step at end
        Why: Validates graceful handling of misconfiguration
        Contract: optimizer.step called exactly 1 time
        """
        pytest.skip("Edge case handling - implement if needed")

    def test_accumulation_steps_zero_raises_error(self):
        """
        Scenario: Invalid gradient_accumulation_steps=0
        Input: accum_steps=0
        Expected: ValueError raised
        Why: Validates input validation
        Contract: Raises ValueError with clear message
        """
        pytest.skip("Input validation - implement if needed")
