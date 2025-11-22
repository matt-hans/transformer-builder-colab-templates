"""
Integration tests for Trainer API contract and notebook compatibility.

These tests ensure that:
1. Trainer.train() returns the expected schema (v4.0+)
2. Legacy fields are present for backward compatibility (v3.x)
3. Notebook metrics reporting works without KeyError
4. W&B integration doesn't crash training
5. API changes are caught before production deployment
"""

import pytest
import pandas as pd
import torch
import torch.nn as nn
from types import SimpleNamespace

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def tiny_model():
    """Minimal transformer model for fast testing."""
    class TinyTransformer(nn.Module):
        def __init__(self, vocab_size=100, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            return self.output(x)

    return TinyTransformer()


@pytest.fixture
def tiny_config():
    """Minimal model config."""
    return SimpleNamespace(
        vocab_size=100,
        d_model=64,
        max_seq_len=16,
        pad_token_id=0
    )


@pytest.fixture
def tiny_training_config():
    """Minimal training config for fast tests."""
    return TrainingConfig(
        epochs=2,
        batch_size=4,
        max_train_samples=20,
        max_val_samples=10,
        learning_rate=1e-3,
        checkpoint_dir="/tmp/test_checkpoints",
        save_every_n_epochs=1,
        wandb_project=None,  # Disable W&B for tests
        random_seed=42
    )


@pytest.fixture
def tiny_task_spec():
    """Minimal task spec."""
    return TaskSpec.text_tiny()


@pytest.fixture
def tiny_tokenizer():
    """Mock tokenizer for text tasks."""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.vocab_size = 100

        def __call__(self, texts, **kwargs):
            # Simple mock: return random token IDs
            return {
                'input_ids': torch.randint(0, self.vocab_size, (len(texts), 16)),
                'attention_mask': torch.ones(len(texts), 16)
            }

    return MockTokenizer()


@pytest.fixture
def tiny_train_data():
    """Minimal training dataset."""
    return [
        {'input_ids': torch.randint(0, 100, (16,)), 'labels': torch.randint(0, 100, (16,))}
        for _ in range(20)
    ]


@pytest.fixture
def tiny_val_data():
    """Minimal validation dataset."""
    return [
        {'input_ids': torch.randint(0, 100, (16,)), 'labels': torch.randint(0, 100, (16,))}
        for _ in range(10)
    ]


# =============================================================================
# Core API Contract Tests
# =============================================================================

def test_trainer_return_value_schema_modern_api(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Test that Trainer.train() returns the expected v4.0+ schema.

    This is the PRIMARY integration test that catches API regressions.
    """
    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    results = trainer.train(tiny_train_data, tiny_val_data)

    # Modern API (v4.0+) - Required fields
    assert 'metrics_summary' in results, "Missing 'metrics_summary' field"
    assert isinstance(results['metrics_summary'], pd.DataFrame), \
        f"'metrics_summary' should be DataFrame, got {type(results['metrics_summary'])}"

    # Check DataFrame structure
    df = results['metrics_summary']
    assert not df.empty, "metrics_summary should not be empty after training"
    assert 'train/loss' in df.columns, "Missing 'train/loss' column"
    assert 'val/loss' in df.columns, "Missing 'val/loss' column"
    assert len(df) == tiny_training_config.epochs, \
        f"Expected {tiny_training_config.epochs} rows, got {len(df)}"

    # Check other required fields
    assert 'best_epoch' in results, "Missing 'best_epoch' field"
    assert isinstance(results['best_epoch'], int), \
        f"'best_epoch' should be int, got {type(results['best_epoch'])}"

    assert 'final_loss' in results, "Missing 'final_loss' field"
    assert isinstance(results['final_loss'], float), \
        f"'final_loss' should be float, got {type(results['final_loss'])}"

    assert 'checkpoint_path' in results, "Missing 'checkpoint_path' field"

    assert 'training_time' in results, "Missing 'training_time' field"
    assert isinstance(results['training_time'], float), \
        f"'training_time' should be float, got {type(results['training_time'])}"
    assert results['training_time'] > 0, "training_time should be positive"


def test_trainer_return_value_schema_legacy_api(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Test that Trainer.train() includes legacy v3.x fields for backward compatibility.

    This test ensures existing notebooks don't break when upgrading to v4.0.
    """
    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    results = trainer.train(tiny_train_data, tiny_val_data)

    # Legacy API (v3.x) - Deprecated but required for compatibility
    assert 'loss_history' in results, "Missing 'loss_history' field (v3.x compatibility)"
    assert isinstance(results['loss_history'], list), \
        f"'loss_history' should be list, got {type(results['loss_history'])}"
    assert len(results['loss_history']) == tiny_training_config.epochs, \
        f"Expected {tiny_training_config.epochs} losses, got {len(results['loss_history'])}"

    assert 'val_loss_history' in results, "Missing 'val_loss_history' field (v3.x compatibility)"
    assert isinstance(results['val_loss_history'], list), \
        f"'val_loss_history' should be list, got {type(results['val_loss_history'])}"

    # Verify consistency between old and new APIs
    assert results['loss_history'][-1] == results['final_loss'], \
        "loss_history[-1] should match final_loss"

    assert results['loss_history'] == results['metrics_summary']['train/loss'].tolist(), \
        "loss_history should match metrics_summary['train/loss']"


def test_notebook_metrics_reporting_flow(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Simulate notebook Cell 31 metrics reporting to catch KeyError regressions.

    This test replicates the exact code path that failed in production,
    ensuring the fix prevents future occurrences.
    """
    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    results = trainer.train(tiny_train_data, tiny_val_data)

    # Simulate notebook Cell 31 - Legacy path (v3.x)
    # This line caused KeyError: 'loss_history' before the fix
    try:
        final_train_loss = results['loss_history'][-1]
    except KeyError as e:
        pytest.fail(f"Legacy metrics reporting failed (v3.x path): {e}")

    # Simulate notebook Cell 31 - Modern path (v4.0+)
    try:
        if 'metrics_summary' in results and not results['metrics_summary'].empty:
            final_metrics = results['metrics_summary'].iloc[-1]
            val_loss = final_metrics['val/loss']
            val_ppl = final_metrics['val/perplexity']
            val_acc = final_metrics['val/accuracy']
    except (KeyError, IndexError) as e:
        pytest.fail(f"Modern metrics reporting failed (v4.0+ path): {e}")

    # Both paths should work
    assert final_train_loss is not None, "Legacy path failed to retrieve final_train_loss"
    assert val_loss is not None, "Modern path failed to retrieve val_loss"


def test_trainer_with_wandb_disabled(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Test that training completes successfully when W&B is disabled.

    Ensures W&B failures don't crash training (error resilience).
    """
    # W&B disabled via config
    tiny_training_config.wandb_project = None

    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    # Should complete without crashing
    results = trainer.train(tiny_train_data, tiny_val_data)
    assert results is not None
    assert 'metrics_summary' in results


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU-only test"
)
def test_trainer_with_wandb_enabled_but_not_initialized(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Test that W&B failures are caught gracefully (don't crash training).

    Simulates the production error: wandb.log() called without wandb.init().
    """
    # Enable W&B but don't call wandb.init()
    tiny_training_config.wandb_project = "test-project"

    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    # Should complete with warning, not crash
    # W&B failures are caught in MetricsTracker.log_epoch()
    results = trainer.train(tiny_train_data, tiny_val_data)
    assert results is not None
    assert 'metrics_summary' in results


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_trainer_with_empty_metrics(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data
):
    """
    Test Trainer behavior when metrics_summary is empty (0 epochs).

    Edge case: Ensures no IndexError or KeyError when accessing empty DataFrame.
    """
    # 0 epochs training (edge case)
    tiny_training_config.epochs = 0

    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    results = trainer.train(tiny_train_data, None)

    # Should return empty structures gracefully
    assert 'metrics_summary' in results
    assert results['metrics_summary'].empty
    assert results['loss_history'] == []
    assert results['val_loss_history'] == []
    assert results['final_loss'] == 0.0  # Default when empty


def test_trainer_without_validation_data(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data
):
    """
    Test Trainer behavior when validation data is not provided.

    Should complete successfully with only training metrics.
    """
    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    results = trainer.train(tiny_train_data, val_data=None)

    # Should have training metrics, no validation metrics
    assert 'metrics_summary' in results
    assert 'train/loss' in results['metrics_summary'].columns

    # val_loss_history should be empty list (not None, not KeyError)
    assert 'val_loss_history' in results
    assert results['val_loss_history'] == []


# =============================================================================
# Regression Tests for Known Failures
# =============================================================================

def test_regression_keyerror_loss_history(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Regression test for production bug: KeyError 'loss_history'.

    This test would have FAILED before the fix, preventing deployment.
    After the fix, it should PASS.
    """
    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    results = trainer.train(tiny_train_data, tiny_val_data)

    # The exact line that failed in production
    # Before fix: KeyError: 'loss_history'
    # After fix: Should work
    try:
        final_train_loss = results['loss_history'][-1]
        assert final_train_loss > 0, "Expected positive loss"
    except KeyError:
        pytest.fail(
            "REGRESSION: KeyError 'loss_history' bug has returned! "
            "This is the exact failure that occurred in production. "
            "See MLOPS_FAILURE_ANALYSIS.md for details."
        )


# =============================================================================
# Performance / Sanity Tests
# =============================================================================

def test_trainer_completes_in_reasonable_time(
    tiny_model, tiny_config, tiny_training_config, tiny_task_spec,
    tiny_tokenizer, tiny_train_data, tiny_val_data
):
    """
    Sanity test: Ensure training completes in reasonable time.

    Prevents regressions that drastically slow down training.
    """
    import time

    trainer = Trainer(
        model=tiny_model,
        config=tiny_config,
        training_config=tiny_training_config,
        task_spec=tiny_task_spec,
        tokenizer=tiny_tokenizer
    )

    start_time = time.time()
    results = trainer.train(tiny_train_data, tiny_val_data)
    elapsed = time.time() - start_time

    # Tiny dataset (20 samples, 2 epochs) should complete in < 30 seconds on CPU
    assert elapsed < 30, f"Training took {elapsed:.1f}s (expected < 30s)"

    # Verify training_time field matches
    assert abs(results['training_time'] - elapsed) < 1.0, \
        "training_time field doesn't match actual elapsed time"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
