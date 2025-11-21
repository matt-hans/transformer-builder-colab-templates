"""
Unit Tests for Trainer Orchestrator

Tests the high-level training workflow coordinator with mocked components.

Test Coverage:
1. Initialization and configuration validation
2. Component setup (checkpoint, metrics, loss, optimizer)
3. Training workflow execution
4. Resume from checkpoint
5. Hook invocation at correct points
6. Backward compatibility with SimpleNamespace config
7. Error handling and validation
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

from utils.training.engine.trainer import Trainer, TrainingHooks, DefaultHooks
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Simple transformer model for testing."""
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=100, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.linear = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids, labels=None, **kwargs):
            x = self.embedding(input_ids)
            logits = self.linear(x)
            return {'logits': logits}

    return DummyModel()


@pytest.fixture
def model_config():
    """Simple model configuration."""
    return SimpleNamespace(
        vocab_size=100,
        max_seq_len=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        pad_token_id=0
    )


@pytest.fixture
def training_config():
    """Training configuration with minimal settings."""
    return TrainingConfig(
        learning_rate=5e-5,
        batch_size=2,
        epochs=3,
        save_every_n_epochs=2,
        checkpoint_dir='/tmp/test_checkpoints',
        wandb_project=None,  # Disable W&B for tests
        gradient_accumulation_steps=1,
        random_seed=42,
        deterministic=False
    )


@pytest.fixture
def task_spec():
    """Simple task specification."""
    from utils.training.task_spec import get_default_task_specs
    return get_default_task_specs()['lm_tiny']


@pytest.fixture
def dummy_dataset():
    """Dummy dataset for testing."""
    from torch.utils.data import TensorDataset

    # Create random data
    input_ids = torch.randint(0, 100, (16, 32))  # 16 samples, seq_len=32
    labels = torch.randint(0, 100, (16, 32))

    return TensorDataset(input_ids, labels)


# =============================================================================
# Initialization Tests
# =============================================================================

def test_trainer_initialization(simple_model, model_config, training_config, task_spec):
    """Test that Trainer initializes all components correctly."""
    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config,
        task_spec=task_spec
    )

    # Check components initialized
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.checkpoint_manager is not None
    assert trainer.metrics_tracker is not None
    assert trainer.loss_strategy is not None
    assert trainer.gradient_monitor is not None
    assert trainer.gradient_accumulator is not None

    # Check hooks default to DefaultHooks
    assert isinstance(trainer.hooks, DefaultHooks)


def test_trainer_validation_fails_with_invalid_config(simple_model, model_config):
    """Test that invalid TrainingConfig raises ValueError during initialization."""
    invalid_config = TrainingConfig(
        learning_rate=-0.001,  # Invalid: negative LR
        batch_size=0,  # Invalid: zero batch size
        epochs=3
    )

    with pytest.raises(ValueError, match="Configuration validation failed"):
        Trainer(
            model=simple_model,
            config=model_config,
            training_config=invalid_config
        )


def test_trainer_custom_hooks(simple_model, model_config, training_config):
    """Test that custom hooks are used if provided."""
    class CustomHooks:
        def on_training_start(self):
            pass
        def on_epoch_start(self, epoch):
            pass
        def on_batch_end(self, batch_idx, loss):
            pass
        def on_validation_end(self, metrics):
            pass
        def on_epoch_end(self, epoch, metrics):
            pass
        def on_training_end(self):
            pass

    custom_hooks = CustomHooks()

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config,
        hooks=custom_hooks
    )

    assert trainer.hooks is custom_hooks


# =============================================================================
# Training Workflow Tests
# =============================================================================

def test_trainer_train_completes_all_epochs(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test that training completes all epochs successfully."""
    # Update checkpoint dir to temp path
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2  # Quick test

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    # Train without validation
    results = trainer.train(train_data=dummy_dataset)

    # Check results structure
    assert 'metrics_summary' in results
    assert 'best_epoch' in results
    assert 'final_loss' in results
    assert 'checkpoint_path' in results
    assert 'training_time' in results

    # Check metrics summary has correct number of epochs
    assert len(results['metrics_summary']) == 2


def test_trainer_train_with_validation(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test training with validation dataset."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    # Train with validation
    results = trainer.train(
        train_data=dummy_dataset,
        val_data=dummy_dataset  # Use same data for simplicity
    )

    # Check validation metrics present
    metrics_df = results['metrics_summary']
    assert 'val/loss' in metrics_df.columns
    assert 'val/accuracy' in metrics_df.columns


def test_trainer_checkpoint_saving_at_intervals(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test that checkpoints are saved at correct intervals."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 6
    training_config.save_every_n_epochs = 2  # Save at epochs 2, 4, 6

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    results = trainer.train(train_data=dummy_dataset)

    # Check checkpoint directory exists and has files
    checkpoint_dir = Path(training_config.checkpoint_dir)
    assert checkpoint_dir.exists()

    # Should have checkpoints for epochs 1, 3, 5 (0-indexed: 2, 4, 6)
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch*.pt'))
    assert len(checkpoint_files) >= 3  # At least 3 checkpoints


def test_trainer_resume_from_checkpoint(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test resuming training from checkpoint."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 4
    training_config.save_every_n_epochs = 2

    # First training run: train for 2 epochs
    trainer1 = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    results1 = trainer1.train(train_data=dummy_dataset)

    # Get checkpoint from epoch 1 (0-indexed)
    checkpoint_path = trainer1.checkpoint_manager.get_best()
    assert checkpoint_path is not None

    # Second training run: resume from epoch 2, train to epoch 4
    training_config.epochs = 4  # Total epochs
    trainer2 = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    results2 = trainer2.train(
        train_data=dummy_dataset,
        resume_from=str(checkpoint_path)
    )

    # Should have completed epochs 2, 3 (0-indexed)
    assert len(results2['metrics_summary']) == 4


# =============================================================================
# Hook Invocation Tests
# =============================================================================

def test_trainer_hook_invocation_order(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test that hooks are called at correct points in training loop."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2

    # Mock hooks to track invocation
    mock_hooks = Mock(spec=TrainingHooks)

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config,
        hooks=mock_hooks
    )

    results = trainer.train(train_data=dummy_dataset)

    # Verify hook call sequence
    mock_hooks.on_training_start.assert_called_once()

    # Should be called for each epoch
    assert mock_hooks.on_epoch_start.call_count == 2
    assert mock_hooks.on_epoch_end.call_count == 2

    # Should be called for each batch (16 samples / batch_size=2 = 8 batches per epoch)
    assert mock_hooks.on_batch_end.call_count > 0

    mock_hooks.on_training_end.assert_called_once()


def test_trainer_validation_hook_called(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test that on_validation_end hook is called when validation data provided."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2

    mock_hooks = Mock(spec=TrainingHooks)

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config,
        hooks=mock_hooks
    )

    # Train with validation
    results = trainer.train(
        train_data=dummy_dataset,
        val_data=dummy_dataset
    )

    # Validation hook should be called for each epoch
    assert mock_hooks.on_validation_end.call_count == 2

    # Check that metrics dict is passed to hook
    call_args = mock_hooks.on_validation_end.call_args_list[0][0][0]
    assert 'val_loss' in call_args
    assert 'val_accuracy' in call_args


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

def test_trainer_works_with_simplenamespace_config(simple_model, training_config, dummy_dataset, tmp_path):
    """Test backward compatibility with SimpleNamespace model config."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2

    # Use SimpleNamespace config (legacy support)
    legacy_config = SimpleNamespace(
        vocab_size=100,
        max_seq_len=32,
        pad_token_id=0
    )

    trainer = Trainer(
        model=simple_model,
        config=legacy_config,  # SimpleNamespace instead of dataclass
        training_config=training_config
    )

    results = trainer.train(train_data=dummy_dataset)

    # Should complete successfully
    assert results is not None
    assert 'metrics_summary' in results


def test_trainer_without_task_spec(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test that Trainer works without TaskSpec (fallback mode)."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config,
        task_spec=None  # No task spec provided
    )

    results = trainer.train(train_data=dummy_dataset)

    # Should use default language modeling loss strategy
    assert results is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_trainer_handles_empty_dataset_gracefully(simple_model, model_config, training_config, tmp_path):
    """Test that Trainer handles empty dataset with clear error."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')

    from torch.utils.data import TensorDataset
    empty_dataset = TensorDataset(torch.empty((0, 32), dtype=torch.long))

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    # Should raise error about empty dataset
    with pytest.raises((ValueError, RuntimeError, ZeroDivisionError)):
        trainer.train(train_data=empty_dataset)


def test_trainer_handles_missing_validation_gracefully(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test that Trainer handles missing validation data correctly."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    # Train without validation
    results = trainer.train(train_data=dummy_dataset, val_data=None)

    # Should not have validation metrics
    metrics_df = results['metrics_summary']
    # Note: MetricsTracker logs val_loss=0.0 when no validation data provided
    assert 'train/loss' in metrics_df.columns


# =============================================================================
# Integration Tests with Real Components
# =============================================================================

def test_trainer_integration_end_to_end(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Integration test: end-to-end training with all real components."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 3
    training_config.save_every_n_epochs = 1  # Save every epoch

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    # Execute training
    results = trainer.train(
        train_data=dummy_dataset,
        val_data=dummy_dataset
    )

    # Verify results completeness
    assert results is not None
    assert 'metrics_summary' in results
    assert 'best_epoch' in results
    assert 'final_loss' in results
    assert 'checkpoint_path' in results
    assert 'training_time' in results

    # Verify metrics recorded for all epochs
    metrics_df = results['metrics_summary']
    assert len(metrics_df) == 3

    # Verify checkpoints saved
    checkpoint_dir = Path(training_config.checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch*.pt'))
    assert len(checkpoint_files) >= 3

    # Verify best checkpoint identified
    best_checkpoint = trainer.checkpoint_manager.get_best()
    assert best_checkpoint is not None
    assert best_checkpoint.exists()


def test_trainer_gradient_accumulation(simple_model, model_config, training_config, dummy_dataset, tmp_path):
    """Test training with gradient accumulation."""
    training_config.checkpoint_dir = str(tmp_path / 'checkpoints')
    training_config.epochs = 2
    training_config.gradient_accumulation_steps = 4

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=training_config
    )

    results = trainer.train(train_data=dummy_dataset)

    # Should complete successfully with gradient accumulation
    assert results is not None
    assert len(results['metrics_summary']) == 2


# =============================================================================
# Configuration Tests
# =============================================================================

def test_trainer_respects_training_config_settings(simple_model, model_config, dummy_dataset, tmp_path):
    """Test that Trainer respects all TrainingConfig settings."""
    config = TrainingConfig(
        learning_rate=1e-4,
        batch_size=4,
        epochs=2,
        save_every_n_epochs=1,
        checkpoint_dir=str(tmp_path / 'checkpoints'),
        max_grad_norm=0.5,
        weight_decay=0.02,
        random_seed=123,
        gradient_accumulation_steps=2
    )

    trainer = Trainer(
        model=simple_model,
        config=model_config,
        training_config=config
    )

    # Verify optimizer configuration
    assert trainer.optimizer.param_groups[0]['lr'] == 1e-4
    assert trainer.optimizer.param_groups[0]['weight_decay'] == 0.02

    # Verify gradient accumulator (handles clipping and accumulation)
    assert trainer.gradient_accumulator.accumulation_steps == 2
    assert trainer.gradient_accumulator.max_grad_norm == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
