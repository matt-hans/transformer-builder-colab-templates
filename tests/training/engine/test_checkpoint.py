"""
Unit tests for CheckpointManager.

Tests cover:
1. Save checkpoint after epoch 5 → Load → Verify epoch counter = 6
2. Save with optimizer state → Load → Verify learning rate preserved
3. Corrupt checkpoint file → Load → Raises clear error with recovery steps
4. Multiple checkpoints → get_best() → Returns checkpoint with lowest val_loss
5. Save with custom state → Load → Custom state restored
6. RNG state preservation: train 10 steps, checkpoint, resume → identical random outputs
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil
import tempfile
import json
import random
import numpy as np

from utils.training.engine.checkpoint import CheckpointManager, CheckpointMetadata


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_model():
    """Create simple PyTorch model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    return model


@pytest.fixture
def optimizer_and_scheduler(simple_model):
    """Create optimizer and scheduler."""
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return optimizer, scheduler


class TestCheckpointManager:
    """Test suite for CheckpointManager."""

    def test_save_and_load_basic(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test 1: Save checkpoint after epoch 5 → Load → Verify epoch counter = 6."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            keep_best_k=3,
            keep_last_n=5,
            monitor='val_loss',
            mode='min'
        )

        # Save checkpoint at epoch 5
        metrics = {'val_loss': 0.5, 'train_loss': 0.6}
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            metrics=metrics,
            global_step=500
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_state = manager.load(checkpoint_path)

        # Verify epoch counter should be 6 for next epoch
        assert loaded_state['epoch'] == 5
        next_epoch = loaded_state['epoch'] + 1
        assert next_epoch == 6

        # Verify metrics preserved
        assert loaded_state['metrics']['val_loss'] == 0.5
        assert loaded_state['global_step'] == 500

    def test_optimizer_state_preserved(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test 2: Save with optimizer state → Load → Verify learning rate preserved."""
        optimizer, scheduler = optimizer_and_scheduler

        # Set custom learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Save checkpoint
        metrics = {'val_loss': 0.4}
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            metrics=metrics
        )

        # Load checkpoint
        loaded_state = manager.load(checkpoint_path)

        # Create new optimizer and load state
        new_optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        new_optimizer.load_state_dict(loaded_state['optimizer_state_dict'])

        # Verify learning rate preserved
        assert new_optimizer.param_groups[0]['lr'] == 0.005

    def test_corrupted_checkpoint_handling(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test 3: Corrupt checkpoint file → Load → Raises clear error with recovery steps."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Save valid checkpoint
        metrics = {'val_loss': 0.3}
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            metrics=metrics
        )

        # Corrupt the checkpoint file
        with open(checkpoint_path, 'wb') as f:
            f.write(b'corrupted data')

        # Attempt to load corrupted checkpoint
        with pytest.raises(RuntimeError) as exc_info:
            manager.load(checkpoint_path)

        # Verify error message contains recovery guidance
        error_message = str(exc_info.value)
        assert 'corrupted' in error_message.lower()
        assert 'Recovery' in error_message or 'recovery' in error_message.lower()

    def test_get_best_checkpoint(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test 4: Multiple checkpoints → get_best() → Returns checkpoint with lowest val_loss."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            keep_best_k=3,
            monitor='val_loss',
            mode='min'
        )

        # Save multiple checkpoints with different val_loss
        checkpoints_data = [
            (1, 0.8),
            (2, 0.5),
            (3, 0.3),  # Best
            (4, 0.4),
            (5, 0.6)
        ]

        for epoch, val_loss in checkpoints_data:
            metrics = {'val_loss': val_loss}
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                global_step=epoch * 100
            )

        # Get best checkpoint
        best_checkpoint = manager.get_best()
        assert best_checkpoint is not None

        # Load and verify it's the one with val_loss=0.3 (epoch 3)
        loaded_state = manager.load(best_checkpoint)
        assert loaded_state['metrics']['val_loss'] == 0.3
        assert loaded_state['epoch'] == 3

    def test_custom_state_preservation(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test 5: Save with custom state → Load → Custom state restored."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Create custom state
        custom_state = {
            'loss_strategy_config': {'task_type': 'language_modeling', 'vocab_size': 50257},
            'metrics_tracker_state': {'best_val_loss': 0.35, 'patience_counter': 2},
            'training_config': {'learning_rate': 5e-5, 'batch_size': 4}
        }

        # Save checkpoint with custom state
        metrics = {'val_loss': 0.35}
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=4,
            metrics=metrics,
            custom_state=custom_state
        )

        # Load checkpoint
        loaded_state = manager.load(checkpoint_path)

        # Verify custom state preserved
        assert 'custom_state' in loaded_state
        assert loaded_state['custom_state']['loss_strategy_config']['task_type'] == 'language_modeling'
        assert loaded_state['custom_state']['metrics_tracker_state']['best_val_loss'] == 0.35
        assert loaded_state['custom_state']['training_config']['learning_rate'] == 5e-5

    def test_rng_state_preservation(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test 6: RNG state preservation → train 10 steps, checkpoint, resume → identical outputs."""
        optimizer, scheduler = optimizer_and_scheduler

        # Set initial seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Generate some random numbers before checkpoint
        random_python_before = [random.random() for _ in range(3)]
        random_numpy_before = np.random.rand(3)
        random_torch_before = torch.rand(3)

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Save checkpoint with RNG state
        metrics = {'val_loss': 0.4}
        checkpoint_path = manager.save(
            model=simple_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            metrics=metrics
        )

        # Generate more random numbers after checkpoint
        random_python_after_save = [random.random() for _ in range(3)]
        random_numpy_after_save = np.random.rand(3)
        random_torch_after_save = torch.rand(3)

        # Reset seeds to different values (simulate starting fresh)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        # Load checkpoint (should restore RNG state)
        loaded_state = manager.load(checkpoint_path)

        # Generate random numbers after restore
        random_python_after_restore = [random.random() for _ in range(3)]
        random_numpy_after_restore = np.random.rand(3)
        random_torch_after_restore = torch.rand(3)

        # Verify RNG state restored: values after restore should match values after save
        assert np.allclose(random_python_after_save, random_python_after_restore)
        assert np.allclose(random_numpy_after_save, random_numpy_after_restore)
        assert torch.allclose(random_torch_after_save, random_torch_after_restore)

    def test_checkpoint_retention_policy(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test checkpoint cleanup with keep_best_k and keep_last_n."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            keep_best_k=2,
            keep_last_n=2,
            monitor='val_loss',
            mode='min'
        )

        # Save 5 checkpoints
        for epoch in range(1, 6):
            metrics = {'val_loss': 0.1 * epoch}  # Increasing loss
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                global_step=epoch * 100
            )

        # Verify retention: keep_best_k=2 (epochs 1,2) + keep_last_n=2 (epochs 4,5)
        # Expected: epochs 1, 2, 4, 5 retained (epoch 3 deleted)
        checkpoints = manager.list_checkpoints()
        epochs_retained = {ckpt.epoch for ckpt in checkpoints}

        assert len(epochs_retained) == 4
        assert 1 in epochs_retained  # Best
        assert 2 in epochs_retained  # Second best
        assert 4 in epochs_retained  # Second to last
        assert 5 in epochs_retained  # Last

    def test_list_checkpoints(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test list_checkpoints returns sorted list."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Save multiple checkpoints
        for epoch in [3, 1, 5, 2, 4]:
            metrics = {'val_loss': 0.5}
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics
            )

        # List checkpoints (should be sorted by epoch descending)
        checkpoints = manager.list_checkpoints()
        epochs = [ckpt.epoch for ckpt in checkpoints]

        assert epochs == [5, 4, 3, 2, 1]

    def test_no_checkpoints_error(self, temp_checkpoint_dir):
        """Test error when no checkpoints exist."""
        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Attempt to get best checkpoint when none exist
        with pytest.raises(FileNotFoundError) as exc_info:
            manager.load()  # Load with no path defaults to best

        assert 'No checkpoints found' in str(exc_info.value)

    def test_monitor_metric_missing_error(self, temp_checkpoint_dir, simple_model, optimizer_and_scheduler):
        """Test error when monitor metric not in metrics dict."""
        optimizer, scheduler = optimizer_and_scheduler

        manager = CheckpointManager(
            checkpoint_dir=str(temp_checkpoint_dir),
            monitor='val_loss',
            mode='min'
        )

        # Try to save with missing monitor metric
        metrics = {'train_loss': 0.5}  # Missing val_loss
        with pytest.raises(ValueError) as exc_info:
            manager.save(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                metrics=metrics
            )

        error_message = str(exc_info.value)
        assert 'val_loss' in error_message
        assert 'not found' in error_message.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
