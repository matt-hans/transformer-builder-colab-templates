"""
Tests for checkpoint recovery utilities.

Verifies that training results can be recovered from checkpoints
for interrupted training, analysis, and resume workflows.
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
from utils.training.engine.recovery import recover_training_results, list_checkpoints


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_checkpoint_with_metrics(tmp_path):
    """Create a checkpoint with full metrics history."""
    metrics_history = [
        {'epoch': 0, 'train/loss': 4.5, 'val/loss': 4.2, 'val/perplexity': 66.7},
        {'epoch': 1, 'train/loss': 3.8, 'val/loss': 3.9, 'val/perplexity': 49.4},
        {'epoch': 2, 'train/loss': 3.2, 'val/loss': 3.7, 'val/perplexity': 40.4},
    ]

    checkpoint = {
        'epoch': 2,
        'global_step': 300,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'metrics': {'train_loss': 3.2, 'val_loss': 3.7},
        'custom_state': {
            'metrics_history': metrics_history,
            'training_time': 1200.5,
            'training_config': {'epochs': 3}
        }
    }

    ckpt_path = tmp_path / 'checkpoint_epoch0002_step000300_20251122_123456.pt'
    torch.save(checkpoint, ckpt_path)

    return ckpt_path, metrics_history


@pytest.fixture
def mock_checkpoint_without_metrics(tmp_path):
    """Create a checkpoint without metrics history (legacy)."""
    checkpoint = {
        'epoch': 5,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'metrics': {'train_loss': 2.5, 'val_loss': 2.8}
        # No custom_state
    }

    ckpt_path = tmp_path / 'checkpoint_epoch0005_step000500_20251122_100000.pt'
    torch.save(checkpoint, ckpt_path)

    return ckpt_path


@pytest.fixture
def multiple_checkpoints(tmp_path):
    """Create multiple checkpoints for testing list/best selection."""
    checkpoints = []

    for epoch in range(5):
        metrics_history = [
            {'epoch': i, 'train/loss': 5.0 - i * 0.5, 'val/loss': 5.0 - i * 0.3}
            for i in range(epoch + 1)
        ]

        checkpoint = {
            'epoch': epoch,
            'global_step': (epoch + 1) * 100,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'metrics': {'train_loss': metrics_history[-1]['train/loss'], 'val_loss': metrics_history[-1]['val/loss']},
            'custom_state': {
                'metrics_history': metrics_history,
                'training_time': (epoch + 1) * 300.0
            },
            'timestamp': f'20251122_1234{epoch:02d}'
        }

        ckpt_path = tmp_path / f'checkpoint_epoch{epoch:04d}_step{(epoch+1)*100:06d}_20251122_1234{epoch:02d}.pt'
        torch.save(checkpoint, ckpt_path)
        checkpoints.append(ckpt_path)

    return tmp_path, checkpoints


# ============================================================================
# Test recover_training_results()
# ============================================================================

def test_recover_training_results_with_checkpoint_path(mock_checkpoint_with_metrics):
    """Test recovery from specific checkpoint path."""
    ckpt_path, expected_metrics = mock_checkpoint_with_metrics

    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # Verify return structure matches Trainer.train() format
    assert 'metrics_summary' in results
    assert 'best_epoch' in results
    assert 'final_loss' in results
    assert 'checkpoint_path' in results
    assert 'training_time' in results
    assert 'loss_history' in results  # Backward compatibility
    assert 'val_loss_history' in results  # Backward compatibility

    # Verify data integrity
    assert isinstance(results['metrics_summary'], pd.DataFrame)
    assert len(results['metrics_summary']) == 3  # 3 epochs
    assert results['best_epoch'] == 2
    assert results['final_loss'] == 3.2
    assert results['training_time'] == 1200.5

    # Verify backward compatibility
    assert results['loss_history'] == [4.5, 3.8, 3.2]
    assert results['val_loss_history'] == [4.2, 3.9, 3.7]

    # Verify DataFrame columns
    assert 'train/loss' in results['metrics_summary'].columns
    assert 'val/loss' in results['metrics_summary'].columns


def test_recover_training_results_with_checkpoint_dir(multiple_checkpoints):
    """Test recovery from best checkpoint in directory."""
    ckpt_dir, ckpt_paths = multiple_checkpoints

    # Recover best checkpoint (lowest val_loss = epoch 4)
    results = recover_training_results(
        checkpoint_dir=str(ckpt_dir),
        monitor='val_loss',
        mode='min'
    )

    # Should recover from epoch 4 (best val_loss)
    assert results['best_epoch'] == 4
    assert len(results['loss_history']) == 5  # epochs 0-4


def test_recover_training_results_missing_checkpoint(tmp_path):
    """Test error handling for missing checkpoint."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        recover_training_results(checkpoint_path=str(tmp_path / 'nonexistent.pt'))


def test_recover_training_results_missing_metrics_history(mock_checkpoint_without_metrics):
    """Test error handling for checkpoint without metrics_history."""
    ckpt_path = mock_checkpoint_without_metrics

    with pytest.raises(ValueError, match="does not contain metrics_history"):
        recover_training_results(checkpoint_path=str(ckpt_path))


def test_recover_training_results_no_val_data(tmp_path):
    """Test recovery when checkpoint only has training data (no validation)."""
    metrics_history = [
        {'epoch': 0, 'train/loss': 4.5},
        {'epoch': 1, 'train/loss': 3.8},
    ]

    checkpoint = {
        'epoch': 1,
        'model_state_dict': {},
        'custom_state': {
            'metrics_history': metrics_history,
        }
    }

    ckpt_path = tmp_path / 'checkpoint_train_only.pt'
    torch.save(checkpoint, ckpt_path)

    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # Should work but val_loss_history is empty
    assert results['loss_history'] == [4.5, 3.8]
    assert results['val_loss_history'] == []  # No validation data


def test_recover_training_results_empty_metrics(tmp_path):
    """Test recovery with empty metrics history (edge case)."""
    checkpoint = {
        'epoch': 0,
        'model_state_dict': {},
        'custom_state': {
            'metrics_history': []  # Empty
        }
    }

    ckpt_path = tmp_path / 'checkpoint_empty.pt'
    torch.save(checkpoint, ckpt_path)

    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # Should return empty lists
    assert results['loss_history'] == []
    assert results['val_loss_history'] == []
    assert results['final_loss'] == 0.0  # Default


# ============================================================================
# Test list_checkpoints()
# ============================================================================

def test_list_checkpoints(multiple_checkpoints):
    """Test listing all checkpoints in directory."""
    ckpt_dir, ckpt_paths = multiple_checkpoints

    checkpoints = list_checkpoints(str(ckpt_dir))

    # Should find all 5 checkpoints
    assert len(checkpoints) == 5

    # Should be sorted by epoch (descending)
    assert checkpoints[0]['epoch'] == 4
    assert checkpoints[-1]['epoch'] == 0

    # Verify metadata
    for ckpt in checkpoints:
        assert 'path' in ckpt
        assert 'filename' in ckpt
        assert 'epoch' in ckpt
        assert 'train_loss' in ckpt
        assert 'val_loss' in ckpt


def test_list_checkpoints_empty_dir(tmp_path):
    """Test listing checkpoints in empty directory."""
    checkpoints = list_checkpoints(str(tmp_path))
    assert checkpoints == []


def test_list_checkpoints_nonexistent_dir():
    """Test listing checkpoints in nonexistent directory."""
    checkpoints = list_checkpoints('/nonexistent/directory')
    assert checkpoints == []


def test_list_checkpoints_with_corrupted_file(tmp_path):
    """Test that corrupted checkpoints are skipped."""
    # Create a valid checkpoint
    valid_ckpt = {
        'epoch': 1,
        'model_state_dict': {},
        'metrics': {'train_loss': 3.0, 'val_loss': 3.2}
    }
    valid_path = tmp_path / 'checkpoint_epoch0001.pt'
    torch.save(valid_ckpt, valid_path)

    # Create a corrupted checkpoint (invalid file)
    corrupted_path = tmp_path / 'checkpoint_epoch0002.pt'
    corrupted_path.write_text('corrupted data')

    # Should return only the valid checkpoint
    checkpoints = list_checkpoints(str(tmp_path))
    assert len(checkpoints) == 1
    assert checkpoints[0]['epoch'] == 1


# ============================================================================
# Test Integration with Notebook Workflow
# ============================================================================

def test_notebook_workflow_recovery(mock_checkpoint_with_metrics):
    """Test that recovery works exactly like notebook Cell 32 expects."""
    ckpt_path, _ = mock_checkpoint_with_metrics

    # Recover results
    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # Simulate notebook code (Cell 32 final metrics)
    train_loss = results['loss_history'][-1]  # Should not raise KeyError
    assert train_loss == 3.2

    # Simulate notebook code (ExperimentDB logging)
    for epoch, loss in enumerate(results['loss_history']):
        assert isinstance(loss, (int, float))
        assert epoch >= 0

    # Simulate notebook code (validation metrics)
    if 'metrics_summary' in results and not results['metrics_summary'].empty:
        final_metrics = results['metrics_summary'].iloc[-1]
        if 'val/loss' in final_metrics:
            val_loss = final_metrics['val/loss']
            assert val_loss == 3.7


def test_backward_compatibility_with_v3_notebook_code(mock_checkpoint_with_metrics):
    """Test that recovery provides v3.x-compatible fields."""
    ckpt_path, _ = mock_checkpoint_with_metrics

    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # v3.x notebooks expect these fields
    assert isinstance(results['loss_history'], list)
    assert isinstance(results['val_loss_history'], list)
    assert all(isinstance(x, (int, float)) for x in results['loss_history'])


# ============================================================================
# Test Error Messages
# ============================================================================

def test_helpful_error_messages():
    """Test that error messages guide users to solutions."""
    with pytest.raises(ValueError, match="Must provide either checkpoint_path or checkpoint_dir"):
        recover_training_results()  # No arguments


def test_recovery_requires_v4_checkpoint(mock_checkpoint_without_metrics):
    """Test that error message explains v4.0 requirement."""
    ckpt_path = mock_checkpoint_without_metrics

    with pytest.raises(ValueError, match="before v4.0 or training failed"):
        recover_training_results(checkpoint_path=str(ckpt_path))


def test_recover_with_session_metadata_v4(tmp_path):
    """Test that recovery extracts workspace_root and run_name from v4.0+ checkpoint."""
    metrics_history = [
        {'epoch': 0, 'train/loss': 4.5, 'val/loss': 4.2},
        {'epoch': 1, 'train/loss': 3.8, 'val/loss': 3.9},
    ]

    checkpoint = {
        'epoch': 1,
        'model_state_dict': {},
        'custom_state': {
            'metrics_history': metrics_history,
            'workspace_root': '/content/workspace',
            'run_name': 'run_20251122_065455',
        }
    }

    ckpt_path = tmp_path / 'checkpoint_epoch0001.pt'
    torch.save(checkpoint, ckpt_path)

    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # Verify session metadata extracted from checkpoint
    assert results['workspace_root'] == '/content/workspace'
    assert results['run_name'] == 'run_20251122_065455'
    assert results['best_epoch'] == 1
    assert len(results['loss_history']) == 2


def test_recover_legacy_checkpoint_fallback(tmp_path):
    """Test that recovery falls back to path parsing for v3.x checkpoints."""
    metrics_history = [
        {'epoch': 0, 'train/loss': 4.5, 'val/loss': 4.2},
    ]

    # v3.x checkpoint without session metadata
    checkpoint = {
        'epoch': 0,
        'model_state_dict': {},
        'custom_state': {
            'metrics_history': metrics_history,
            # No workspace_root or run_name
        }
    }

    # Create checkpoint in checkpoints subdirectory
    checkpoints_dir = tmp_path / 'checkpoints'
    checkpoints_dir.mkdir()
    ckpt_path = checkpoints_dir / 'checkpoint_run_20251122_epoch0000.pt'
    torch.save(checkpoint, ckpt_path)

    results = recover_training_results(checkpoint_path=str(ckpt_path))

    # Should fallback to path parsing
    assert results['workspace_root'] == str(tmp_path)  # parent.parent
    assert 'checkpoint' in results['run_name']  # Parsed from filename
