"""
Shared test fixtures and utilities.

This module provides common test fixtures to reduce code duplication
and maintain consistency across test files.

Available fixtures:
- tracked_adamw_factory: Factory for optimizer step tracking
- simple_model: Minimal transformer model for testing
- model_config: Simple model configuration (SimpleNamespace)
- training_config: TrainingConfig with sensible test defaults
- task_spec: Default task specification (lm_tiny)
- dummy_dataset: Small synthetic dataset for training
- temp_checkpoint_dir: Temporary directory for checkpoints (auto-cleanup)
- temp_registry_db: Temporary SQLite database (auto-cleanup)
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from pathlib import Path
import tempfile
import shutil

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


# =============================================================================
# Optimizer Tracking Fixtures
# =============================================================================

@pytest.fixture
def tracked_adamw_factory():
    """
    Factory fixture that creates a TrackedAdamW class for testing optimizer step counts.

    Returns a tuple of (TrackedAdamW class, step_calls list) where:
    - TrackedAdamW: Subclass of torch.optim.AdamW that tracks step() calls
    - step_calls: List that accumulates step counts for verification

    Usage:
        TrackedAdamW, step_calls = tracked_adamw_factory
        with patch('module.torch.optim.AdamW', TrackedAdamW):
            # Run code that creates AdamW optimizer
            # Verify: assert len(step_calls) == expected_count
    """
    step_calls = []

    def track_step(original_step):
        """Wrapper that records step calls while preserving original behavior."""
        def wrapper(closure=None):
            step_calls.append(1)
            return original_step(closure)
        # Preserve __func__ attribute for PyTorch scheduler compatibility
        wrapper.__func__ = original_step
        return wrapper

    original_adamw = torch.optim.AdamW

    class TrackedAdamW(original_adamw):
        """AdamW variant that tracks step() calls for test verification."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Wrap step method after initialization
            self.step = track_step(super().step)

    return TrackedAdamW, step_calls


# =============================================================================
# Model and Config Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """
    Simple transformer model for testing.

    Returns a minimal PyTorch model with:
    - Embedding layer (vocab_size=100, d_model=64)
    - Linear projection to logits
    - Forward pass returns dict with 'logits' key
    """
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
    """
    Simple model configuration (SimpleNamespace).

    Contains standard transformer hyperparameters:
    - vocab_size: 100
    - max_seq_len: 32
    - d_model: 64
    - num_layers: 2
    - num_heads: 4
    - pad_token_id: 0
    """
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
    """
    TrainingConfig with sensible test defaults.

    Configured for fast tests:
    - learning_rate: 5e-5
    - batch_size: 2
    - epochs: 3
    - No W&B logging
    - Fast mode (deterministic=False)
    """
    from utils.training.training_config import TrainingConfig

    return TrainingConfig(
        learning_rate=5e-5,
        batch_size=2,
        epochs=3,
        save_every_n_epochs=2,
        checkpoint_dir='/tmp/test_checkpoints',
        wandb_project=None,  # Disable W&B for tests
        gradient_accumulation_steps=1,
        random_seed=42,
        deterministic=False  # Fast mode for tests
    )


@pytest.fixture
def task_spec():
    """
    Default task specification (lm_tiny).

    Returns the minimal language modeling task spec from defaults.
    """
    from utils.training.task_spec import get_default_task_specs

    return get_default_task_specs()['lm_tiny']


# =============================================================================
# Dataset Fixtures
# =============================================================================

@pytest.fixture
def dummy_dataset():
    """
    Small synthetic dataset for testing.

    Returns a TensorDataset with:
    - 16 samples
    - Sequence length: 32
    - Vocab range: [0, 100)
    - Contains (input_ids, labels) tuples
    """
    from torch.utils.data import TensorDataset

    # Create random data
    input_ids = torch.randint(0, 100, (16, 32))  # 16 samples, seq_len=32
    labels = torch.randint(0, 100, (16, 32))

    return TensorDataset(input_ids, labels)


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """
    Temporary directory for checkpoint testing with auto-cleanup.

    Returns a Path object to a fresh directory that is automatically
    cleaned up after the test completes.
    """
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    yield checkpoint_dir
    # Cleanup happens automatically via tmp_path


@pytest.fixture
def temp_registry_db(tmp_path):
    """
    Temporary SQLite database for model registry testing with auto-cleanup.

    Returns a Path object to a temporary .db file that is automatically
    removed after the test completes.
    """
    db_path = tmp_path / 'test_registry.db'
    yield db_path
    # Cleanup happens automatically via tmp_path
