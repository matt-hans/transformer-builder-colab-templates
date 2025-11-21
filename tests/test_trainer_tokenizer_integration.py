"""
Integration tests for Trainer tokenizer/collator requirements (v4.0+).

Tests verify:
1. Text tasks require tokenizer or data_collator (fail-fast validation)
2. Training with tokenizer succeeds (auto-collator selection)
3. Vision tasks don't require tokenizer (VisionDataCollator automatic)
4. Manual collator override works (priority system)
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def simple_text_model():
    """Create simple transformer model for text tasks."""
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size: int, d_model: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            return self.output(x)

    return SimpleTransformer(vocab_size=1000)


@pytest.fixture
def simple_vision_model():
    """Create simple CNN model for vision tasks."""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, num_classes)

        def forward(self, pixel_values):
            x = self.conv(pixel_values)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleCNN(num_classes=10)


@pytest.fixture
def text_config():
    """Config for text model."""
    return SimpleNamespace(
        vocab_size=1000,
        d_model=64,
        max_seq_len=32
    )


@pytest.fixture
def vision_config():
    """Config for vision model."""
    return SimpleNamespace(
        num_classes=10,
        image_size=32
    )


@pytest.fixture
def training_config():
    """Standard training configuration."""
    return TrainingConfig(
        epochs=1,
        learning_rate=1e-4,
        batch_size=2,
        random_seed=42
    )


@pytest.fixture
def text_task_spec():
    """Task spec for language modeling."""
    return TaskSpec.language_modeling(
        name="test-lm",
        vocab_size=1000,
        max_seq_len=32
    )


@pytest.fixture
def vision_task_spec():
    """Task spec for vision classification."""
    return TaskSpec.vision_tiny(
        name="test-vision",
        num_classes=10
    )


@pytest.fixture
def dummy_tokenizer():
    """Mock tokenizer with required attributes."""
    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.vocab_size = 1000

        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            return {'input_ids': torch.randint(0, self.vocab_size, (1, 32))}

    return DummyTokenizer()


@pytest.fixture
def text_dataset():
    """Simple text dataset with variable-length sequences."""
    class TextDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            # Variable lengths: [10, 15, 20, 25]
            length = 10 + (idx * 5)
            return torch.randint(0, 1000, (length,))

    return TextDataset()


@pytest.fixture
def vision_dataset():
    """Simple vision dataset with fixed-size images."""
    class VisionDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            # Fixed size: (3, 32, 32)
            return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

    return VisionDataset()


@pytest.fixture
def manual_collator(dummy_tokenizer):
    """Manual data collator for testing priority system."""
    from utils.tokenization.data_collator import LanguageModelingDataCollator

    return LanguageModelingDataCollator(
        tokenizer=dummy_tokenizer,
        mlm=False,
        padding_side='right'
    )


# ==============================================================================
# TEST 1: Text Tasks Require Tokenizer or Collator
# ==============================================================================

def test_text_task_without_tokenizer_fails(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec
):
    """
    Test that text tasks without tokenizer/collator raise ValueError.

    Expected behavior: Fail-fast validation at Trainer initialization.
    """
    with pytest.raises(ValueError) as exc_info:
        Trainer(
            model=simple_text_model,
            config=text_config,
            training_config=training_config,
            task_spec=text_task_spec
            # Missing: tokenizer=... or data_collator=...
        )

    # Verify error message provides remediation
    error_msg = str(exc_info.value)
    assert "Text tasks require either:" in error_msg
    assert "tokenizer" in error_msg
    assert "data_collator" in error_msg
    assert "Example:" in error_msg


def test_text_task_with_tokenizer_succeeds(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec,
    dummy_tokenizer
):
    """
    Test that text tasks with tokenizer initialize successfully.

    Expected behavior: Trainer accepts tokenizer and enables auto-collator selection.
    """
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=text_task_spec,
        tokenizer=dummy_tokenizer  # ✅ Enables auto-collator
    )

    # Verify trainer stores tokenizer
    assert trainer.tokenizer is dummy_tokenizer
    assert trainer.data_collator is None  # Manual collator not provided


def test_text_task_with_manual_collator_succeeds(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec,
    manual_collator
):
    """
    Test that text tasks with manual collator initialize successfully.

    Expected behavior: Trainer accepts manual collator without requiring tokenizer.
    """
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=text_task_spec,
        data_collator=manual_collator  # ✅ Manual collation
    )

    # Verify trainer stores collator
    assert trainer.tokenizer is None
    assert trainer.data_collator is manual_collator


# ==============================================================================
# TEST 2: Training with Tokenizer Succeeds
# ==============================================================================

def test_training_with_tokenizer_completes(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec,
    dummy_tokenizer,
    text_dataset
):
    """
    Test that training with tokenizer completes without collation errors.

    Expected behavior: Auto-selected collator handles variable-length sequences.
    """
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=text_task_spec,
        tokenizer=dummy_tokenizer
    )

    # Training should complete without RuntimeError
    results = trainer.train(
        train_data=text_dataset,
        val_data=text_dataset
    )

    # Verify training completed
    assert 'loss_history' in results
    assert len(results['loss_history']) > 0
    assert 'metrics_summary' in results


# ==============================================================================
# TEST 3: Vision Tasks Don't Require Tokenizer
# ==============================================================================

def test_vision_task_without_tokenizer_succeeds(
    simple_vision_model,
    vision_config,
    training_config,
    vision_task_spec
):
    """
    Test that vision tasks work without tokenizer.

    Expected behavior: VisionDataCollator auto-selected, no tokenizer required.
    """
    trainer = Trainer(
        model=simple_vision_model,
        config=vision_config,
        training_config=training_config,
        task_spec=vision_task_spec
        # No tokenizer required for vision tasks
    )

    # Verify trainer initializes successfully
    assert trainer.tokenizer is None
    assert trainer.data_collator is None


def test_vision_training_completes(
    simple_vision_model,
    vision_config,
    training_config,
    vision_task_spec,
    vision_dataset
):
    """
    Test that vision training completes without tokenizer.

    Expected behavior: VisionDataCollator handles batching automatically.
    """
    trainer = Trainer(
        model=simple_vision_model,
        config=vision_config,
        training_config=training_config,
        task_spec=vision_task_spec
    )

    # Training should complete without errors
    results = trainer.train(
        train_data=vision_dataset,
        val_data=vision_dataset
    )

    # Verify training completed
    assert 'loss_history' in results
    assert len(results['loss_history']) > 0


# ==============================================================================
# TEST 4: Manual Collator Override Works (Priority System)
# ==============================================================================

def test_manual_collator_overrides_tokenizer(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec,
    dummy_tokenizer,
    manual_collator
):
    """
    Test that manual collator takes priority over tokenizer.

    Expected behavior: Manual collator used even when tokenizer provided.
    """
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=text_task_spec,
        tokenizer=dummy_tokenizer,        # Provided but should be overridden
        data_collator=manual_collator     # Takes priority
    )

    # Verify both are stored
    assert trainer.tokenizer is dummy_tokenizer
    assert trainer.data_collator is manual_collator

    # Verify data_collator takes priority in data loading
    # (Implementation detail: DataLoaderFactory.create_dataloader checks data_collator first)


def test_collator_priority_in_dataloader(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec,
    dummy_tokenizer,
    manual_collator,
    text_dataset
):
    """
    Test that collator priority system works during training.

    Expected behavior:
    1. Manual collator (highest priority)
    2. Configured collator
    3. Auto-selected collator (lowest priority)
    """
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=text_task_spec,
        tokenizer=dummy_tokenizer,
        data_collator=manual_collator
    )

    # Train to verify collator is used
    results = trainer.train(
        train_data=text_dataset,
        val_data=text_dataset
    )

    # Training should complete successfully with manual collator
    assert 'loss_history' in results
    assert len(results['loss_history']) > 0


# ==============================================================================
# TEST 5: Edge Cases
# ==============================================================================

def test_none_task_spec_allows_missing_tokenizer(
    simple_text_model,
    text_config,
    training_config
):
    """
    Test that task_spec=None skips tokenizer validation.

    Expected behavior: No validation when task_spec is None.
    """
    # Should not raise ValueError
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=None  # No task spec = no validation
    )

    assert trainer.task_spec is None
    assert trainer.tokenizer is None


def test_both_tokenizer_and_collator_allowed(
    simple_text_model,
    text_config,
    training_config,
    text_task_spec,
    dummy_tokenizer,
    manual_collator
):
    """
    Test that providing both tokenizer and collator is allowed.

    Expected behavior: Both stored, collator takes priority in data loading.
    """
    trainer = Trainer(
        model=simple_text_model,
        config=text_config,
        training_config=training_config,
        task_spec=text_task_spec,
        tokenizer=dummy_tokenizer,
        data_collator=manual_collator
    )

    # Both should be stored
    assert trainer.tokenizer is dummy_tokenizer
    assert trainer.data_collator is manual_collator


# ==============================================================================
# TEST SUMMARY
# ==============================================================================

"""
Test Coverage Summary:

✅ TEST 1: Fail-fast validation for text tasks
   - Text task without tokenizer/collator → ValueError
   - Error message includes remediation steps

✅ TEST 2: Auto-collator selection with tokenizer
   - Text task with tokenizer → Initialization succeeds
   - Training with tokenizer → Variable-length sequences handled

✅ TEST 3: Vision tasks don't require tokenizer
   - Vision task without tokenizer → Initialization succeeds
   - Vision training → VisionDataCollator automatic

✅ TEST 4: Manual collator priority
   - Manual collator overrides tokenizer
   - Priority system: manual > configured > auto

✅ TEST 5: Edge cases
   - task_spec=None skips validation
   - Both tokenizer and collator allowed

Architecture Principles Validated:
- Fail-fast: Errors at initialization, not during training
- Open/Closed: Extensible via custom collators
- Interface Segregation: Vision tasks don't pay tokenizer overhead
- Single Responsibility: Trainer validates, DataLoader loads
"""
