"""Tests for Trainer data quality validation."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
import tempfile
import shutil

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig


def test_trainer_validation_passes_with_good_data():
    """Test validation passes when data quality is good."""
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100, 64)

            def forward(self, input_ids):
                return self.embedding(input_ids).mean(dim=1)

        # Create trainer
        model = MockModel()
        config = SimpleNamespace(vocab_size=100, d_model=64)
        training_config = TrainingConfig(batch_size=4, epochs=1, checkpoint_dir=temp_dir)
        task_spec = SimpleNamespace(modality='text', task_type='lm')

        class MockTokenizer:
            pad_token_id = 0

        trainer = Trainer(model, config, training_config, task_spec, tokenizer=MockTokenizer())

        # Create good data loader (all sequences >= 2 tokens)
        good_data = TensorDataset(
            torch.randint(0, 100, (16, 10)),  # 16 samples, 10 tokens each
        )
        good_loader = DataLoader(good_data, batch_size=4)

        # Should pass validation (no exception)
        trainer._validate_data_quality(good_loader)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_trainer_validation_fails_with_short_sequences():
    """Test validation fails when preprocessing was skipped."""
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(100, 64)

            def forward(self, input_ids):
                return self.embedding(input_ids).mean(dim=1)

        # Create trainer
        model = MockModel()
        config = SimpleNamespace(vocab_size=100, d_model=64)
        training_config = TrainingConfig(batch_size=4, epochs=1, checkpoint_dir=temp_dir)
        task_spec = SimpleNamespace(modality='text', task_type='lm')

        class MockTokenizer:
            pad_token_id = 0

        trainer = Trainer(model, config, training_config, task_spec, tokenizer=MockTokenizer())

        # Create custom collate function that preserves variable lengths
        # This simulates what happens when user skips preprocessing
        def custom_collate(batch):
            # Return dict with ragged input_ids (different lengths)
            return {
                'input_ids': [
                    torch.tensor([1]),  # 1 token - TOO SHORT
                    torch.tensor([2, 3, 4, 5]),  # 4 tokens - GOOD
                    torch.tensor([6]),  # 1 token - TOO SHORT
                    torch.tensor([7, 8, 9, 10]),  # 4 tokens - GOOD
                ]
            }

        # Create minimal dataset (collate_fn does the work)
        short_data = TensorDataset(torch.tensor([0, 1, 2, 3]))  # Dummy data
        short_loader = DataLoader(short_data, batch_size=4, collate_fn=custom_collate)

        # Should raise ValueError with "preprocessing was skipped"
        with pytest.raises(ValueError) as exc_info:
            trainer._validate_data_quality(short_loader)

        assert "preprocessing was skipped" in str(exc_info.value).lower()

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
