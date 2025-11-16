"""
Simple smoke tests for gradient accumulation (fast, no mocking).

These tests verify basic functionality without complex mocking or long training loops.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.tier3_training_utilities import test_fine_tuning


class TinyModel(nn.Module):
    """Minimal model for fast testing."""

    def __init__(self, vocab_size=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.linear = nn.Linear(16, vocab_size)

    def forward(self, input_ids):
        return self.linear(self.embedding(input_ids))


class TestGradientAccumulationSmoke:
    """Smoke tests to verify gradient accumulation doesn't crash and produces valid output."""

    def test_training_completes_with_accumulation(self):
        """
        Scenario: Run training with gradient_accumulation_steps=2
        Input: Small dataset, accum_steps=2
        Expected: Training completes successfully
        Why: Validates basic functionality doesn't crash
        Contract: Returns valid result dict with loss history
        """
        torch.manual_seed(42)

        model = TinyModel(vocab_size=30)
        config = SimpleNamespace(vocab_size=30)

        # Create small dataset (4 samples)
        train_data = [torch.randint(0, 30, (8,)) for _ in range(4)]

        # Run training
        result = test_fine_tuning(
            model=model,
            config=config,
            train_data=train_data,
            val_data=train_data[:2],
            n_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=2,
            use_wandb=False,
            use_amp=False
        )

        # Verify result structure
        assert 'loss_history' in result
        assert 'final_loss' in result
        assert len(result['loss_history']) > 0
        assert result['final_loss'] > 0

    def test_effective_batch_size_printed(self, capfd):
        """
        Scenario: Training with gradient_accumulation_steps=3
        Input: batch_size=2, accum_steps=3
        Expected: Effective batch size = 6 printed to stdout
        Why: Validates configuration is logged correctly
        Contract: "Effective batch size: 6" appears in output
        """
        torch.manual_seed(42)

        model = TinyModel(vocab_size=30)
        config = SimpleNamespace(vocab_size=30)
        train_data = [torch.randint(0, 30, (8,)) for _ in range(3)]

        test_fine_tuning(
            model=model,
            config=config,
            train_data=train_data,
            val_data=train_data[:1],
            n_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=3,
            use_wandb=False,
            use_amp=False
        )

        captured = capfd.readouterr()
        assert 'Effective batch size: 6' in captured.out
        assert 'Gradient accumulation steps: 3' in captured.out

    def test_default_accumulation_is_one(self, capfd):
        """
        Scenario: Training without specifying gradient_accumulation_steps
        Input: Default parameters
        Expected: gradient_accumulation_steps defaults to 1
        Why: Validates backward compatibility
        Contract: Default behavior unchanged
        """
        torch.manual_seed(42)

        model = TinyModel(vocab_size=30)
        config = SimpleNamespace(vocab_size=30)
        train_data = [torch.randint(0, 30, (8,)) for _ in range(3)]

        test_fine_tuning(
            model=model,
            config=config,
            train_data=train_data,
            val_data=train_data[:1],
            n_epochs=1,
            batch_size=2,
            use_wandb=False,
            use_amp=False
            # Note: gradient_accumulation_steps not specified
        )

        captured = capfd.readouterr()
        # Default is 1, so effective batch size = batch_size
        assert 'Effective batch size: 2' in captured.out
        assert 'Gradient accumulation steps: 1' in captured.out

    def test_loss_decreases_with_accumulation(self):
        """
        Scenario: Train for multiple epochs with gradient accumulation
        Input: accum_steps=2, n_epochs=2
        Expected: Loss decreases from epoch 0 to epoch 1
        Why: Validates training is actually working
        Contract: final_loss < initial_loss
        """
        torch.manual_seed(42)

        model = TinyModel(vocab_size=30)
        config = SimpleNamespace(vocab_size=30)
        train_data = [torch.randint(0, 30, (8,)) for _ in range(8)]

        result = test_fine_tuning(
            model=model,
            config=config,
            train_data=train_data,
            val_data=train_data[:2],
            n_epochs=2,  # Multiple epochs to see loss decrease
            batch_size=2,
            gradient_accumulation_steps=2,
            use_wandb=False,
            use_amp=False
        )

        # Loss should decrease
        assert result['final_loss'] < result['initial_loss'], (
            f"Loss should decrease: initial={result['initial_loss']:.4f}, "
            f"final={result['final_loss']:.4f}"
        )
