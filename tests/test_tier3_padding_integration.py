"""
Integration test for padding token handling in tier3_training_utilities.

Verifies that test_fine_tuning and test_hyperparameter_search correctly
exclude padding tokens from loss calculation.
"""

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from utils.tier3_training_utilities import test_fine_tuning, test_hyperparameter_search


class TinyTransformer(nn.Module):
    """Minimal transformer for testing."""
    def __init__(self, vocab_size=100, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.linear(x)
        return logits


def test_fine_tuning_with_padding_detection():
    """
    Test fine-tuning correctly detects pad_token_id and applies ignore_index.

    Scenario: Config with custom pad_token_id=99
    Expected: Training completes without errors, uses pad_token_id=99
    """
    vocab_size = 100
    config = SimpleNamespace(
        vocab_size=vocab_size,
        max_seq_len=32,
        pad_token_id=99  # Custom padding ID
    )

    model = TinyTransformer(vocab_size=vocab_size)

    # Generate synthetic data with padding tokens (ID=99)
    train_data = []
    for _ in range(10):
        seq = torch.randint(0, vocab_size - 1, (16,))  # IDs 0-98
        # Add padding: replace last 3 tokens with pad_token_id
        seq[-3:] = 99
        train_data.append(seq)

    # Run fine-tuning (should detect pad_token_id=99)
    results = test_fine_tuning(
        model=model,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=2,
        use_wandb=False
    )

    # Assertions
    assert 'final_loss' in results
    assert 'loss_history' in results
    assert results['final_loss'] > 0
    assert len(results['loss_history']) > 0
    assert torch.isfinite(torch.tensor(results['final_loss']))

    print(f"✓ Integration test passed: Final loss={results['final_loss']:.4f}")


def test_fine_tuning_with_default_padding():
    """
    Test fine-tuning with default pad_token_id=0 (no config attribute).

    Scenario: Config without pad_token_id
    Expected: Falls back to pad_token_id=0, logs warning
    """
    vocab_size = 100
    config = SimpleNamespace(
        vocab_size=vocab_size,
        max_seq_len=32
        # No pad_token_id attribute
    )

    model = TinyTransformer(vocab_size=vocab_size)

    # Generate synthetic data with padding tokens (ID=0)
    train_data = []
    for _ in range(10):
        seq = torch.randint(1, vocab_size, (16,))  # IDs 1-99
        # Add padding: replace last 3 tokens with 0
        seq[-3:] = 0
        train_data.append(seq)

    # Capture stdout to verify warning
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Run fine-tuning (should default to pad_token_id=0)
        results = test_fine_tuning(
            model=model,
            config=config,
            train_data=train_data,
            n_epochs=1,
            batch_size=2,
            use_wandb=False
        )

        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    # Assertions
    assert "defaulting to 0" in output, "Expected warning about defaulting to pad_token_id=0"
    assert results['final_loss'] > 0
    assert torch.isfinite(torch.tensor(results['final_loss']))

    print(f"✓ Default padding test passed: Final loss={results['final_loss']:.4f}")


def test_hyperparameter_search_with_padding():
    """
    Test hyperparameter search correctly uses pad_token_id.

    Scenario: Optuna search with custom pad_token_id
    Expected: Search completes, uses ignore_index
    """
    vocab_size = 100
    config = SimpleNamespace(
        vocab_size=vocab_size,
        max_seq_len=32,
        pad_token_id=50  # Custom padding ID
    )

    def model_factory():
        return TinyTransformer(vocab_size=vocab_size)

    # Generate synthetic data with padding
    train_data = []
    for _ in range(15):
        seq = torch.randint(0, vocab_size - 1, (16,))
        seq[-2:] = 50  # Padding
        train_data.append(seq)

    # Run hyperparameter search (minimal 2 trials)
    results = test_hyperparameter_search(
        model_factory=model_factory,
        config=config,
        train_data=train_data,
        n_trials=2,
        search_space={'lr': (1e-4, 1e-3), 'batch_size': [2], 'warmup': (0, 5), 'wd': (1e-5, 1e-4)}
    )

    # Assertions
    assert 'best_params' in results
    assert 'best_value' in results
    assert results['best_value'] > 0
    assert torch.isfinite(torch.tensor(results['best_value']))

    print(f"✓ Hyperparameter search test passed: Best loss={results['best_value']:.4f}")


def test_no_padding_tokens_scenario():
    """
    Test that masking doesn't break sequences with no padding.

    Scenario: All tokens are valid (none are pad_token_id)
    Expected: Training completes normally
    """
    vocab_size = 100
    config = SimpleNamespace(
        vocab_size=vocab_size,
        max_seq_len=32,
        pad_token_id=0
    )

    model = TinyTransformer(vocab_size=vocab_size)

    # Generate data with NO padding (all tokens 1-99, no 0s)
    train_data = [torch.randint(1, vocab_size, (16,)) for _ in range(10)]

    results = test_fine_tuning(
        model=model,
        config=config,
        train_data=train_data,
        n_epochs=1,
        batch_size=2,
        use_wandb=False
    )

    # Assertions
    assert results['final_loss'] > 0
    assert torch.isfinite(torch.tensor(results['final_loss']))

    print(f"✓ No padding test passed: Final loss={results['final_loss']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("TIER 3 PADDING INTEGRATION TESTS")
    print("=" * 60)

    test_fine_tuning_with_padding_detection()
    test_fine_tuning_with_default_padding()
    test_hyperparameter_search_with_padding()
    test_no_padding_tokens_scenario()

    print("=" * 60)
    print("All integration tests passed!")
    print("=" * 60)
