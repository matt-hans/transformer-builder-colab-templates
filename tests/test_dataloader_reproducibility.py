"""
Integration test for DataLoader reproducibility in test_fine_tuning.

Verifies that test_fine_tuning() with worker_init_fn and seeded generator
produces bit-identical results across runs with the same seed.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace

from utils.tier3_training_utilities import test_fine_tuning


class TinyTransformer(nn.Module):
    """Minimal transformer for testing."""
    def __init__(self, vocab_size=100, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        return self.linear(x)


def test_dataloader_deterministic_reproducibility():
    """
    Verify that test_fine_tuning with deterministic=True produces
    bit-identical loss trajectories across runs.
    """
    config = SimpleNamespace(
        vocab_size=100,
        max_seq_len=32,
        pad_token_id=0
    )

    # Generate synthetic data for reproducibility
    torch.manual_seed(999)
    train_data = [torch.randint(0, 100, (32,)) for _ in range(20)]

    # Run 1: Train with deterministic mode
    model1 = TinyTransformer(vocab_size=100).cpu()
    results1 = test_fine_tuning(
        model=model1,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=4,
        use_wandb=False,
        use_amp=False,
        random_seed=42,
        deterministic=True
    )

    # Run 2: Train again with same seed and deterministic mode
    model2 = TinyTransformer(vocab_size=100).cpu()
    results2 = test_fine_tuning(
        model=model2,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=4,
        use_wandb=False,
        use_amp=False,
        random_seed=42,
        deterministic=True
    )

    # Loss histories should be bit-identical
    losses1 = results1['loss_history']
    losses2 = results2['loss_history']

    assert len(losses1) == len(losses2), \
        f"Loss history lengths differ: {len(losses1)} vs {len(losses2)}"

    for i, (loss1, loss2) in enumerate(zip(losses1, losses2)):
        assert abs(loss1 - loss2) < 1e-7, \
            f"Step {i}: losses differ {loss1:.10f} vs {loss2:.10f}"

    # Final loss should be identical
    assert results1['final_loss'] == results2['final_loss'], \
        f"Final losses differ: {results1['final_loss']} vs {results2['final_loss']}"


def test_dataloader_fast_mode_still_uses_workers():
    """
    Verify that fast mode (deterministic=False) still uses worker_init_fn
    for reproducible batch ordering.
    """
    config = SimpleNamespace(
        vocab_size=100,
        max_seq_len=32,
        pad_token_id=0
    )

    torch.manual_seed(999)
    train_data = [torch.randint(0, 100, (32,)) for _ in range(20)]

    # Run 1: Fast mode
    model1 = TinyTransformer(vocab_size=100).cpu()
    results1 = test_fine_tuning(
        model=model1,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=4,
        use_wandb=False,
        use_amp=False,
        random_seed=42,
        deterministic=False  # Fast mode
    )

    # Run 2: Fast mode again
    model2 = TinyTransformer(vocab_size=100).cpu()
    results2 = test_fine_tuning(
        model=model2,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=4,
        use_wandb=False,
        use_amp=False,
        random_seed=42,
        deterministic=False  # Fast mode
    )

    # Losses should be very close (not necessarily bit-identical due to cuDNN)
    # but worker seeding ensures batch order is identical
    losses1 = results1['loss_history']
    losses2 = results2['loss_history']

    # Allow small tolerance for fast mode (cuDNN non-determinism)
    for i, (loss1, loss2) in enumerate(zip(losses1, losses2)):
        # Losses should be within 1% even in fast mode
        rel_diff = abs(loss1 - loss2) / (abs(loss1) + 1e-8)
        assert rel_diff < 0.01, \
            f"Step {i}: losses too different {loss1:.6f} vs {loss2:.6f} (rel diff: {rel_diff:.4f})"


def test_dataloader_different_seeds_produce_different_results():
    """
    Verify that different seeds lead to different training trajectories.
    """
    config = SimpleNamespace(
        vocab_size=100,
        max_seq_len=32,
        pad_token_id=0
    )

    torch.manual_seed(999)
    train_data = [torch.randint(0, 100, (32,)) for _ in range(20)]

    # Run 1: seed=42
    model1 = TinyTransformer(vocab_size=100).cpu()
    results1 = test_fine_tuning(
        model=model1,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=4,
        use_wandb=False,
        use_amp=False,
        random_seed=42,
        deterministic=True
    )

    # Run 2: seed=123
    model2 = TinyTransformer(vocab_size=100).cpu()
    results2 = test_fine_tuning(
        model=model2,
        config=config,
        train_data=train_data,
        n_epochs=2,
        batch_size=4,
        use_wandb=False,
        use_amp=False,
        random_seed=123,
        deterministic=True
    )

    # Loss histories should differ
    losses1 = results1['loss_history']
    losses2 = results2['loss_history']

    # Check that at least 50% of losses are different
    different_count = sum(1 for l1, l2 in zip(losses1, losses2) if abs(l1 - l2) > 1e-6)
    assert different_count > len(losses1) * 0.5, \
        f"Expected different trajectories with different seeds, but {different_count}/{len(losses1)} steps differ"
