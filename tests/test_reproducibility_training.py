"""
Integration-style reproducibility test: verifies that two training runs
with the same seed produce identical loss trajectories and identical
final model weights, and that different seeds produce different results.

This uses a tiny CPU-only model and synthetic dataset for speed.
"""

import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.training.seed_manager import set_random_seed, seed_worker, create_seeded_generator


class TinyDataset(Dataset):
    """Deterministic synthetic regression dataset."""
    def __init__(self, n=256, d=16, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, d, generator=g)
        true_w = torch.randn(d, 1, generator=g)
        self.y = self.X @ true_w + 0.1 * torch.randn(n, 1, generator=g)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _checksum_model(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _train_once(seed: int, steps: int = 50):
    # Ensure full determinism
    set_random_seed(seed, deterministic=True)

    # Fresh model
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    # Dataset and DataLoader with worker seeding + seeded generator
    dataset = TinyDataset(n=256, d=16, seed=seed)
    g = create_seeded_generator(seed)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=g,
    )

    losses = []
    it = iter(loader)
    for _ in range(steps):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader)
            X, y = next(it)
        opt.zero_grad(set_to_none=True)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return np.array(losses, dtype=np.float64), _checksum_model(model)


def test_same_seed_identical_losses_and_weights():
    """Two runs with same seed produce identical losses and weights."""
    losses1, ck1 = _train_once(seed=42)
    losses2, ck2 = _train_once(seed=42)

    np.testing.assert_allclose(losses1, losses2, rtol=0.0, atol=0.0,
                               err_msg="Loss curves differ for same seed")
    assert ck1 == ck2, "Model weights differ for same seed"


def test_different_seeds_produce_different_results():
    """Different seeds lead to different losses and/or weights."""
    losses1, ck1 = _train_once(seed=42)
    losses2, ck2 = _train_once(seed=123)

    # Expect at least one difference: either curve or weights
    different_curve = not np.allclose(losses1, losses2)
    different_weights = ck1 != ck2
    assert different_curve or different_weights, "Expected differences for different seeds"

