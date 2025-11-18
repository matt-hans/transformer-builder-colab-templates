import torch
import torch.nn as nn

from utils.tier3_training_utilities import _compute_gradient_norm


def test_gradient_norm_calculation_matches_clip_grad_norm():
    model = nn.Linear(10, 5)
    x = torch.randn(3, 10)
    loss = model(x).sum()
    loss.backward()

    our_norm = _compute_gradient_norm(model)
    torch_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)

    assert abs(float(our_norm) - float(torch_norm)) < 1e-5
    assert our_norm > 0.0


def test_gradient_norm_no_gradients_returns_zero():
    model = nn.Linear(10, 5)
    norm = _compute_gradient_norm(model)
    assert norm == 0.0

