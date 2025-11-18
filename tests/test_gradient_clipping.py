import torch
import torch.nn as nn

from utils.tier3_training_utilities import _compute_gradient_norm


def _assign_grads(model: nn.Module, grads):
    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone().detach().requires_grad_(False)


def test_large_gradients_are_clipped_to_max_norm():
    # Single linear layer with 2 weights; set grad to [3, 4] so ||g||=5
    model = nn.Linear(2, 1, bias=False)
    model.zero_grad(set_to_none=True)
    grads = [torch.tensor([[3.0, 4.0]])]
    _assign_grads(model, grads)

    pre = _compute_gradient_norm(model)
    assert abs(pre - 5.0) < 1e-6

    post = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    assert abs(float(post) - 1.0) < 1e-6
    # Norm after clipping recomputed should also be ~1
    post_recompute = _compute_gradient_norm(model)
    assert abs(post_recompute - 1.0) < 1e-6


def test_small_gradients_remain_unchanged_when_below_threshold():
    model = nn.Linear(2, 1, bias=False)
    model.zero_grad(set_to_none=True)
    grads = [torch.tensor([[0.3, 0.4]])]  # norm = 0.5
    _assign_grads(model, grads)

    pre = _compute_gradient_norm(model)
    assert abs(pre - 0.5) < 1e-6

    post = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    assert abs(float(post) - 0.5) < 1e-6
    post_recompute = _compute_gradient_norm(model)
    assert abs(post_recompute - 0.5) < 1e-6

