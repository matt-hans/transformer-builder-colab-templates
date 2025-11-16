"""
Metrics utility functions (perplexity and related helpers).

These functions avoid heavy dependencies and can be unit-tested without torch.
"""

import math
from typing import Union


def calculate_perplexity(loss: Union[float, int]) -> float:
    """
    Calculate perplexity from cross-entropy loss with numerical stability.

    Perplexity = exp(loss). Loss is clipped to a max of 20 to prevent overflow,
    which corresponds to exp(20) ≈ 4.85e8.
    """
    try:
        x = float(loss)
    except Exception:
        x = 0.0
    x = min(x, 20.0)
    return math.exp(x)

