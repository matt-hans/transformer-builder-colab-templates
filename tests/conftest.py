"""
Shared test fixtures and utilities for gradient accumulation tests.

This module provides common test fixtures to reduce code duplication
and maintain consistency across test files.
"""

import pytest
import torch


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
