"""
Random seed management for reproducible training.

Provides comprehensive seeding across all randomness sources:
- Python's built-in random module
- NumPy random number generator
- PyTorch CPU random number generator
- PyTorch CUDA random number generators (all GPUs)
- DataLoader worker processes

Supports two modes:
1. Fast mode (default): ~20% faster, minor GPU non-determinism
2. Deterministic mode: Bit-exact reproducibility, slower due to disabled optimizations
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    This function ensures reproducible results by seeding all major sources
    of randomness in the Python ML ecosystem. Use the same seed to get
    identical results across runs.

    Args:
        seed: Integer seed value (e.g., 42). Any valid Python int is accepted.
        deterministic: If True, enable fully deterministic mode (slower).
            - Fast mode (False): Enables cuDNN benchmark for ~20% speedup,
              may have minor non-determinism from GPU operations (<0.1% variation)
            - Deterministic mode (True): Bit-exact reproducibility, disables
              cuDNN optimizations, ~20% slower training

    Example:
        >>> # Fast mode (default) - good for experimentation
        >>> set_random_seed(42)
        >>> model = MyModel()  # Will have reproducible initialization
        >>>
        >>> # Deterministic mode - for publishing results
        >>> set_random_seed(42, deterministic=True)
        >>> model = MyModel()  # Bit-exact reproducibility

    Note:
        - Call this BEFORE any model initialization or data loading
        - For DataLoader reproducibility, also use seed_worker() and Generator
        - Deterministic mode sets CUBLAS_WORKSPACE_CONFIG environment variable
        - Multi-GPU setups may still have edge cases even in deterministic mode

    References:
        PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Seed Python's built-in random module
    random.seed(seed)

    # Seed NumPy random number generator
    np.random.seed(seed)

    # Seed PyTorch CPU random number generator
    torch.manual_seed(seed)

    # Seed PyTorch CUDA random number generators (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure deterministic/fast mode
    if deterministic:
        _enable_deterministic_mode()
        print(f"✅ Random seed set to {seed}")
        print("   Fully deterministic mode enabled")
    else:
        _enable_fast_mode()
        print(f"✅ Random seed set to {seed}")
        print("   Fast mode (may have minor non-determinism from cuDNN)")


def _enable_deterministic_mode() -> None:
    """
    Enable fully deterministic mode for bit-exact reproducibility.

    This disables cuDNN optimizations and enables PyTorch's deterministic
    algorithms. Training will be ~20% slower but results will be bit-exact
    reproducible across runs.

    Side effects:
        - Sets torch.backends.cudnn.deterministic = True
        - Sets torch.backends.cudnn.benchmark = False
        - Calls torch.use_deterministic_algorithms(True)
        - Sets CUBLAS_WORKSPACE_CONFIG environment variable
    """
    # Disable cuDNN benchmark mode (which selects fastest algorithms non-deterministically)
    torch.backends.cudnn.benchmark = False

    # Enable cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True

    # Enable PyTorch deterministic algorithms
    # This makes operations like scatter_add deterministic
    torch.use_deterministic_algorithms(True)

    # Set environment variable for cuBLAS workspace config
    # Required for some CUDA operations to be deterministic
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def _enable_fast_mode() -> None:
    """
    Enable fast mode with cuDNN optimizations.

    This enables cuDNN's benchmark mode which selects the fastest algorithms
    for your specific hardware. Results in ~20% speedup but may have minor
    non-determinism from GPU operations (typically <0.1% variation).

    Side effects:
        - Sets torch.backends.cudnn.benchmark = True
    """
    # Enable cuDNN benchmark mode for auto-tuning to find fastest algorithms
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers to ensure reproducible shuffling.

    This function should be passed to DataLoader's worker_init_fn parameter.
    It seeds each worker process with a unique but deterministic seed derived
    from PyTorch's initial seed and the worker ID.

    Args:
        worker_id: Worker process ID (automatically passed by DataLoader)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> # Create seeded generator
        >>> g = torch.Generator()
        >>> g.manual_seed(42)
        >>>
        >>> # Create DataLoader with reproducible shuffling
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=True,
        ...     worker_init_fn=seed_worker,  # Seed each worker
        ...     generator=g,                  # Use seeded generator
        ...     num_workers=4
        ... )

    Note:
        Without this function, DataLoader workers will have non-deterministic
        seeds, leading to different data shuffling across runs even with
        set_random_seed() called.

    References:
        PyTorch DataLoader: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    """
    # Get worker seed from PyTorch's initial seed
    # Modulo 2**32 to ensure it fits in NumPy's seed range
    worker_seed = torch.initial_seed() % 2**32

    # Seed NumPy for this worker
    np.random.seed(worker_seed)

    # Seed Python random for this worker
    random.seed(worker_seed)


def create_seeded_generator(seed: int) -> torch.Generator:
    """
    Create a seeded PyTorch Generator for use with DataLoader.

    Convenience function to create a properly seeded Generator that can be
    passed to DataLoader for reproducible data shuffling.

    Args:
        seed: Integer seed value

    Returns:
        torch.Generator: Seeded generator ready for DataLoader use

    Example:
        >>> from torch.utils.data import DataLoader
        >>> g = create_seeded_generator(42)
        >>> loader = DataLoader(
        ...     dataset,
        ...     shuffle=True,
        ...     generator=g,
        ...     worker_init_fn=seed_worker
        ... )
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# Public API
__all__ = [
    'set_random_seed',
    'seed_worker',
    'create_seeded_generator',
]
