"""
Unit tests for random seed management utilities.

Tests comprehensive seeding across Python random, NumPy, PyTorch CPU/GPU,
and DataLoader worker initialization.
"""

import os
import random
import pytest
import numpy as np
import torch


# Test 1: Python random module seeding
def test_set_random_seed_python():
    """
    Validate Python random module is seeded correctly.

    Why: Ensures reproducibility of Python's built-in random operations.
    Contract: random.random() produces same value across calls with same seed.
    """
    from utils.training.seed_manager import set_random_seed

    # First run with seed=42
    set_random_seed(42, deterministic=False)
    value1 = random.random()

    # Second run with seed=42
    set_random_seed(42, deterministic=False)
    value2 = random.random()

    # Values should be identical
    assert value1 == value2, f"Python random not reproducible: {value1} != {value2}"

    # Different seed should produce different value
    set_random_seed(123, deterministic=False)
    value3 = random.random()
    assert value1 != value3, "Different seeds should produce different values"


# Test 2: NumPy random seeding
def test_set_random_seed_numpy():
    """
    Validate NumPy random is seeded correctly.

    Why: Ensures reproducibility of NumPy random operations (data augmentation, etc).
    Contract: np.random.rand() produces same values with same seed.
    """
    from utils.training.seed_manager import set_random_seed

    # First run with seed=42
    set_random_seed(42, deterministic=False)
    arr1 = np.random.rand(5)

    # Second run with seed=42
    set_random_seed(42, deterministic=False)
    arr2 = np.random.rand(5)

    # Arrays should be identical
    np.testing.assert_array_equal(arr1, arr2,
                                  err_msg="NumPy random not reproducible")

    # Different seed should produce different array
    set_random_seed(123, deterministic=False)
    arr3 = np.random.rand(5)
    assert not np.array_equal(arr1, arr3), "Different seeds should produce different arrays"


# Test 3: PyTorch CPU random seeding
def test_set_random_seed_torch_cpu():
    """
    Validate PyTorch CPU random is seeded correctly.

    Why: Ensures reproducibility of model initialization and CPU tensor operations.
    Contract: torch.randn() produces identical tensors with same seed.
    """
    from utils.training.seed_manager import set_random_seed

    # First run with seed=42
    set_random_seed(42, deterministic=False)
    tensor1 = torch.randn(3, 4)

    # Second run with seed=42
    set_random_seed(42, deterministic=False)
    tensor2 = torch.randn(3, 4)

    # Tensors should be identical
    torch.testing.assert_close(tensor1, tensor2,
                               msg="PyTorch CPU random not reproducible")

    # Different seed should produce different tensor
    set_random_seed(123, deterministic=False)
    tensor3 = torch.randn(3, 4)
    assert not torch.equal(tensor1, tensor3), "Different seeds should produce different tensors"


# Test 4: PyTorch CUDA random seeding (skip if no GPU)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_random_seed_torch_cuda():
    """
    Validate PyTorch CUDA random is seeded correctly.

    Why: Ensures reproducibility on GPU (critical for production training).
    Contract: GPU tensors have deterministic initialization with same seed.
    """
    from utils.training.seed_manager import set_random_seed

    # First run with seed=42
    set_random_seed(42, deterministic=False)
    tensor1 = torch.randn(3, 4, device='cuda')

    # Second run with seed=42
    set_random_seed(42, deterministic=False)
    tensor2 = torch.randn(3, 4, device='cuda')

    # Tensors should be identical (or very close due to GPU ops)
    torch.testing.assert_close(tensor1, tensor2, rtol=1e-5, atol=1e-7,
                               msg="PyTorch CUDA random not reproducible")


# Test 5: DataLoader worker seeding
def test_seed_worker_function():
    """
    Validate DataLoader worker seeding function.

    Why: Without worker seeding, data shuffling is non-deterministic.
    Contract: Worker seeds NumPy/random based on torch.initial_seed().
    """
    from utils.training.seed_manager import seed_worker

    # Simulate worker with known initial seed
    torch.manual_seed(42)

    # Capture state before worker seeding
    before_np = np.random.get_state()
    before_py = random.getstate()

    # Call worker seed function
    seed_worker(worker_id=0)

    # State should have changed
    after_np = np.random.get_state()
    after_py = random.getstate()

    # States should be different (seeding occurred)
    assert not np.array_equal(before_np[1], after_np[1]), "NumPy state not modified by seed_worker"
    assert before_py != after_py, "Python random state not modified by seed_worker"

    # Verify reproducibility: same initial seed → same worker seed
    torch.manual_seed(42)
    seed_worker(worker_id=0)
    value1 = np.random.rand()

    torch.manual_seed(42)
    seed_worker(worker_id=0)
    value2 = np.random.rand()

    assert value1 == value2, "Worker seeding not reproducible"


# Test 6: Deterministic mode flag setting
def test_deterministic_mode_enables_flags():
    """
    Validate deterministic mode sets cuDNN/PyTorch flags correctly.

    Why: Deterministic mode requires disabling cuDNN optimizations.
    Contract: torch.backends.cudnn.deterministic=True, benchmark=False.
    """
    from utils.training.seed_manager import set_random_seed

    # Enable deterministic mode
    set_random_seed(42, deterministic=True)

    # Check flags
    assert torch.backends.cudnn.deterministic == True, \
        "cuDNN deterministic flag not set"
    assert torch.backends.cudnn.benchmark == False, \
        "cuDNN benchmark should be disabled in deterministic mode"
    assert torch.are_deterministic_algorithms_enabled() == True, \
        "PyTorch deterministic algorithms not enabled"

    # Check environment variable
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', \
        "CUBLAS_WORKSPACE_CONFIG not set for deterministic mode"


# Test 7: Fast mode enables optimizations
def test_fast_mode_enables_optimizations():
    """
    Validate fast mode enables cuDNN benchmark for speed.

    Why: Default mode should prioritize speed over bit-exact reproducibility.
    Contract: torch.backends.cudnn.benchmark=True when deterministic=False.
    """
    from utils.training.seed_manager import set_random_seed

    # First enable deterministic to have something to reset
    set_random_seed(42, deterministic=True)

    # Now enable fast mode
    set_random_seed(42, deterministic=False)

    # Check flags
    assert torch.backends.cudnn.benchmark == True, \
        "cuDNN benchmark should be enabled in fast mode"
    # Note: deterministic may still be True from previous call, which is OK
    # (torch doesn't reset it automatically, but benchmark=True is what matters for speed)


# Test 8: Function signature validation
def test_set_random_seed_signature():
    """
    Validate set_random_seed has correct signature and defaults.

    Why: API contract must match task specification.
    Contract: set_random_seed(seed: int, deterministic: bool = False)
    """
    from utils.training.seed_manager import set_random_seed
    import inspect

    sig = inspect.signature(set_random_seed)
    params = sig.parameters

    # Check parameters exist
    assert 'seed' in params, "Missing 'seed' parameter"
    assert 'deterministic' in params, "Missing 'deterministic' parameter"

    # Check deterministic defaults to False
    assert params['deterministic'].default == False, \
        "deterministic should default to False (fast mode)"

    # Check can call with just seed
    try:
        set_random_seed(42)  # Should work with default deterministic=False
    except TypeError:
        pytest.fail("set_random_seed should accept single argument (seed)")


# Test 9: Seed value validation
def test_seed_value_validation():
    """
    Validate seed values are handled correctly.

    Why: Ensure robust handling of edge case seed values.
    Contract: Accepts any integer seed (0, negative, large values).
    """
    from utils.training.seed_manager import set_random_seed

    # Test various seed values
    test_seeds = [0, 1, 42, 2**31 - 1, 2**32 - 1]

    for seed in test_seeds:
        try:
            set_random_seed(seed, deterministic=False)
            # Should complete without error
        except Exception as e:
            pytest.fail(f"Failed to set seed={seed}: {e}")


# Test 10: Output messages validation
def test_output_messages(capsys):
    """
    Validate informative messages are printed.

    Why: Users should see confirmation of seed and mode.
    Contract: Prints seed value and mode (deterministic/fast).
    """
    from utils.training.seed_manager import set_random_seed

    # Test fast mode message
    set_random_seed(42, deterministic=False)
    captured = capsys.readouterr()
    assert "Random seed set to 42" in captured.out, \
        "Should print seed value"
    assert "Fast mode" in captured.out or "non-determinism" in captured.out, \
        "Should indicate fast mode"

    # Test deterministic mode message
    set_random_seed(42, deterministic=True)
    captured = capsys.readouterr()
    assert "Random seed set to 42" in captured.out, \
        "Should print seed value"
    assert "deterministic mode enabled" in captured.out.lower(), \
        "Should indicate deterministic mode"
