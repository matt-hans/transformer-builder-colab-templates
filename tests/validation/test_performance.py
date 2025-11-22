"""Performance benchmarks for validation architecture.

Demonstrates the performance benefits of preprocessing (1× filter) vs
runtime (100-400× filter per epoch) validation.
"""

import time
import pytest
from utils.training.validation import SequenceLengthValidator
from utils.training.data_quality import filter_short_sequences
from utils.training.constants import TASK_MIN_SEQ_LEN


# Skip benchmarks in CI unless explicitly requested
pytestmark = pytest.mark.benchmark


def create_mixed_quality_dataset(size: int, empty_ratio: float = 0.20):
    """Create dataset with specified empty sequence ratio.

    Args:
        size: Number of sequences
        empty_ratio: Fraction of sequences that are empty (0.0-1.0)

    Returns:
        List of dicts with 'input_ids' field
    """
    dataset = []
    num_empty = int(size * empty_ratio)

    # Add empty sequences
    for _ in range(num_empty):
        dataset.append({'input_ids': []})

    # Add valid sequences
    for i in range(size - num_empty):
        dataset.append({'input_ids': list(range(10 + (i % 50)))})

    return dataset


@pytest.mark.benchmark
def test_benchmark_preprocessing_vs_runtime():
    """Benchmark: Preprocessing (1× filter) vs Runtime (100× filter)."""
    # Create dataset with 20% empty sequences (WikiText-like)
    dataset_size = 10000
    dataset = create_mixed_quality_dataset(dataset_size, empty_ratio=0.20)

    # ========================================================================
    # PREPROCESSING APPROACH (New v4.0 Architecture)
    # ========================================================================
    # Filter once before training

    start_preprocess = time.perf_counter()
    filtered_dataset = filter_short_sequences(
        dataset.copy(),
        min_length=TASK_MIN_SEQ_LEN['lm'],
        verbose=False
    )
    preprocess_time = time.perf_counter() - start_preprocess

    # ========================================================================
    # RUNTIME APPROACH (Legacy v3.x Architecture)
    # ========================================================================
    # Simulate filtering on every batch during training
    # Typical training: 100 epochs, 125 batches/epoch (10,000 / batch_size=80)

    num_epochs = 100
    num_batches_per_epoch = 125
    batch_size = 80

    start_runtime = time.perf_counter()

    for epoch in range(num_epochs):
        for batch_idx in range(num_batches_per_epoch):
            # Simulate collator filtering batch (legacy approach)
            # Each batch filters 80 sequences
            batch_start_idx = (batch_idx * batch_size) % dataset_size
            batch_end_idx = min(batch_start_idx + batch_size, dataset_size)
            batch = dataset[batch_start_idx:batch_end_idx]

            # Filter batch (this happens 12,500 times!)
            filtered_batch = [
                ex for ex in batch
                if len(ex.get('input_ids', [])) >= TASK_MIN_SEQ_LEN['lm']
            ]

    runtime_time = time.perf_counter() - start_runtime

    # ========================================================================
    # RESULTS
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"PERFORMANCE BENCHMARK: Preprocessing vs Runtime Validation")
    print(f"{'='*70}")
    print(f"Dataset size: {dataset_size:,} sequences ({len(dataset) - len(filtered_dataset):,} empty)")
    print(f"Training: {num_epochs} epochs × {num_batches_per_epoch} batches = {num_epochs * num_batches_per_epoch:,} iterations")
    print(f"\nPreprocessing (1× filter):  {preprocess_time:.4f}s")
    print(f"Runtime (12,500× filter):   {runtime_time:.4f}s")
    print(f"\nSpeedup: {runtime_time / preprocess_time:.1f}x faster")
    print(f"Time saved: {runtime_time - preprocess_time:.2f}s")
    print(f"{'='*70}\n")

    # Preprocessing should be significantly faster
    assert runtime_time > preprocess_time * 10  # At least 10x faster


@pytest.mark.benchmark
def test_benchmark_large_dataset_sampling():
    """Benchmark: Validation with sampling (1000) vs full scan."""
    # Create large dataset
    large_dataset = create_mixed_quality_dataset(100000, empty_ratio=0.15)

    # ========================================================================
    # SAMPLING APPROACH (v4.0)
    # ========================================================================
    # Samples first 1000 sequences

    validator_sampling = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.20
    )

    start_sampling = time.perf_counter()
    result_sampling = validator_sampling.validate(large_dataset)
    sampling_time = time.perf_counter() - start_sampling

    # ========================================================================
    # FULL SCAN APPROACH (Naive)
    # ========================================================================
    # Scans all 100,000 sequences

    start_full = time.perf_counter()

    filter_count = sum(
        1 for ex in large_dataset
        if len(ex.get('input_ids', [])) < TASK_MIN_SEQ_LEN['lm']
    )
    filter_rate_full = filter_count / len(large_dataset)

    full_scan_time = time.perf_counter() - start_full

    # ========================================================================
    # RESULTS
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"SAMPLING BENCHMARK: 1,000 Sample vs Full Scan")
    print(f"{'='*70}")
    print(f"Dataset size: {len(large_dataset):,} sequences")
    print(f"\nSampling (1,000 sequences): {sampling_time:.4f}s")
    print(f"Full scan (100,000 sequences): {full_scan_time:.4f}s")
    print(f"\nSpeedup: {full_scan_time / sampling_time:.1f}x faster")
    print(f"Estimated filter rate (sampling): {result_sampling.metrics['filter_rate']:.1%}")
    print(f"Actual filter rate (full scan): {filter_rate_full:.1%}")
    print(f"Estimation error: {abs(result_sampling.metrics['filter_rate'] - filter_rate_full):.1%}")
    print(f"{'='*70}\n")

    # Sampling should be significantly faster
    assert full_scan_time > sampling_time * 5  # At least 5x faster

    # Estimation should be accurate (within 5%)
    assert abs(result_sampling.metrics['filter_rate'] - filter_rate_full) < 0.05


@pytest.mark.benchmark
def test_benchmark_dataset_vs_batch_validation():
    """Benchmark: Dataset-level (1× validation) vs Batch-level (125× validation)."""
    # Create dataset
    dataset = create_mixed_quality_dataset(10000, empty_ratio=0.20)

    # ========================================================================
    # DATASET-LEVEL VALIDATION (v4.0)
    # ========================================================================
    # Validate once before training

    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.30
    )

    start_dataset = time.perf_counter()
    result_dataset = validator.validate(dataset)
    dataset_validation_time = time.perf_counter() - start_dataset

    # ========================================================================
    # BATCH-LEVEL VALIDATION (Legacy)
    # ========================================================================
    # Validate every batch during one epoch

    num_batches = 125
    batch_size = 80

    start_batch = time.perf_counter()

    for batch_idx in range(num_batches):
        batch_start_idx = (batch_idx * batch_size) % len(dataset)
        batch_end_idx = min(batch_start_idx + batch_size, len(dataset))
        batch = dataset[batch_start_idx:batch_end_idx]

        # Count invalid sequences in batch
        invalid_count = sum(
            1 for ex in batch
            if len(ex.get('input_ids', [])) < TASK_MIN_SEQ_LEN['lm']
        )
        batch_filter_rate = invalid_count / len(batch)

        # Check threshold (would log warning in legacy approach)
        if batch_filter_rate > 0.10:
            pass  # Legacy: log warning

    batch_validation_time = time.perf_counter() - start_batch

    # ========================================================================
    # RESULTS
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"VALIDATION LEVEL BENCHMARK: Dataset vs Batch")
    print(f"{'='*70}")
    print(f"Dataset size: {len(dataset):,} sequences")
    print(f"Batches per epoch: {num_batches}")
    print(f"\nDataset-level (1× validation): {dataset_validation_time:.4f}s")
    print(f"Batch-level (125× validation): {batch_validation_time:.4f}s")
    print(f"\nSpeedup: {batch_validation_time / dataset_validation_time:.1f}x faster")
    print(f"{'='*70}\n")

    # Dataset-level should be faster
    assert batch_validation_time > dataset_validation_time


@pytest.mark.benchmark
def test_benchmark_statistical_validity():
    """Benchmark: Statistical validity of batch-level thresholds."""
    # Demonstrate why batch-level thresholds are statistically invalid

    # Create dataset with exactly 10% empty sequences
    dataset = create_mixed_quality_dataset(1000, empty_ratio=0.10)

    # Simulate 100 random batches of size 4
    import random

    batch_size = 4
    num_batches = 100
    batch_filter_rates = []
    false_positives = 0  # Count batches that exceed 10% threshold

    for _ in range(num_batches):
        # Random sample (without replacement)
        batch = random.sample(dataset, batch_size)

        # Calculate filter rate for this batch
        invalid_count = sum(
            1 for ex in batch
            if len(ex.get('input_ids', [])) < TASK_MIN_SEQ_LEN['lm']
        )
        batch_filter_rate = invalid_count / batch_size
        batch_filter_rates.append(batch_filter_rate)

        # Legacy approach: flag if >10%
        if batch_filter_rate > 0.10:
            false_positives += 1

    # Calculate variance
    import statistics
    mean_filter_rate = statistics.mean(batch_filter_rates)
    stdev_filter_rate = statistics.stdev(batch_filter_rates)

    print(f"\n{'='*70}")
    print(f"STATISTICAL VALIDITY: Batch-Level Thresholds (batch_size=4)")
    print(f"{'='*70}")
    print(f"Dataset filter rate: 10.0% (ground truth)")
    print(f"Batch filter rates:")
    print(f"  Mean: {mean_filter_rate:.1%}")
    print(f"  Std Dev: {stdev_filter_rate:.1%}")
    print(f"  Min: {min(batch_filter_rates):.1%}")
    print(f"  Max: {max(batch_filter_rates):.1%}")
    print(f"\nFalse positives (batches flagged as >10%): {false_positives}/{num_batches} ({false_positives/num_batches:.1%})")
    print(f"\nConclusion: {false_positives/num_batches:.1%} of batches trigger false alarms")
    print(f"            This is why batch-level validation is unreliable!")
    print(f"{'='*70}\n")

    # With batch_size=4, expect high variance (~40-50% false positives)
    # This demonstrates why we moved to dataset-level validation
    assert false_positives > 20  # Expect >20% false positive rate


if __name__ == '__main__':
    # Run benchmarks directly
    print("Running performance benchmarks...")
    print("(This may take 30-60 seconds)\n")

    test_benchmark_preprocessing_vs_runtime()
    test_benchmark_large_dataset_sampling()
    test_benchmark_dataset_vs_batch_validation()
    test_benchmark_statistical_validity()

    print("\n✅ All benchmarks complete!")
