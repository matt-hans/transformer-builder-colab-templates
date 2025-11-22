"""Integration tests for validation workflow with realistic datasets."""

import pytest
from utils.training.validation import SequenceLengthValidator
from utils.training.data_quality import filter_short_sequences
from utils.training.constants import TASK_MIN_SEQ_LEN


# ============================================================================
# Realistic Dataset Scenarios
# ============================================================================

def test_wikitext_scenario():
    """Test validation with WikiText-like dataset (15-25% empty lines)."""
    # Simulate WikiText-2 with paragraph separators (empty lines)
    # Typical structure: text, text, text, empty, text, text, empty, ...
    wikitext_dataset = []

    # Add 75 valid sequences
    for i in range(75):
        wikitext_dataset.append({
            'input_ids': list(range(i % 50 + 5))  # 5-54 tokens
        })

    # Add 25 empty sequences (paragraph separators)
    for _ in range(25):
        wikitext_dataset.append({
            'input_ids': []  # Empty line
        })

    # Total: 100 sequences, 25% empty

    # Validation should PASS with permissive threshold (20%)
    # This demonstrates why we need dataset-level thresholds
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.20  # Permissive for WikiText
    )

    # This should FAIL because 25% > 20%
    result = validator.validate(wikitext_dataset)
    assert result.passed is False
    assert result.metrics['filter_rate'] == 0.25

    # But with 30% threshold, it passes
    validator_permissive = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.30
    )
    result_permissive = validator_permissive.validate(wikitext_dataset)
    assert result_permissive.passed is True


def test_c4_scenario():
    """Test validation with C4-like dataset (clean, <5% filter rate)."""
    # Simulate C4 (Colossal Clean Crawled Corpus) - high quality
    c4_dataset = []

    # Add 97 valid sequences (web documents)
    for i in range(97):
        c4_dataset.append({
            'input_ids': list(range(20 + (i % 100)))  # 20-119 tokens
        })

    # Add 3 short sequences (rare but possible)
    for _ in range(3):
        c4_dataset.append({
            'input_ids': [1]  # Too short
        })

    # Total: 100 sequences, 3% filter rate

    # Validation should PASS with strict threshold
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.05  # Strict for clean datasets
    )

    result = validator.validate(c4_dataset)
    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.03


def test_corrupted_dataset_scenario():
    """Test validation with severely corrupted dataset (>50% invalid)."""
    # Simulate dataset with severe quality issues
    corrupted_dataset = []

    # 60% invalid sequences
    for _ in range(60):
        corrupted_dataset.append({'input_ids': []})

    # 40% valid sequences
    for i in range(40):
        corrupted_dataset.append({'input_ids': list(range(10))})

    # Validation should FAIL even with permissive threshold
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.20
    )

    result = validator.validate(corrupted_dataset)
    assert result.passed is False
    assert result.metrics['filter_rate'] == 0.60
    assert "60" in result.message


def test_perfect_dataset_scenario():
    """Test validation with perfect dataset (0% filter rate)."""
    # Simulate ideal preprocessed dataset
    perfect_dataset = [
        {'input_ids': list(range(10 + i))}
        for i in range(100)
    ]

    # All sequences >= 10 tokens
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.05
    )

    result = validator.validate(perfect_dataset)
    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.0
    assert len(result.warnings) == 0


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

def test_end_to_end_validation_filter_workflow():
    """Test complete workflow: validate → filter → verify."""
    # Step 1: Create dataset with mixed quality
    mixed_dataset = [
        {'input_ids': [1, 2, 3]},      # Valid
        {'input_ids': []},             # Invalid
        {'input_ids': [4, 5, 6, 7]},   # Valid
        {'input_ids': [8]},            # Invalid
        {'input_ids': [9, 10]},        # Valid
    ]

    # Step 2: Validate
    validator = SequenceLengthValidator(
        min_seq_len=2,
        max_filter_rate=0.50  # Permissive for demo
    )
    result = validator.validate(mixed_dataset)

    assert result.passed is True  # 40% filter rate < 50%
    assert result.metrics['filter_rate'] == 0.40  # 2 of 5 invalid

    # Step 3: Filter dataset
    filtered_dataset = filter_short_sequences(
        mixed_dataset,
        min_length=2,
        verbose=False
    )

    # Step 4: Verify filtered dataset
    assert len(filtered_dataset) == 3

    # Step 5: Re-validate filtered dataset (should be perfect)
    result_after_filter = validator.validate(filtered_dataset)
    assert result_after_filter.passed is True
    assert result_after_filter.metrics['filter_rate'] == 0.0


def test_validation_prevents_training_failure():
    """Test that validation catches issues before training starts."""
    # Scenario: User tokenizes dataset but forgets to filter
    # Dataset has ALL empty sequences (would crash training)
    bad_dataset = [
        {'input_ids': []},
        {'input_ids': []},
        {'input_ids': []},
    ]

    # Validation catches this BEFORE training
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.05
    )

    result = validator.validate(bad_dataset)

    # Should FAIL with clear message
    assert result.passed is False
    assert result.metrics['filter_rate'] == 1.0

    # User sees this error IMMEDIATELY (fail-fast)
    # instead of after GPU allocation + first training batch


# ============================================================================
# Task-Specific Integration Tests
# ============================================================================

def test_classification_task_integration():
    """Test validation for classification task (min_seq_len=1)."""
    # Classification datasets can have single-token sequences
    classification_dataset = [
        {'input_ids': [1]},        # Valid for classification
        {'input_ids': [2, 3]},     # Valid
        {'input_ids': []},         # Invalid (empty)
        {'input_ids': [4, 5, 6]},  # Valid
    ]

    # For classification, min_seq_len=1
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['classification'],
        max_filter_rate=0.30
    )

    result = validator.validate(classification_dataset)

    # Only 1 of 4 invalid (25%)
    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.25


def test_causal_lm_task_integration():
    """Test validation for causal LM task (min_seq_len=2)."""
    # Causal LM requires at least 2 tokens for shifting
    lm_dataset = [
        {'input_ids': [1, 2]},     # Valid (minimum for LM)
        {'input_ids': [3]},        # Invalid (too short)
        {'input_ids': [4, 5, 6]},  # Valid
        {'input_ids': []},         # Invalid (empty)
    ]

    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.60
    )

    result = validator.validate(lm_dataset)

    # 2 of 4 invalid (50%)
    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.50


# ============================================================================
# Large Dataset Integration Tests
# ============================================================================

def test_large_dataset_sampling_integration():
    """Test that validator efficiently handles large datasets via sampling."""
    # Simulate 10,000 sequence dataset
    # Validator should only sample first 1000
    large_dataset = []

    # First 800: valid
    for i in range(800):
        large_dataset.append({'input_ids': list(range(10))})

    # Next 200: invalid (in sample window)
    for _ in range(200):
        large_dataset.append({'input_ids': []})

    # Next 9000: all valid (NOT sampled)
    for i in range(9000):
        large_dataset.append({'input_ids': list(range(10))})

    # Total: 10,000 sequences
    # Actual filter rate: 200 / 10,000 = 2%
    # Sampled filter rate: 200 / 1000 = 20%

    validator = SequenceLengthValidator(
        min_seq_len=2,
        max_filter_rate=0.25
    )

    result = validator.validate(large_dataset)

    # Should sample first 1000 and estimate 20% filter rate
    assert result.metrics['sample_size'] == 1000
    assert result.metrics['dataset_size'] == 10000
    assert result.metrics['filter_rate'] == 0.20
    assert result.passed is True


def test_realistic_preprocessing_workflow():
    """Test complete preprocessing workflow as user would experience."""
    # Step 1: User tokenizes raw text
    raw_dataset = [
        {'input_ids': list(range(50))},   # Normal document
        {'input_ids': []},                # Empty line (paragraph break)
        {'input_ids': list(range(100))},  # Long document
        {'input_ids': [1]},               # Single token (rare but valid for some tasks)
        {'input_ids': list(range(25))},   # Normal document
        {'input_ids': []},                # Empty line
    ]

    # Step 2: Validate BEFORE filtering (user sees quality report)
    validator = SequenceLengthValidator(
        min_seq_len=TASK_MIN_SEQ_LEN['lm'],
        max_filter_rate=0.40
    )

    pre_filter_result = validator.validate(raw_dataset)

    # 3 of 6 sequences are < 2 tokens (50%)
    assert pre_filter_result.passed is False  # Exceeds 40% threshold
    assert pre_filter_result.metrics['filter_rate'] == 0.50

    # Step 3: User sees error and applies filter
    filtered_dataset = filter_short_sequences(
        raw_dataset,
        min_length=TASK_MIN_SEQ_LEN['lm'],
        verbose=False
    )

    # Step 4: Verify filter worked
    assert len(filtered_dataset) == 3  # Only valid sequences remain

    # Step 5: Re-validate (should pass now)
    post_filter_result = validator.validate(filtered_dataset)
    assert post_filter_result.passed is True
    assert post_filter_result.metrics['filter_rate'] == 0.0
