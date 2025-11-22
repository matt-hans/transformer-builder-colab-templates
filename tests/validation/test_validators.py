"""Tests for dataset validators."""

import pytest
from utils.training.validation.validators import SequenceLengthValidator
from utils.training.validation import ValidationResult


def test_sequence_length_validator_passes_with_clean_dataset():
    """Test validator passes when all sequences meet minimum length."""
    # Mock dataset with all sequences >= 2 tokens
    mock_dataset = [
        {'input_ids': [1, 2, 3]},
        {'input_ids': [4, 5]},
        {'input_ids': [6, 7, 8, 9]},
    ]

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.05)
    result = validator.validate(mock_dataset)

    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.0
    assert result.metrics['sample_size'] == 3
    assert result.metrics['valid_sequences'] == 3


def test_sequence_length_validator_fails_with_high_filter_rate():
    """Test validator fails when filter rate exceeds threshold."""
    # Mock dataset with 50% sequences < 2 tokens
    mock_dataset = [
        {'input_ids': [1, 2, 3]},    # Valid
        {'input_ids': []},           # Invalid
        {'input_ids': [4, 5]},       # Valid
        {'input_ids': [6]},          # Invalid
    ]

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.20)
    result = validator.validate(mock_dataset)

    assert result.passed is False
    assert result.metrics['filter_rate'] == 0.50
    assert "50.0%" in result.message


def test_sequence_length_validator_warns_on_moderate_filter_rate():
    """Test validator warns when filter rate is moderate but acceptable."""
    # Mock dataset with 15% filter rate (acceptable for WikiText)
    mock_dataset = [
        {'input_ids': [1, 2]},       # Valid
        {'input_ids': []},           # Invalid (1 of 7 = 14.3%)
    ] + [{'input_ids': [i, i+1]} for i in range(5)]  # 5 more valid

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.20)
    result = validator.validate(mock_dataset)

    assert result.passed is True
    assert len(result.warnings) > 0
    assert "WikiText" in result.warnings[0] or "moderate" in result.warnings[0].lower()
