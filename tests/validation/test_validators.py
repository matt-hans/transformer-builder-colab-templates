"""Comprehensive tests for dataset validators (>90% coverage)."""

import pytest
from utils.training.validation.validators import (
    DataValidator,
    SequenceLengthValidator,
)
from utils.training.validation import (
    ValidationResult,
    ValidationError,
    SequenceLengthError,
    EmptyDatasetError,
)


# ============================================================================
# SequenceLengthValidator Tests
# ============================================================================

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
    """Test validator reports severity for high filter rate (v4.1+: permissive)."""
    # Mock dataset with 50% sequences < 2 tokens
    mock_dataset = [
        {'input_ids': [1, 2, 3]},    # Valid
        {'input_ids': []},           # Invalid
        {'input_ids': [4, 5]},       # Valid
        {'input_ids': [6]},          # Invalid
    ]

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.20)
    result = validator.validate(mock_dataset)

    # v4.1+: Always passes, provides severity instead
    assert result.passed is True
    assert result.severity == 'very_high'  # 50% is in very_high zone (40-60%)
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


def test_sequence_length_validator_with_empty_dataset():
    """Test validator handles empty dataset gracefully."""
    mock_dataset = []

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.20)
    result = validator.validate(mock_dataset)

    # Empty dataset has 0 filter rate (no sequences to filter)
    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.0
    assert result.metrics['sample_size'] == 0


def test_sequence_length_validator_with_single_sequence():
    """Test validator with single sequence dataset."""
    # Valid sequence
    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.05)
    result = validator.validate([{'input_ids': [1, 2, 3]}])

    assert result.passed is True
    assert result.severity == 'excellent'
    assert result.metrics['filter_rate'] == 0.0

    # Invalid sequence (v4.1+: passes but with critical severity)
    result = validator.validate([{'input_ids': [1]}])

    assert result.passed is True  # v4.1+: Always passes
    assert result.severity == 'critical'  # 100% filter rate
    assert result.metrics['filter_rate'] == 1.0


def test_sequence_length_validator_samples_large_dataset():
    """Test validator samples max 1000 examples from large dataset."""
    # Create 2000 sequences, first 900 valid, next 100 invalid, rest valid
    large_dataset = (
        [{'input_ids': [1, 2, 3]} for _ in range(900)] +
        [{'input_ids': [1]} for _ in range(100)] +
        [{'input_ids': [1, 2, 3]} for _ in range(1000)]
    )

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.20)
    result = validator.validate(large_dataset)

    # Should sample first 1000 (900 valid + 100 invalid = 10% filter rate)
    assert result.metrics['sample_size'] == 1000
    assert result.metrics['dataset_size'] == 2000
    assert result.metrics['filter_rate'] == 0.10
    assert result.passed is True


def test_sequence_length_validator_with_custom_field_name():
    """Test validator with custom field name (not 'input_ids')."""
    mock_dataset = [
        {'tokens': [1, 2, 3]},
        {'tokens': [4, 5]},
        {'tokens': []},  # Invalid
    ]

    validator = SequenceLengthValidator(
        min_seq_len=2,
        max_filter_rate=0.40,
        field_name='tokens'
    )
    result = validator.validate(mock_dataset)

    assert result.passed is True
    assert result.metrics['filter_rate'] == pytest.approx(0.333, abs=0.01)


def test_sequence_length_validator_with_missing_field():
    """Test validator handles missing field gracefully."""
    mock_dataset = [
        {'text': 'hello world'},  # No 'input_ids' field
        {'text': 'foo bar'},
    ]

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.05)
    result = validator.validate(mock_dataset)

    # Missing field treated as empty sequence (length 0)
    # v4.1+: Always passes, critical severity for 100% filter rate
    assert result.passed is True
    assert result.severity == 'critical'
    assert result.metrics['filter_rate'] == 1.0


def test_sequence_length_validator_with_tuple_dataset():
    """Test validator works with tuple (immutable list)."""
    mock_dataset = tuple([
        {'input_ids': [1, 2, 3]},
        {'input_ids': [4, 5]},
    ])

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.05)
    result = validator.validate(mock_dataset)

    assert result.passed is True
    assert result.metrics['filter_rate'] == 0.0


def test_sequence_length_validator_with_huggingface_dataset():
    """Test validator works with HuggingFace Dataset (has .select())."""
    # Mock HuggingFace Dataset
    class MockHFDataset:
        def __init__(self, data):
            self._data = data

        def __len__(self):
            return len(self._data)

        def __iter__(self):
            return iter(self._data)

        def select(self, indices):
            """Mimic HuggingFace Dataset.select()"""
            selected_data = [self._data[i] for i in indices]
            return MockHFDataset(selected_data)

    mock_dataset = MockHFDataset([
        {'input_ids': [1, 2, 3]},
        {'input_ids': [4, 5]},
        {'input_ids': []},  # Invalid
    ])

    validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.40)
    result = validator.validate(mock_dataset)

    assert result.passed is True
    assert result.metrics['filter_rate'] == pytest.approx(0.333, abs=0.01)


def test_sequence_length_validator_different_thresholds():
    """Test validator with different threshold values (v4.1+: max_filter_rate deprecated)."""
    mock_dataset = [
        {'input_ids': [1, 2]},    # Valid
        {'input_ids': []},        # Invalid (25% filter rate)
        {'input_ids': [3, 4]},    # Valid
        {'input_ids': [5, 6]},    # Valid
    ]

    # v4.1+: max_filter_rate parameter is deprecated in favor of severity zones
    # Both validators should return passed=True with severity='high' (25% is in high zone)
    validator_strict = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.05)
    result_strict = validator_strict.validate(mock_dataset)
    assert result_strict.passed is True  # v4.1+: Always passes
    assert result_strict.severity == 'high'  # 25% is in high zone (20-40%)

    # Permissive threshold (30%)
    validator_permissive = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.30)
    result_permissive = validator_permissive.validate(mock_dataset)
    assert result_permissive.passed is True
    assert result_permissive.severity == 'high'  # Same severity zone


def test_sequence_length_validator_different_min_lengths():
    """Test validator with different minimum sequence lengths."""
    mock_dataset = [
        {'input_ids': []},           # 0 tokens
        {'input_ids': [1]},          # 1 token
        {'input_ids': [1, 2]},       # 2 tokens
        {'input_ids': [1, 2, 3]},    # 3 tokens
    ]

    # min_seq_len=1 (classification tasks)
    validator_cls = SequenceLengthValidator(min_seq_len=1, max_filter_rate=0.30)
    result_cls = validator_cls.validate(mock_dataset)
    assert result_cls.metrics['filter_rate'] == 0.25  # 1 of 4 filtered

    # min_seq_len=2 (causal LM)
    validator_lm = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.60)
    result_lm = validator_lm.validate(mock_dataset)
    assert result_lm.metrics['filter_rate'] == 0.50  # 2 of 4 filtered


# ============================================================================
# ValidationResult Tests
# ============================================================================

def test_validation_result_passed():
    """Test ValidationResult for passing validation."""
    result = ValidationResult(
        passed=True,
        message="Dataset validation passed (5.2% filter rate)",
        metrics={'filter_rate': 0.052, 'sample_size': 1000},
        warnings=[]
    )

    assert result.passed is True
    assert "5.2%" in result.message
    assert len(result.warnings) == 0
    assert "✅ PASS" in str(result)


def test_validation_result_failed():
    """Test ValidationResult for failing validation (v4.1+: severity-based icons)."""
    result = ValidationResult(
        passed=False,
        message="Dataset has 30% sequences below 2 tokens",
        metrics={'filter_rate': 0.30, 'sample_size': 1000},
        warnings=[]
    )

    assert result.passed is False
    assert "30%" in result.message
    # v4.1+: Icon based on severity (defaults to 'normal' = ✅), not passed status
    assert "✅ FAIL" in str(result)


def test_validation_result_with_warnings():
    """Test ValidationResult with warnings."""
    result = ValidationResult(
        passed=True,
        message="Dataset validation passed",
        metrics={'filter_rate': 0.15},
        warnings=["Moderate filter rate - normal for WikiText"]
    )

    assert result.passed is True
    assert len(result.warnings) == 1
    assert "WikiText" in result.warnings[0]


# ============================================================================
# DataValidator Base Class Tests
# ============================================================================

def test_data_validator_is_abstract():
    """Test that DataValidator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        # DataValidator is abstract and requires validate() implementation
        DataValidator()


def test_data_validator_subclass_implementation():
    """Test that subclasses must implement validate()."""
    class IncompleteValidator(DataValidator):
        pass

    with pytest.raises(TypeError):
        # Missing validate() implementation
        IncompleteValidator()


def test_data_validator_valid_subclass():
    """Test that proper subclass implementation works."""
    class CustomValidator(DataValidator):
        def validate(self, dataset):
            return ValidationResult(
                passed=True,
                message="Custom validation passed",
                metrics={},
                warnings=[]
            )

    validator = CustomValidator()
    result = validator.validate([])

    assert result.passed is True
    assert "Custom" in result.message
