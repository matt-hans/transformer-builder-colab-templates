"""Dataset validators for quality assurance.

Provides abstract base class and concrete implementations for validating
datasets before training. Validators check data quality and return structured
results with metrics and warnings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

from utils.training.validation.results import ValidationResult
from utils.training.data_quality import DataQualityFilter

logger = logging.getLogger(__name__)


class DataValidator(ABC):
    """
    Abstract base class for dataset validators.

    Subclasses must implement validate() to check dataset quality
    and return ValidationResult with pass/fail status.
    """

    @abstractmethod
    def validate(self, dataset: Any) -> ValidationResult:
        """
        Validate dataset and return structured result.

        Args:
            dataset: Dataset to validate (HuggingFace, PyTorch, or list)

        Returns:
            ValidationResult with pass/fail status, message, and metrics
        """
        pass


class SequenceLengthValidator(DataValidator):
    """
    Validates that dataset sequences meet minimum length requirements.

    Uses statistical sampling (1000 examples OR full dataset, whichever is smaller)
    to estimate filter rate without scanning entire dataset (performance optimization).

    Example:
        >>> validator = SequenceLengthValidator(min_seq_len=2, max_filter_rate=0.20)
        >>> result = validator.validate(train_dataset)
        >>> if not result.passed:
        ...     raise ValueError(result.message)
    """

    def __init__(self,
                 min_seq_len: int,
                 max_filter_rate: float = 0.05,
                 field_name: str = 'input_ids'):
        """
        Initialize sequence length validator.

        Args:
            min_seq_len: Minimum sequence length required (task-specific)
            max_filter_rate: Maximum acceptable filter rate (0.0-1.0)
            field_name: Field containing tokenized sequences
        """
        self.min_seq_len = min_seq_len
        self.max_filter_rate = max_filter_rate
        self.field_name = field_name

    def validate(self, dataset: Any) -> ValidationResult:
        """
        Sample dataset and estimate filter rate.

        Args:
            dataset: Dataset to validate

        Returns:
            ValidationResult with metrics and warnings
        """
        # Reuse existing DataQualityFilter for actual filtering logic (DRY)
        filter_fn = DataQualityFilter(self.min_seq_len, self.field_name)

        # Sample for estimation (1000 examples OR full dataset)
        dataset_size = len(dataset)
        sample_size = min(1000, dataset_size)

        # Handle different dataset types
        if hasattr(dataset, 'select'):
            # HuggingFace Dataset
            sample = dataset.select(range(sample_size))
        elif isinstance(dataset, (list, tuple)):
            # Python list/tuple
            sample = dataset[:sample_size]
        else:
            # Fallback: try iteration
            sample = list(dataset)[:sample_size]

        # Count sequences that would be filtered
        valid_count = sum(1 for ex in sample if filter_fn(ex))
        filtered_count = sample_size - valid_count
        filter_rate = filtered_count / sample_size if sample_size > 0 else 0.0

        # Compute metrics
        metrics = {
            'filter_rate': filter_rate,
            'sample_size': sample_size,
            'valid_sequences': valid_count,
            'filtered_sequences': filtered_count,
            'dataset_size': dataset_size,
        }

        # Determine pass/fail
        passed = filter_rate <= self.max_filter_rate

        # Build message
        if not passed:
            message = (
                f"Dataset has {filter_rate:.1%} sequences below {self.min_seq_len} tokens "
                f"(threshold: {self.max_filter_rate:.1%}). "
                f"Filtered {filtered_count} of {sample_size} sampled sequences."
            )
        else:
            message = f"Dataset validation passed ({filter_rate:.1%} filter rate)"

        # Add warnings for moderate filter rates (10-20% is normal for WikiText)
        warnings = []
        if 0.10 < filter_rate <= self.max_filter_rate:
            warnings.append(
                f"Moderate filter rate ({filter_rate:.1%}). "
                f"This is normal for datasets like WikiText with empty lines."
            )

        return ValidationResult(
            passed=passed,
            message=message,
            metrics=metrics,
            warnings=warnings
        )
