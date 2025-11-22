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
        Sample dataset and estimate filter rate with severity-based assessment.

        v4.1+: Permissive validation - always returns passed=True, provides
        severity-based warnings instead of blocking training.

        Args:
            dataset: Dataset to validate

        Returns:
            ValidationResult with metrics, severity, and warnings (always passed=True)
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

        # Determine severity (v4.1+)
        severity = self._get_severity(filter_rate)

        # Build severity-appropriate message
        if severity == 'excellent':
            message = f"Excellent data quality ({filter_rate:.1%} filter rate)"
        elif severity == 'good':
            message = f"Good data quality ({filter_rate:.1%} filter rate)"
        elif severity == 'high':
            message = (
                f"High filter rate ({filter_rate:.1%}) - normal for structured datasets like WikiText. "
                f"Filtered {filtered_count} of {sample_size} sampled sequences."
            )
        elif severity == 'very_high':
            message = (
                f"Very high filter rate ({filter_rate:.1%}) - review recommended. "
                f"Filtered {filtered_count} of {sample_size} sampled sequences."
            )
        else:  # critical
            message = (
                f"Critical filter rate ({filter_rate:.1%}) - possible data corruption. "
                f"Filtered {filtered_count} of {sample_size} sampled sequences."
            )

        # Add context-appropriate warnings
        warnings = []
        if severity == 'good':
            warnings.append(
                "Moderate filtering is normal for some datasets (e.g., WikiText has empty lines)."
            )
        elif severity == 'high':
            warnings.append(
                "This is EXPECTED for WikiText-raw and similar datasets with structural empty lines."
            )
        elif severity == 'very_high':
            warnings.append(
                "Review your dataset source and tokenization settings if this seems incorrect."
            )
        elif severity == 'critical':
            warnings.append(
                "CRITICAL: This suggests data corruption or incorrect dataset. Verify your data source!"
            )

        return ValidationResult(
            passed=True,  # v4.1+: Always pass (permissive validation)
            message=message,
            metrics=metrics,
            warnings=warnings,
            severity=severity
        )

    def _get_severity(self, filter_rate: float) -> str:
        """
        Determine severity level based on filter rate.

        v4.1+: Uses FILTER_RATE_ZONES for multi-level warning system.

        Args:
            filter_rate: Fraction of sequences filtered (0.0-1.0)

        Returns:
            Severity level: 'excellent', 'good', 'high', 'very_high', or 'critical'
        """
        from utils.training.constants import FILTER_RATE_ZONES

        if filter_rate < FILTER_RATE_ZONES['excellent']:
            return 'excellent'
        elif filter_rate < FILTER_RATE_ZONES['good']:
            return 'good'
        elif filter_rate < FILTER_RATE_ZONES['high']:
            return 'high'
        elif filter_rate < FILTER_RATE_ZONES['very_high']:
            return 'very_high'
        else:
            return 'critical'
