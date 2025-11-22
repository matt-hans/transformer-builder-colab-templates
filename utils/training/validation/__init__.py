"""Dataset validation infrastructure.

Provides validators for checking data quality before training, ensuring
datasets meet task-specific requirements (minimum sequence lengths, etc.).

Public API:
    - ValidationResult: Result data structure
    - DataValidator: Abstract base class for validators
    - SequenceLengthValidator: Validates sequence length requirements
    - ValidationError: Base exception for validation failures
"""

from utils.training.validation.results import ValidationResult
from utils.training.validation.exceptions import (
    ValidationError,
    SequenceLengthError,
    EmptyDatasetError,
)
from utils.training.validation.validators import (
    DataValidator,
    SequenceLengthValidator,
)

__all__ = [
    'ValidationResult',
    'ValidationError',
    'SequenceLengthError',
    'EmptyDatasetError',
    'DataValidator',
    'SequenceLengthValidator',
]
