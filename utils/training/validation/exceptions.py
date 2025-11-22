"""Custom exceptions for validation errors."""


class ValidationError(ValueError):
    """
    Base exception for dataset validation failures.

    Raised when dataset quality does not meet requirements for training.
    All validation errors should inherit from this class.
    """
    pass


class SequenceLengthError(ValidationError):
    """
    Raised when too many sequences are below the minimum length threshold.

    Example:
        Dataset has 30% sequences < 2 tokens, but max allowed is 20%.
    """
    pass


class EmptyDatasetError(ValidationError):
    """
    Raised when dataset is completely empty or 100% filtered.

    This is a critical error indicating severe data quality issues.
    """
    pass
