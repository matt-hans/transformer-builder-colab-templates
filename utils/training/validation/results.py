"""Validation result data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationResult:
    """
    Result of dataset validation.

    Contains pass/fail status, human-readable message, metrics for analysis,
    and optional warnings for edge cases (e.g., moderate filter rates).

    Example:
        >>> result = ValidationResult(
        ...     passed=True,
        ...     message="Dataset validation passed (12.3% filter rate)",
        ...     metrics={'filter_rate': 0.123, 'sample_size': 1000},
        ...     warnings=["Moderate filter rate - normal for WikiText"]
        ... )
        >>> if not result.passed:
        ...     raise ValueError(result.message)
    """
    passed: bool
    message: str
    metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.message}"
