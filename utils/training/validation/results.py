"""Validation result data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationResult:
    """
    Result of dataset validation.

    Contains pass/fail status, human-readable message, metrics for analysis,
    optional warnings for edge cases, and severity level (v4.1+).

    Severity Levels (v4.1+):
        - 'excellent': 0-10% filter rate - no warning
        - 'good': 10-20% filter rate - info only
        - 'high': 20-40% filter rate - warning (normal for WikiText)
        - 'very_high': 40-60% filter rate - strong warning
        - 'critical': 60-100% filter rate - critical warning

    Example:
        >>> result = ValidationResult(
        ...     passed=True,
        ...     message="Dataset validation passed (12.3% filter rate)",
        ...     metrics={'filter_rate': 0.123, 'sample_size': 1000},
        ...     warnings=["Moderate filter rate - normal for WikiText"],
        ...     severity='good'
        ... )
        >>> if result.severity == 'critical':
        ...     # Handle critical validation issues
        ...     pass
    """
    passed: bool
    message: str
    metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    severity: str = 'normal'  # v4.1+: 'excellent', 'good', 'high', 'very_high', 'critical'

    def __str__(self) -> str:
        """Human-readable representation with severity indicator."""
        # Severity emojis (v4.1+)
        severity_icons = {
            'excellent': '✅',
            'good': 'ℹ️',
            'high': '⚠️',
            'very_high': '🔶',
            'critical': '🚨',
            'normal': '✅',  # Fallback for backward compatibility
        }

        icon = severity_icons.get(self.severity, '✅')
        status = "PASS" if self.passed else "FAIL"
        return f"{icon} {status}: {self.message}"
