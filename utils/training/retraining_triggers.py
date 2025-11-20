"""
Automated retraining trigger system for ML model lifecycle management.

This module provides a comprehensive trigger system for detecting when model
retraining is needed based on multiple criteria:
- Data drift (distribution changes)
- Performance degradation (accuracy drop, loss increase)
- Time elapsed since last training
- Data volume thresholds
- Composite triggers (AND/OR logic)

The trigger system integrates with:
- MetricsEngine: For drift detection and performance monitoring
- ModelRegistry: For model performance history
- ExperimentDB: For logging trigger events

Example Usage:
    >>> from utils.training.retraining_triggers import (
    ...     RetrainingTriggerManager, DriftTrigger, PerformanceTrigger, TriggerConfig
    ... )
    >>>
    >>> # Initialize trigger manager
    >>> manager = RetrainingTriggerManager()
    >>>
    >>> # Register drift trigger
    >>> drift_trigger = DriftTrigger(
    ...     threshold=0.15,
    ...     metric_name='js_divergence',
    ...     severity='warning'
    ... )
    >>> manager.register_trigger('drift_monitor', drift_trigger)
    >>>
    >>> # Check if retraining is needed
    >>> report = manager.evaluate(
    ...     drift_metrics={'js_divergence': 0.18},
    ...     current_metrics={'val_loss': 0.45, 'val_accuracy': 0.82}
    ... )
    >>>
    >>> if report.triggered:
    ...     print(f"Retraining recommended: {report.recommendations}")

Author: MLOps Agent (Phase 2-4 - Production Hardening)
Version: 3.8.0
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Protocol, Union
from pathlib import Path

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Trigger Configuration
# -------------------------------------------------------------------------

@dataclass
class TriggerConfig:
    """
    Base configuration for retraining triggers.

    Attributes:
        name: Human-readable trigger name
        enabled: Whether trigger is active
        severity: Trigger severity level (info, warning, critical)
        description: Human-readable description
        metadata: Additional configuration metadata
    """
    name: str
    enabled: bool = True
    severity: Literal['info', 'warning', 'critical'] = 'warning'
    description: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TriggerConfig:
        """Load from dictionary."""
        return cls(**data)

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> TriggerConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class DriftTriggerConfig(TriggerConfig):
    """
    Configuration for drift-based triggers.

    Attributes:
        threshold: JS divergence threshold (0-1, higher = more drift)
        metric_name: Drift metric to monitor (js_divergence, seq_length_js, etc.)
        window_size: Number of recent samples to compare
    """
    threshold: float = 0.15
    metric_name: str = 'js_divergence'
    window_size: int = 1000


@dataclass
class PerformanceTriggerConfig(TriggerConfig):
    """
    Configuration for performance degradation triggers.

    Attributes:
        threshold: Performance change threshold (e.g., 0.05 = 5% degradation)
        metric_name: Performance metric to monitor (val_loss, val_accuracy, etc.)
        mode: Optimization mode ('min' for loss, 'max' for accuracy)
        lookback_epochs: Number of recent epochs to compare against baseline
    """
    threshold: float = 0.05
    metric_name: str = 'val_loss'
    mode: Literal['min', 'max'] = 'min'
    lookback_epochs: int = 5


@dataclass
class TimeTriggerConfig(TriggerConfig):
    """
    Configuration for time-based triggers.

    Attributes:
        interval_hours: Hours between scheduled retraining
        last_training_time: ISO timestamp of last training (None = now)
    """
    interval_hours: float = 168.0  # 1 week default
    last_training_time: Optional[str] = None


@dataclass
class DataVolumeTriggerConfig(TriggerConfig):
    """
    Configuration for data volume triggers.

    Attributes:
        threshold_samples: Number of new samples to trigger retraining
        threshold_percentage: Percentage increase to trigger (e.g., 0.2 = 20%)
        baseline_count: Baseline sample count (None = current count)
    """
    threshold_samples: Optional[int] = None
    threshold_percentage: Optional[float] = 0.2
    baseline_count: Optional[int] = None


# -------------------------------------------------------------------------
# Trigger Report
# -------------------------------------------------------------------------

@dataclass
class TriggerDetail:
    """
    Details about a single trigger evaluation.

    Attributes:
        trigger_name: Name of the trigger
        triggered: Whether trigger fired
        severity: Trigger severity level
        reason: Human-readable reason for triggering
        metrics: Metrics that caused the trigger
        threshold: Threshold value
        actual_value: Actual measured value
    """
    trigger_name: str
    triggered: bool
    severity: Literal['info', 'warning', 'critical']
    reason: str
    metrics: Dict[str, Any]
    threshold: Optional[float] = None
    actual_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RetrainingReport:
    """
    Comprehensive report of trigger evaluation.

    Attributes:
        triggered: Whether any trigger fired
        trigger_details: List of individual trigger results
        recommendations: List of actionable recommendations
        severity: Overall severity (highest among triggered)
        timestamp: ISO timestamp of evaluation
        metadata: Additional context information
    """
    triggered: bool
    trigger_details: List[TriggerDetail]
    recommendations: List[str]
    severity: Literal['info', 'warning', 'critical']
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'triggered': self.triggered,
            'trigger_details': [t.to_dict() for t in self.trigger_details],
            'recommendations': self.recommendations,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown(self) -> str:
        """
        Generate human-readable Markdown report.

        Returns:
            Markdown-formatted report string
        """
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'critical': '🚨'
        }

        lines = [
            f"# Retraining Trigger Report",
            f"",
            f"**Status:** {'🔴 TRIGGERED' if self.triggered else '🟢 OK'}",
            f"**Severity:** {severity_emoji.get(self.severity, '')} {self.severity.upper()}",
            f"**Timestamp:** {self.timestamp}",
            f"",
            f"## Trigger Details",
            f""
        ]

        for detail in self.trigger_details:
            status = '✅ Fired' if detail.triggered else '⬜ Not Fired'
            lines.append(f"### {detail.trigger_name} - {status}")
            lines.append(f"")
            lines.append(f"- **Severity:** {detail.severity}")
            lines.append(f"- **Reason:** {detail.reason}")

            if detail.threshold is not None:
                lines.append(f"- **Threshold:** {detail.threshold:.4f}")
            if detail.actual_value is not None:
                lines.append(f"- **Actual Value:** {detail.actual_value:.4f}")

            if detail.metrics:
                lines.append(f"- **Metrics:**")
                for key, value in detail.metrics.items():
                    lines.append(f"  - {key}: {value}")
            lines.append(f"")

        if self.recommendations:
            lines.append(f"## Recommendations")
            lines.append(f"")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append(f"")

        if self.metadata:
            lines.append(f"## Metadata")
            lines.append(f"")
            lines.append(f"```json")
            lines.append(json.dumps(self.metadata, indent=2))
            lines.append(f"```")

        return '\n'.join(lines)


# -------------------------------------------------------------------------
# Trigger Protocol
# -------------------------------------------------------------------------

class RetrainingTrigger(Protocol):
    """
    Protocol for retraining triggers.

    All trigger implementations must implement the evaluate() method.
    """

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriggerDetail:
        """
        Evaluate trigger condition.

        Args:
            drift_metrics: Drift detection metrics from MetricsEngine
            current_metrics: Current model performance metrics
            baseline_metrics: Baseline performance metrics
            metadata: Additional context (timestamps, data counts, etc.)

        Returns:
            TriggerDetail with evaluation result
        """
        ...


# -------------------------------------------------------------------------
# Trigger Implementations
# -------------------------------------------------------------------------

class DriftTrigger:
    """
    Trigger that fires when data drift exceeds threshold.

    Monitors JS divergence or other drift metrics from MetricsEngine and
    triggers retraining when distribution shift is detected.

    Example:
        >>> trigger = DriftTrigger(
        ...     threshold=0.15,
        ...     metric_name='js_divergence',
        ...     severity='warning'
        ... )
        >>> result = trigger.evaluate(drift_metrics={'js_divergence': 0.18})
        >>> print(result.triggered)  # True
    """

    def __init__(
        self,
        threshold: float = 0.15,
        metric_name: str = 'js_divergence',
        severity: Literal['info', 'warning', 'critical'] = 'warning',
        name: str = 'drift_trigger'
    ):
        """
        Initialize drift trigger.

        Args:
            threshold: JS divergence threshold (0-1, higher = more drift)
            metric_name: Drift metric to monitor
            severity: Trigger severity level
            name: Trigger name
        """
        self.threshold = threshold
        self.metric_name = metric_name
        self.severity = severity
        self.name = name

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriggerDetail:
        """Evaluate drift trigger."""
        if drift_metrics is None:
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason="No drift metrics provided",
                metrics={},
                threshold=self.threshold
            )

        # Extract drift value
        drift_value = drift_metrics.get(self.metric_name)

        if drift_value is None:
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason=f"Metric '{self.metric_name}' not found in drift metrics",
                metrics=drift_metrics,
                threshold=self.threshold
            )

        # Check threshold
        triggered = drift_value > self.threshold

        if triggered:
            reason = (
                f"Drift detected: {self.metric_name}={drift_value:.4f} "
                f"exceeds threshold {self.threshold:.4f}"
            )
        else:
            reason = (
                f"No drift: {self.metric_name}={drift_value:.4f} "
                f"within threshold {self.threshold:.4f}"
            )

        return TriggerDetail(
            trigger_name=self.name,
            triggered=triggered,
            severity=self.severity,
            reason=reason,
            metrics=drift_metrics,
            threshold=self.threshold,
            actual_value=drift_value
        )


class PerformanceTrigger:
    """
    Trigger that fires when model performance degrades.

    Monitors validation loss, accuracy, or other metrics and triggers
    retraining when performance drops below acceptable threshold.

    Example:
        >>> trigger = PerformanceTrigger(
        ...     threshold=0.05,  # 5% degradation
        ...     metric_name='val_loss',
        ...     mode='min'
        ... )
        >>> result = trigger.evaluate(
        ...     current_metrics={'val_loss': 0.45},
        ...     baseline_metrics={'val_loss': 0.40}
        ... )
        >>> print(result.triggered)  # True (12.5% increase)
    """

    def __init__(
        self,
        threshold: float = 0.05,
        metric_name: str = 'val_loss',
        mode: Literal['min', 'max'] = 'min',
        severity: Literal['info', 'warning', 'critical'] = 'warning',
        name: str = 'performance_trigger'
    ):
        """
        Initialize performance trigger.

        Args:
            threshold: Performance change threshold (e.g., 0.05 = 5%)
            metric_name: Performance metric to monitor
            mode: 'min' for loss (lower is better), 'max' for accuracy
            severity: Trigger severity level
            name: Trigger name
        """
        self.threshold = threshold
        self.metric_name = metric_name
        self.mode = mode
        self.severity = severity
        self.name = name

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriggerDetail:
        """Evaluate performance trigger."""
        if current_metrics is None or baseline_metrics is None:
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason="Missing current or baseline metrics",
                metrics={},
                threshold=self.threshold
            )

        current_value = current_metrics.get(self.metric_name)
        baseline_value = baseline_metrics.get(self.metric_name)

        if current_value is None or baseline_value is None:
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason=f"Metric '{self.metric_name}' not found in current or baseline metrics",
                metrics={'current': current_metrics, 'baseline': baseline_metrics},
                threshold=self.threshold
            )

        # Avoid division by zero
        if baseline_value == 0:
            change = float('inf') if current_value > 0 else 0.0
        else:
            change = (current_value - baseline_value) / abs(baseline_value)

        # For 'min' mode (loss), degradation is positive change (increase)
        # For 'max' mode (accuracy), degradation is negative change (decrease)
        if self.mode == 'min':
            degradation = change
            triggered = degradation > self.threshold
        else:  # mode == 'max'
            degradation = -change
            triggered = degradation > self.threshold

        if triggered:
            reason = (
                f"Performance degradation: {self.metric_name} changed by "
                f"{change*100:+.2f}% (baseline={baseline_value:.4f}, "
                f"current={current_value:.4f})"
            )
        else:
            reason = (
                f"Performance stable: {self.metric_name} changed by "
                f"{change*100:+.2f}% within threshold {self.threshold*100:.1f}%"
            )

        return TriggerDetail(
            trigger_name=self.name,
            triggered=triggered,
            severity=self.severity,
            reason=reason,
            metrics={
                'current_value': current_value,
                'baseline_value': baseline_value,
                'change_percent': change * 100
            },
            threshold=self.threshold,
            actual_value=abs(degradation)
        )


class TimeTrigger:
    """
    Trigger that fires after time interval elapsed.

    Implements scheduled retraining based on time since last training.
    Useful for maintaining model freshness even without detected drift.

    Example:
        >>> trigger = TimeTrigger(interval_hours=168)  # 1 week
        >>> result = trigger.evaluate(
        ...     metadata={'last_training_time': '2025-01-01T00:00:00'}
        ... )
        >>> print(result.triggered)  # True if >1 week elapsed
    """

    def __init__(
        self,
        interval_hours: float = 168.0,  # 1 week
        severity: Literal['info', 'warning', 'critical'] = 'info',
        name: str = 'time_trigger'
    ):
        """
        Initialize time trigger.

        Args:
            interval_hours: Hours between scheduled retraining
            severity: Trigger severity level
            name: Trigger name
        """
        self.interval_hours = interval_hours
        self.severity = severity
        self.name = name

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriggerDetail:
        """Evaluate time trigger."""
        if metadata is None:
            metadata = {}

        # Get last training time from metadata
        last_training_str = metadata.get('last_training_time')

        if last_training_str is None:
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason="No last training time provided",
                metrics={},
                threshold=self.interval_hours
            )

        try:
            last_training = datetime.fromisoformat(last_training_str)
        except (ValueError, TypeError):
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason=f"Invalid timestamp format: {last_training_str}",
                metrics={},
                threshold=self.interval_hours
            )

        now = datetime.now()
        elapsed = now - last_training
        elapsed_hours = elapsed.total_seconds() / 3600.0

        triggered = elapsed_hours >= self.interval_hours

        if triggered:
            reason = (
                f"Scheduled retraining due: {elapsed_hours:.1f} hours elapsed "
                f"(threshold: {self.interval_hours:.1f} hours)"
            )
        else:
            remaining_hours = self.interval_hours - elapsed_hours
            reason = (
                f"No retraining needed: {elapsed_hours:.1f} hours elapsed "
                f"({remaining_hours:.1f} hours remaining)"
            )

        return TriggerDetail(
            trigger_name=self.name,
            triggered=triggered,
            severity=self.severity,
            reason=reason,
            metrics={
                'last_training_time': last_training_str,
                'elapsed_hours': elapsed_hours,
                'interval_hours': self.interval_hours
            },
            threshold=self.interval_hours,
            actual_value=elapsed_hours
        )


class DataVolumeTrigger:
    """
    Trigger that fires when new data volume exceeds threshold.

    Monitors dataset size and triggers retraining when sufficient new
    data has been collected.

    Example:
        >>> trigger = DataVolumeTrigger(
        ...     threshold_samples=1000,
        ...     threshold_percentage=0.2
        ... )
        >>> result = trigger.evaluate(
        ...     metadata={'current_count': 6000, 'baseline_count': 5000}
        ... )
        >>> print(result.triggered)  # True (1000 new samples, 20% increase)
    """

    def __init__(
        self,
        threshold_samples: Optional[int] = None,
        threshold_percentage: Optional[float] = 0.2,
        severity: Literal['info', 'warning', 'critical'] = 'info',
        name: str = 'data_volume_trigger'
    ):
        """
        Initialize data volume trigger.

        Args:
            threshold_samples: Absolute number of new samples (OR condition)
            threshold_percentage: Percentage increase (OR condition)
            severity: Trigger severity level
            name: Trigger name
        """
        self.threshold_samples = threshold_samples
        self.threshold_percentage = threshold_percentage
        self.severity = severity
        self.name = name

        if threshold_samples is None and threshold_percentage is None:
            raise ValueError("Must specify at least one of threshold_samples or threshold_percentage")

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriggerDetail:
        """Evaluate data volume trigger."""
        if metadata is None:
            metadata = {}

        current_count = metadata.get('current_count')
        baseline_count = metadata.get('baseline_count')

        if current_count is None or baseline_count is None:
            return TriggerDetail(
                trigger_name=self.name,
                triggered=False,
                severity=self.severity,
                reason="Missing current_count or baseline_count in metadata",
                metrics={},
                threshold=self.threshold_samples or self.threshold_percentage
            )

        new_samples = current_count - baseline_count

        if baseline_count == 0:
            percentage_increase = float('inf') if new_samples > 0 else 0.0
        else:
            percentage_increase = new_samples / baseline_count

        # Check both thresholds (OR condition)
        triggered_by_samples = (
            self.threshold_samples is not None and
            new_samples >= self.threshold_samples
        )
        triggered_by_percentage = (
            self.threshold_percentage is not None and
            percentage_increase >= self.threshold_percentage
        )

        triggered = triggered_by_samples or triggered_by_percentage

        if triggered:
            reasons = []
            if triggered_by_samples:
                reasons.append(
                    f"{new_samples} new samples >= threshold {self.threshold_samples}"
                )
            if triggered_by_percentage and self.threshold_percentage is not None:
                reasons.append(
                    f"{percentage_increase*100:.1f}% increase >= threshold "
                    f"{self.threshold_percentage*100:.1f}%"
                )
            reason = f"Data volume threshold exceeded: {' AND '.join(reasons)}"
        else:
            reason = (
                f"Data volume within threshold: {new_samples} new samples "
                f"({percentage_increase*100:.1f}% increase)"
            )

        return TriggerDetail(
            trigger_name=self.name,
            triggered=triggered,
            severity=self.severity,
            reason=reason,
            metrics={
                'current_count': current_count,
                'baseline_count': baseline_count,
                'new_samples': new_samples,
                'percentage_increase': percentage_increase * 100
            },
            threshold=self.threshold_samples or self.threshold_percentage,
            actual_value=new_samples
        )


class CompositeTrigger:
    """
    Composite trigger that combines multiple triggers with AND/OR logic.

    Enables complex retraining policies like:
    - Retrain if (drift > 0.15 AND performance drops > 5%) OR time > 1 week
    - Retrain if drift > 0.2 OR (performance drops > 10% AND new data > 1000 samples)

    Example:
        >>> drift_trigger = DriftTrigger(threshold=0.15)
        >>> perf_trigger = PerformanceTrigger(threshold=0.05)
        >>> time_trigger = TimeTrigger(interval_hours=168)
        >>>
        >>> # Retrain if (drift AND performance) OR time
        >>> composite = CompositeTrigger(
        ...     triggers=[
        ...         CompositeTrigger([drift_trigger, perf_trigger], logic='AND'),
        ...         time_trigger
        ...     ],
        ...     logic='OR'
        ... )
    """

    def __init__(
        self,
        triggers: List[Union[RetrainingTrigger, 'CompositeTrigger']],
        logic: Literal['AND', 'OR'] = 'OR',
        severity: Literal['info', 'warning', 'critical'] = 'warning',
        name: str = 'composite_trigger'
    ):
        """
        Initialize composite trigger.

        Args:
            triggers: List of triggers to combine
            logic: Combination logic ('AND' or 'OR')
            severity: Trigger severity level
            name: Trigger name
        """
        self.triggers = triggers
        self.logic = logic
        self.severity = severity
        self.name = name

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TriggerDetail:
        """Evaluate composite trigger."""
        # Evaluate all child triggers
        results = []
        for trigger in self.triggers:
            result = trigger.evaluate(
                drift_metrics=drift_metrics,
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                metadata=metadata
            )
            results.append(result)

        # Apply logic
        if self.logic == 'AND':
            triggered = all(r.triggered for r in results)
            reason = f"Composite AND: all {len(results)} triggers must fire"
        else:  # OR
            triggered = any(r.triggered for r in results)
            reason = f"Composite OR: any of {len(results)} triggers must fire"

        # Collect metrics from child triggers
        combined_metrics = {}
        for i, result in enumerate(results):
            combined_metrics[f'trigger_{i}_{result.trigger_name}'] = {
                'triggered': result.triggered,
                'reason': result.reason,
                'metrics': result.metrics
            }

        return TriggerDetail(
            trigger_name=self.name,
            triggered=triggered,
            severity=self.severity,
            reason=reason,
            metrics=combined_metrics
        )


# -------------------------------------------------------------------------
# Trigger Manager
# -------------------------------------------------------------------------

class RetrainingTriggerManager:
    """
    Manager for registering and evaluating retraining triggers.

    Provides centralized interface for:
    - Registering multiple triggers
    - Evaluating all triggers in one call
    - Generating comprehensive reports
    - Logging trigger events to ExperimentDB
    - Integration with MetricsEngine and ModelRegistry

    Example:
        >>> from utils.training.retraining_triggers import RetrainingTriggerManager
        >>> from utils.training.engine.metrics import MetricsEngine
        >>> from utils.training.model_registry import ModelRegistry
        >>> from utils.training.experiment_db import ExperimentDB
        >>>
        >>> # Initialize infrastructure
        >>> engine = MetricsEngine(use_wandb=False)
        >>> registry = ModelRegistry('models.db')
        >>> db = ExperimentDB('experiments.db')
        >>>
        >>> # Create manager
        >>> manager = RetrainingTriggerManager(
        ...     metrics_engine=engine,
        ...     model_registry=registry,
        ...     experiment_db=db
        ... )
        >>>
        >>> # Register triggers
        >>> manager.register_drift_trigger(threshold=0.15)
        >>> manager.register_performance_trigger(threshold=0.05)
        >>> manager.register_time_trigger(interval_hours=168)
        >>>
        >>> # Check if retraining needed
        >>> report = manager.check_retraining_needed()
        >>> if report.triggered:
        ...     print(report.to_markdown())
    """

    def __init__(
        self,
        metrics_engine: Optional[Any] = None,
        model_registry: Optional[Any] = None,
        experiment_db: Optional[Any] = None
    ):
        """
        Initialize trigger manager.

        Args:
            metrics_engine: MetricsEngine instance for drift detection
            model_registry: ModelRegistry instance for performance history
            experiment_db: ExperimentDB instance for logging
        """
        self.metrics_engine = metrics_engine
        self.model_registry = model_registry
        self.experiment_db = experiment_db

        self.triggers: Dict[str, RetrainingTrigger] = {}
        self.trigger_history: List[RetrainingReport] = []

        logger.info("Initialized RetrainingTriggerManager")

    def register_trigger(
        self,
        name: str,
        trigger: RetrainingTrigger
    ) -> None:
        """
        Register a trigger.

        Args:
            name: Unique trigger name
            trigger: Trigger instance

        Example:
            >>> manager = RetrainingTriggerManager()
            >>> drift_trigger = DriftTrigger(threshold=0.15)
            >>> manager.register_trigger('drift_monitor', drift_trigger)
        """
        if name in self.triggers:
            logger.warning(f"Overwriting existing trigger '{name}'")

        self.triggers[name] = trigger
        logger.info(f"Registered trigger '{name}'")

    def register_drift_trigger(
        self,
        threshold: float = 0.15,
        metric_name: str = 'js_divergence',
        severity: Literal['info', 'warning', 'critical'] = 'warning',
        name: str = 'drift_trigger'
    ) -> None:
        """Register a drift trigger (convenience method)."""
        trigger = DriftTrigger(
            threshold=threshold,
            metric_name=metric_name,
            severity=severity,
            name=name
        )
        self.register_trigger(name, trigger)

    def register_performance_trigger(
        self,
        threshold: float = 0.05,
        metric_name: str = 'val_loss',
        mode: Literal['min', 'max'] = 'min',
        severity: Literal['info', 'warning', 'critical'] = 'warning',
        name: str = 'performance_trigger'
    ) -> None:
        """Register a performance trigger (convenience method)."""
        trigger = PerformanceTrigger(
            threshold=threshold,
            metric_name=metric_name,
            mode=mode,
            severity=severity,
            name=name
        )
        self.register_trigger(name, trigger)

    def register_time_trigger(
        self,
        interval_hours: float = 168.0,
        severity: Literal['info', 'warning', 'critical'] = 'info',
        name: str = 'time_trigger'
    ) -> None:
        """Register a time trigger (convenience method)."""
        trigger = TimeTrigger(
            interval_hours=interval_hours,
            severity=severity,
            name=name
        )
        self.register_trigger(name, trigger)

    def register_data_volume_trigger(
        self,
        threshold_samples: Optional[int] = None,
        threshold_percentage: Optional[float] = 0.2,
        severity: Literal['info', 'warning', 'critical'] = 'info',
        name: str = 'data_volume_trigger'
    ) -> None:
        """Register a data volume trigger (convenience method)."""
        trigger = DataVolumeTrigger(
            threshold_samples=threshold_samples,
            threshold_percentage=threshold_percentage,
            severity=severity,
            name=name
        )
        self.register_trigger(name, trigger)

    def evaluate(
        self,
        drift_metrics: Optional[Dict[str, Any]] = None,
        current_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RetrainingReport:
        """
        Evaluate all registered triggers.

        Args:
            drift_metrics: Drift detection metrics from MetricsEngine
            current_metrics: Current model performance metrics
            baseline_metrics: Baseline performance metrics (from ModelRegistry)
            metadata: Additional context (timestamps, data counts, etc.)

        Returns:
            RetrainingReport with evaluation results and recommendations

        Example:
            >>> report = manager.evaluate(
            ...     drift_metrics={'js_divergence': 0.18},
            ...     current_metrics={'val_loss': 0.45, 'val_accuracy': 0.82},
            ...     baseline_metrics={'val_loss': 0.40, 'val_accuracy': 0.85},
            ...     metadata={'last_training_time': '2025-01-01T00:00:00'}
            ... )
            >>> print(f"Triggered: {report.triggered}")
            >>> print(f"Severity: {report.severity}")
        """
        trigger_details = []

        # Evaluate all triggers
        for name, trigger in self.triggers.items():
            detail = trigger.evaluate(
                drift_metrics=drift_metrics,
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                metadata=metadata
            )
            trigger_details.append(detail)

        # Determine overall status
        triggered = any(d.triggered for d in trigger_details)

        # Determine severity (highest among triggered)
        severity_order = {'info': 0, 'warning': 1, 'critical': 2}
        triggered_details = [d for d in trigger_details if d.triggered]

        if triggered_details:
            severity_value = max(
                (d.severity for d in triggered_details),
                key=lambda s: severity_order[s]
            )
            severity: Literal['info', 'warning', 'critical'] = severity_value
        else:
            severity = 'info'

        # Generate recommendations
        recommendations = self._generate_recommendations(trigger_details)

        # Create report
        report = RetrainingReport(
            triggered=triggered,
            trigger_details=trigger_details,
            recommendations=recommendations,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        # Store in history
        self.trigger_history.append(report)

        # Log to ExperimentDB if available
        if self.experiment_db and triggered:
            self._log_trigger_event(report)

        return report

    def check_retraining_needed(
        self,
        model_id: Optional[int] = None,
        run_id: Optional[int] = None
    ) -> RetrainingReport:
        """
        High-level convenience method to check if retraining is needed.

        Automatically fetches drift metrics from MetricsEngine and performance
        metrics from ModelRegistry/ExperimentDB.

        Args:
            model_id: Model ID from ModelRegistry (for performance history)
            run_id: Run ID from ExperimentDB (for performance history)

        Returns:
            RetrainingReport with evaluation results

        Example:
            >>> # Check with explicit model_id
            >>> report = manager.check_retraining_needed(model_id=5)
            >>>
            >>> # Check with explicit run_id
            >>> report = manager.check_retraining_needed(run_id=42)
            >>>
            >>> # Check without IDs (uses latest)
            >>> report = manager.check_retraining_needed()
        """
        # Gather drift metrics from MetricsEngine
        drift_metrics = None
        if self.metrics_engine:
            drift_history = getattr(self.metrics_engine, 'drift_history', [])
            if drift_history:
                latest_drift = drift_history[-1]
                drift_metrics = {
                    'js_divergence': latest_drift.js_divergence,
                    'status': latest_drift.status,
                    **latest_drift.details
                }

        # Gather performance metrics
        current_metrics = None
        baseline_metrics = None
        metadata = {}

        if self.model_registry and model_id:
            # Get model from registry
            model = self.model_registry.get_model(model_id=model_id)
            if model:
                baseline_metrics = model.get('metrics', {})
                metadata['model_id'] = model_id
                metadata['model_version'] = model.get('version')
                metadata['last_training_time'] = model.get('created_at')

        if self.experiment_db and run_id:
            # Get metrics from ExperimentDB
            try:
                run = self.experiment_db.get_run(run_id)
                metrics_df = self.experiment_db.get_metrics(run_id)

                if not metrics_df.empty:
                    # Get latest metrics
                    latest_epoch = metrics_df['epoch'].max()
                    latest_metrics = metrics_df[metrics_df['epoch'] == latest_epoch]

                    current_metrics = {}
                    for _, row in latest_metrics.iterrows():
                        current_metrics[row['metric_name']] = row['value']

                    metadata['run_id'] = run_id
                    metadata['last_training_time'] = run.get('created_at')
            except Exception as e:
                logger.warning(f"Failed to fetch metrics from ExperimentDB: {e}")

        # Evaluate triggers
        return self.evaluate(
            drift_metrics=drift_metrics,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            metadata=metadata
        )

    def _generate_recommendations(
        self,
        trigger_details: List[TriggerDetail]
    ) -> List[str]:
        """Generate actionable recommendations based on triggered conditions."""
        recommendations = []

        for detail in trigger_details:
            if not detail.triggered:
                continue

            if 'drift' in detail.trigger_name.lower():
                recommendations.append(
                    "Data drift detected: Review new data sources and consider "
                    "retraining with expanded dataset to capture distribution changes."
                )
            elif 'performance' in detail.trigger_name.lower():
                recommendations.append(
                    "Performance degradation detected: Retrain model with recent data "
                    "or investigate root cause (data quality, concept drift, etc.)."
                )
            elif 'time' in detail.trigger_name.lower():
                recommendations.append(
                    "Scheduled retraining window reached: Maintain model freshness "
                    "by retraining with latest data."
                )
            elif 'volume' in detail.trigger_name.lower():
                recommendations.append(
                    "Sufficient new data collected: Retrain model to improve coverage "
                    "and generalization with expanded training set."
                )

        if not recommendations:
            recommendations.append("No action needed: All triggers within acceptable thresholds.")

        return recommendations

    def _log_trigger_event(self, report: RetrainingReport) -> None:
        """Log trigger event to ExperimentDB."""
        if not self.experiment_db:
            return

        try:
            # Create a special run for trigger events
            run_id = self.experiment_db.log_run(
                run_name=f"trigger_event_{report.timestamp}",
                config=report.to_dict(),
                notes=f"Retraining trigger event: {report.severity}"
            )

            # Log trigger details as metrics
            for detail in report.trigger_details:
                if detail.triggered:
                    self.experiment_db.log_metric(
                        run_id=run_id,
                        metric_name=f'trigger/{detail.trigger_name}',
                        value=1.0,
                        epoch=0
                    )

                    if detail.actual_value is not None:
                        self.experiment_db.log_metric(
                            run_id=run_id,
                            metric_name=f'trigger/{detail.trigger_name}_value',
                            value=detail.actual_value,
                            epoch=0
                        )

            # Update run status
            self.experiment_db.update_run_status(run_id, 'completed')

            logger.info(f"Logged trigger event to ExperimentDB (run_id={run_id})")

        except Exception as e:
            logger.warning(f"Failed to log trigger event to ExperimentDB: {e}")

    def get_trigger_history(self, limit: int = 10) -> List[RetrainingReport]:
        """
        Get recent trigger evaluation history.

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of RetrainingReport objects
        """
        return self.trigger_history[-limit:]


# -------------------------------------------------------------------------
# Example Configurations
# -------------------------------------------------------------------------

def get_conservative_config() -> Dict[str, TriggerConfig]:
    """
    Conservative trigger configuration for production systems.

    - High drift threshold (0.2)
    - High performance degradation threshold (10%)
    - Long time interval (2 weeks)
    - High data volume threshold (30% increase)

    Returns:
        Dictionary mapping trigger names to configurations
    """
    return {
        'drift': DriftTriggerConfig(
            name='drift_conservative',
            threshold=0.2,
            severity='critical',
            description='Conservative drift detection (high threshold)'
        ),
        'performance': PerformanceTriggerConfig(
            name='performance_conservative',
            threshold=0.10,
            severity='critical',
            description='Conservative performance monitoring (10% degradation)'
        ),
        'time': TimeTriggerConfig(
            name='time_conservative',
            interval_hours=336.0,  # 2 weeks
            severity='info',
            description='Bi-weekly scheduled retraining'
        ),
        'data_volume': DataVolumeTriggerConfig(
            name='data_volume_conservative',
            threshold_percentage=0.30,
            severity='info',
            description='Retrain when data increases by 30%'
        )
    }


def get_aggressive_config() -> Dict[str, TriggerConfig]:
    """
    Aggressive trigger configuration for rapid iteration.

    - Low drift threshold (0.1)
    - Low performance degradation threshold (3%)
    - Short time interval (2 days)
    - Low data volume threshold (10% increase)

    Returns:
        Dictionary mapping trigger names to configurations
    """
    return {
        'drift': DriftTriggerConfig(
            name='drift_aggressive',
            threshold=0.1,
            severity='warning',
            description='Aggressive drift detection (low threshold)'
        ),
        'performance': PerformanceTriggerConfig(
            name='performance_aggressive',
            threshold=0.03,
            severity='warning',
            description='Aggressive performance monitoring (3% degradation)'
        ),
        'time': TimeTriggerConfig(
            name='time_aggressive',
            interval_hours=48.0,  # 2 days
            severity='info',
            description='Frequent scheduled retraining (every 2 days)'
        ),
        'data_volume': DataVolumeTriggerConfig(
            name='data_volume_aggressive',
            threshold_percentage=0.10,
            severity='info',
            description='Retrain when data increases by 10%'
        )
    }


def get_balanced_config() -> Dict[str, TriggerConfig]:
    """
    Balanced trigger configuration (default recommended).

    - Medium drift threshold (0.15)
    - Medium performance degradation threshold (5%)
    - Medium time interval (1 week)
    - Medium data volume threshold (20% increase)

    Returns:
        Dictionary mapping trigger names to configurations
    """
    return {
        'drift': DriftTriggerConfig(
            name='drift_balanced',
            threshold=0.15,
            severity='warning',
            description='Balanced drift detection'
        ),
        'performance': PerformanceTriggerConfig(
            name='performance_balanced',
            threshold=0.05,
            severity='warning',
            description='Balanced performance monitoring (5% degradation)'
        ),
        'time': TimeTriggerConfig(
            name='time_balanced',
            interval_hours=168.0,  # 1 week
            severity='info',
            description='Weekly scheduled retraining'
        ),
        'data_volume': DataVolumeTriggerConfig(
            name='data_volume_balanced',
            threshold_percentage=0.20,
            severity='info',
            description='Retrain when data increases by 20%'
        )
    }


# Public API
__all__ = [
    # Core classes
    'RetrainingTriggerManager',
    'RetrainingTrigger',
    'RetrainingReport',
    'TriggerDetail',

    # Trigger implementations
    'DriftTrigger',
    'PerformanceTrigger',
    'TimeTrigger',
    'DataVolumeTrigger',
    'CompositeTrigger',

    # Configuration
    'TriggerConfig',
    'DriftTriggerConfig',
    'PerformanceTriggerConfig',
    'TimeTriggerConfig',
    'DataVolumeTriggerConfig',

    # Example configs
    'get_conservative_config',
    'get_aggressive_config',
    'get_balanced_config',
]
