"""
Metrics tracking engine with drift detection and performance alerts.

This module provides the MetricsEngine class for comprehensive metrics tracking:
- W&B integration with gradient accumulation awareness
- Drift detection (JS divergence) between train/val distributions
- Confidence tracking (top-1, top-5, entropy)
- Performance alerts with configurable thresholds
- ExperimentDB integration for local tracking

Example:
    >>> from utils.training.engine.metrics import MetricsEngine, AlertConfig
    >>>
    >>> # Initialize with drift detection and alerts
    >>> engine = MetricsEngine(
    ...     use_wandb=True,
    ...     gradient_accumulation_steps=4,
    ...     drift_threshold_warning=0.1,
    ...     drift_threshold_critical=0.2,
    ...     alert_config=AlertConfig(
    ...         val_loss_spike_threshold=0.2,
    ...         accuracy_drop_threshold=0.05
    ...     )
    ... )
    >>>
    >>> # Log epoch metrics with drift detection
    >>> drift_metrics = engine.log_epoch(
    ...     epoch=5,
    ...     train_metrics={'loss': 0.42, 'accuracy': 0.85},
    ...     val_metrics={'loss': 0.38, 'accuracy': 0.87},
    ...     learning_rate=1e-4,
    ...     gradient_norm=0.5,
    ...     epoch_duration=120.5,
    ...     reference_profile=ref_profile,  # Optional drift detection
    ...     current_profile=curr_profile
    ... )
    >>>
    >>> # Log confidence during validation
    >>> engine.log_confidence(logits, labels, step=100)
    >>>
    >>> # Check for performance alerts
    >>> if engine.has_alerts():
    ...     alerts = engine.get_alerts()
    ...     for alert in alerts:
    ...         print(f"🚨 {alert['type']}: {alert['message']}")

Author: MLOps Agent 6
Version: 3.7.0
Phase: P1-2 (Metrics Tracking with Drift Detection)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Literal, Protocol

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """
    Drift detection metrics between reference and current distributions.

    Attributes:
        js_divergence: Jensen-Shannon divergence (0-1, lower is better)
        status: Drift severity ('healthy', 'warning', 'critical')
        affected_features: List of features with significant drift
        timestamp: ISO 8601 timestamp of detection
        details: Feature-level drift scores
    """
    js_divergence: float
    status: Literal['healthy', 'warning', 'critical']
    affected_features: List[str]
    timestamp: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ConfidenceMetrics:
    """
    Prediction confidence metrics from model logits.

    Attributes:
        top1_confidence: Mean confidence of top-1 predictions (0-1)
        top5_confidence: Mean confidence within top-5 predictions (0-1)
        entropy: Mean prediction entropy (lower = more confident)
        calibration_error: Expected calibration error (optional)
        num_samples: Number of samples analyzed
    """
    top1_confidence: float
    top5_confidence: float
    entropy: float
    num_samples: int
    calibration_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AlertConfig:
    """
    Configuration for performance alerts.

    Attributes:
        val_loss_spike_threshold: Threshold for validation loss spike (e.g., 0.2 = 20% increase)
        accuracy_drop_threshold: Threshold for accuracy drop (e.g., 0.05 = 5% decrease)
        gradient_explosion_threshold: Threshold for gradient norm spike (e.g., 10.0)
        enable_email_alerts: Send email notifications (requires SMTP config)
        enable_slack_alerts: Send Slack notifications (requires webhook URL)
    """
    val_loss_spike_threshold: float = 0.2  # 20% spike
    accuracy_drop_threshold: float = 0.05  # 5% drop
    gradient_explosion_threshold: float = 10.0
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False


class AlertCallback(Protocol):
    """Protocol for alert callback functions."""
    def __call__(self, alert_type: str, message: str, metrics: Dict[str, Any]) -> None:
        """Execute alert callback with context."""
        ...


class MetricsEngine:
    """
    Comprehensive metrics tracking engine with drift detection and alerts.

    This engine extends the functionality of the original MetricsTracker with:
    1. Drift detection using JS divergence
    2. Confidence tracking for prediction quality
    3. Performance alerts with configurable thresholds
    4. ExperimentDB integration for local tracking

    Thread-safe for use with multi-worker DataLoader.

    Example:
        >>> engine = MetricsEngine(
        ...     use_wandb=True,
        ...     gradient_accumulation_steps=4,
        ...     drift_threshold_warning=0.1,
        ...     drift_threshold_critical=0.2
        ... )
        >>>
        >>> # Training loop
        >>> for epoch in range(n_epochs):
        ...     train_metrics = train_epoch(model, train_loader)
        ...     val_metrics = validate(model, val_loader)
        ...
        ...     drift = engine.log_epoch(
        ...         epoch=epoch,
        ...         train_metrics=train_metrics,
        ...         val_metrics=val_metrics,
        ...         learning_rate=scheduler.get_last_lr()[0],
        ...         gradient_norm=grad_norm,
        ...         epoch_duration=epoch_time
        ...     )
        ...
        ...     if drift and drift.status != 'healthy':
        ...         print(f"⚠️ Drift detected: {drift.status}")
    """

    def __init__(
        self,
        use_wandb: bool = True,
        gradient_accumulation_steps: int = 1,
        drift_threshold_warning: float = 0.1,
        drift_threshold_critical: float = 0.2,
        alert_config: Optional[AlertConfig] = None,
        alert_callbacks: Optional[List[AlertCallback]] = None,
        experiment_db: Optional[Any] = None,  # ExperimentDB instance
        run_id: Optional[int] = None
    ):
        """
        Initialize metrics engine.

        Args:
            use_wandb: Enable W&B logging (default: True)
            gradient_accumulation_steps: Gradient accumulation steps for effective step tracking
            drift_threshold_warning: JS divergence threshold for warning status (default: 0.1)
            drift_threshold_critical: JS divergence threshold for critical status (default: 0.2)
            alert_config: Alert configuration (default: AlertConfig())
            alert_callbacks: Custom alert callbacks (default: console logging)
            experiment_db: ExperimentDB instance for local tracking
            run_id: Run ID for ExperimentDB logging

        Example:
            >>> from utils.training.experiment_db import ExperimentDB
            >>> db = ExperimentDB('experiments.db')
            >>> run_id = db.log_run('experiment-1', config={})
            >>>
            >>> engine = MetricsEngine(
            ...     use_wandb=True,
            ...     gradient_accumulation_steps=4,
            ...     drift_threshold_warning=0.1,
            ...     alert_config=AlertConfig(val_loss_spike_threshold=0.2),
            ...     experiment_db=db,
            ...     run_id=run_id
            ... )
        """
        self.use_wandb = use_wandb
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.drift_threshold_warning = drift_threshold_warning
        self.drift_threshold_critical = drift_threshold_critical
        self.alert_config = alert_config or AlertConfig()
        self.alert_callbacks = alert_callbacks or [self._default_alert_callback]
        self.experiment_db = experiment_db
        self.run_id = run_id

        # Metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self._step_metrics: List[Dict[str, Any]] = []
        self._global_step = 0
        self._lock = threading.Lock()

        # Drift tracking
        self.drift_history: List[DriftMetrics] = []
        self.reference_profile: Optional[Dict[str, Any]] = None

        # Alert tracking
        self._alerts: List[Dict[str, Any]] = []
        self._previous_metrics: Dict[str, float] = {}

        logger.info(
            f"MetricsEngine initialized: wandb={use_wandb}, "
            f"accumulation={gradient_accumulation_steps}, "
            f"drift_thresholds=({drift_threshold_warning}, {drift_threshold_critical})"
        )

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        gradient_norm: float,
        epoch_duration: float,
        reference_profile: Optional[Dict[str, Any]] = None,
        current_profile: Optional[Dict[str, Any]] = None
    ) -> Optional[DriftMetrics]:
        """
        Log epoch-level metrics with optional drift detection.

        Args:
            epoch: Current epoch number (0-indexed)
            train_metrics: Dict with 'loss' and 'accuracy' keys
            val_metrics: Dict with 'loss' and 'accuracy' keys
            learning_rate: Current learning rate from scheduler
            gradient_norm: Maximum gradient norm this epoch
            epoch_duration: Time taken for epoch (seconds)
            reference_profile: Reference dataset profile (optional)
            current_profile: Current dataset profile for drift comparison (optional)

        Returns:
            DriftMetrics if drift detection performed, else None

        Example:
            >>> drift = engine.log_epoch(
            ...     epoch=5,
            ...     train_metrics={'loss': 0.42, 'accuracy': 0.85},
            ...     val_metrics={'loss': 0.38, 'accuracy': 0.87},
            ...     learning_rate=1e-4,
            ...     gradient_norm=0.5,
            ...     epoch_duration=120.5,
            ...     reference_profile=ref_profile,
            ...     current_profile=curr_profile
            ... )
            >>> if drift and drift.status == 'critical':
            ...     print("🚨 Critical drift detected!")
        """
        # Compute derived metrics
        train_ppl = self._compute_perplexity(train_metrics['loss'])
        val_ppl = self._compute_perplexity(val_metrics['loss'])

        # Compile metrics with namespace prefixes
        metrics_dict = {
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/perplexity': train_ppl,
            'train/accuracy': train_metrics['accuracy'],
            'val/loss': val_metrics['loss'],
            'val/perplexity': val_ppl,
            'val/accuracy': val_metrics['accuracy'],
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            'epoch_duration': epoch_duration,
        }

        # Add GPU metrics if available
        if torch.cuda.is_available():
            gpu_memory_bytes = torch.cuda.max_memory_allocated()
            metrics_dict['system/gpu_memory_mb'] = gpu_memory_bytes / (1024**2)
            metrics_dict['system/gpu_utilization'] = self._get_gpu_utilization()

        # Drift detection
        drift_metrics = None
        if reference_profile and current_profile:
            drift_metrics = self.check_drift(reference_profile, current_profile)

            # Add drift metrics to metrics_dict
            metrics_dict['drift/js_divergence'] = drift_metrics.js_divergence
            # drift/status is logged separately (string, not numeric)

            # Log to ExperimentDB
            if self.experiment_db and self.run_id:
                self._log_drift_to_db(drift_metrics, epoch)

        # Performance alerts
        self._check_performance_alerts(metrics_dict, epoch)

        # Log to W&B
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics_dict, step=epoch)
            except ImportError:
                logger.debug("W&B not available, skipping W&B logging")
            except Exception as e:
                logger.warning(f"W&B logging failed for epoch {epoch}: {e}")

        # Log to ExperimentDB
        if self.experiment_db and self.run_id:
            self._log_epoch_to_db(metrics_dict, epoch)

        # Store locally
        with self._lock:
            self.metrics_history.append(metrics_dict)
            self._previous_metrics = metrics_dict.copy()

        # Console output
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ppl={val_ppl:.2f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        return drift_metrics

    def log_scalar(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """
        Log a scalar metric at a specific training step.

        Thread-safe for multi-worker DataLoader. When gradient_accumulation_steps > 1,
        calculates effective_step and only commits to W&B at accumulation boundaries.

        Args:
            metric_name: Metric identifier (e.g., 'train/learning_rate', 'gpu/memory_mb')
            value: Numeric value to log
            step: Training step/batch index (auto-increments if None)
            commit: Whether to commit to W&B (only at accumulation boundaries)

        Raises:
            ValueError: If metric_name is empty or value is non-numeric

        Example:
            >>> # Log per-batch metrics
            >>> for batch_idx, batch in enumerate(dataloader):
            ...     loss = train_batch(batch)
            ...     engine.log_scalar('train/batch_loss', loss.item(), step=batch_idx)
            ...     # W&B commit only at steps 0, 4, 8, ... (accumulation=4)
        """
        # Validation
        if not metric_name or not isinstance(metric_name, str):
            raise ValueError("metric_name must be a non-empty string")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"value must be numeric, got {type(value).__name__}")

        # Auto-increment step if not provided
        if step is None:
            with self._lock:
                step = self._global_step
                self._global_step += 1

        # Calculate effective step (optimizer updates)
        effective_step = step // self.gradient_accumulation_steps

        # Determine if we should commit to W&B (only at accumulation boundaries)
        should_commit = commit and (step % self.gradient_accumulation_steps == 0)

        # Log to W&B with effective step
        if self.use_wandb:
            try:
                import wandb
                wandb.log({metric_name: value}, step=effective_step, commit=should_commit)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"W&B logging failed: {e}")

        # Store internally
        with self._lock:
            self._step_metrics.append({
                'step': step,
                'effective_step': effective_step,
                'metric': metric_name,
                'value': float(value),
                'timestamp': datetime.now().isoformat()
            })

    def log_confidence(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        ignore_index: int = -100
    ) -> ConfidenceMetrics:
        """
        Log prediction confidence metrics from model logits.

        Computes top-1, top-5 confidence, and prediction entropy. Useful for
        monitoring model calibration and uncertainty over training.

        Args:
            logits: Model logits [batch_size, seq_len, vocab_size] or [batch_size, num_classes]
            labels: Target labels [batch_size, seq_len] or [batch_size]
            step: Training step number
            ignore_index: Label value to ignore (default: -100 for padding)

        Returns:
            ConfidenceMetrics with top-1, top-5, entropy values

        Example:
            >>> # During validation loop
            >>> for batch_idx, batch in enumerate(val_loader):
            ...     logits = model(batch['input_ids'])
            ...     labels = batch['labels']
            ...     confidence = engine.log_confidence(logits, labels, step=batch_idx)
            ...     print(f"Top-1 confidence: {confidence.top1_confidence:.3f}")
        """
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Flatten for sequence models
        if probs.dim() == 3:
            probs = probs.view(-1, probs.size(-1))
            labels = labels.view(-1)

        # Mask out ignored positions
        mask = labels != ignore_index
        probs_masked = probs[mask]
        labels_masked = labels[mask]

        if probs_masked.numel() == 0:
            logger.warning("All labels are ignore_index, cannot compute confidence")
            return ConfidenceMetrics(
                top1_confidence=0.0,
                top5_confidence=0.0,
                entropy=0.0,
                num_samples=0
            )

        # Top-1 confidence (confidence of predicted class)
        top1_probs, top1_indices = probs_masked.max(dim=-1)
        top1_confidence = top1_probs.mean().item()

        # Top-5 confidence (sum of top-5 probabilities)
        k = min(5, probs_masked.size(-1))
        topk_probs, _ = probs_masked.topk(k, dim=-1)
        top5_confidence = topk_probs.sum(dim=-1).mean().item()

        # Entropy (measure of uncertainty)
        entropy = -(probs_masked * torch.log(probs_masked + 1e-10)).sum(dim=-1).mean().item()

        confidence_metrics = ConfidenceMetrics(
            top1_confidence=top1_confidence,
            top5_confidence=top5_confidence,
            entropy=entropy,
            num_samples=int(probs_masked.size(0))
        )

        # Log to W&B
        effective_step = step // self.gradient_accumulation_steps
        if self.use_wandb:
            try:
                import wandb
                wandb.log({
                    'confidence/top1': top1_confidence,
                    'confidence/top5': top5_confidence,
                    'confidence/entropy': entropy
                }, step=effective_step)

                # Log histogram of top-1 confidences (custom chart)
                wandb.log({
                    'confidence/top1_histogram': wandb.Histogram(top1_probs.cpu().numpy().tolist())
                }, step=effective_step)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"W&B confidence logging failed: {e}")

        return confidence_metrics

    def check_drift(
        self,
        reference_profile: Dict[str, Any],
        current_profile: Dict[str, Any]
    ) -> DriftMetrics:
        """
        Detect distribution drift between reference and current profiles.

        Uses Jensen-Shannon divergence to compare distributions. Supports both
        text (sequence length, token frequency) and vision (brightness, channel stats).

        Args:
            reference_profile: Reference dataset profile (from compute_dataset_profile)
            current_profile: Current dataset profile to compare

        Returns:
            DriftMetrics with JS divergence, status, and affected features

        Example:
            >>> from utils.training.drift_metrics import compute_dataset_profile
            >>>
            >>> # Profile datasets
            >>> ref_profile = compute_dataset_profile(train_dataset, task_spec)
            >>> curr_profile = compute_dataset_profile(val_dataset, task_spec)
            >>>
            >>> # Detect drift
            >>> drift = engine.check_drift(ref_profile, curr_profile)
            >>> print(f"Drift status: {drift.status}")
            >>> print(f"Affected features: {drift.affected_features}")
        """
        # Import drift utilities
        from ..drift_metrics import compare_profiles

        # Compute drift scores
        comparison = compare_profiles(reference_profile, current_profile)

        drift_scores = comparison['drift_scores']
        max_drift = comparison['max_drift']
        status_str = comparison['status']

        # Map status string to literal type
        if status_str == 'alert':
            status: Literal['healthy', 'warning', 'critical'] = 'critical'
        elif status_str == 'warn':
            status = 'warning'
        else:
            status = 'healthy'

        # Identify affected features
        affected_features = []
        for feature, score in drift_scores.items():
            # For overlap metrics, use (1 - overlap) as distance
            if feature == 'token_overlap':
                distance = 1.0 - score
            else:
                distance = score

            if distance > self.drift_threshold_warning:
                affected_features.append(feature)

        drift_metrics = DriftMetrics(
            js_divergence=max_drift,
            status=status,
            affected_features=affected_features,
            timestamp=datetime.now().isoformat(),
            details=drift_scores
        )

        # Store in history
        with self._lock:
            self.drift_history.append(drift_metrics)

        # Trigger alert if critical
        if status == 'critical':
            self.trigger_alert(
                alert_type='drift_critical',
                message=f"Critical drift detected (JS={max_drift:.3f}). Affected: {', '.join(affected_features)}",
                metrics={'drift': drift_metrics.to_dict()}
            )
        elif status == 'warning':
            logger.warning(
                f"⚠️ Drift warning (JS={max_drift:.3f}). Affected: {', '.join(affected_features)}"
            )

        return drift_metrics

    def trigger_alert(
        self,
        alert_type: str,
        message: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Trigger alert callbacks with context.

        Args:
            alert_type: Alert type identifier (e.g., 'val_loss_spike', 'drift_critical')
            message: Human-readable alert message
            metrics: Context metrics for the alert

        Example:
            >>> def slack_alert(alert_type: str, message: str, metrics: Dict):
            ...     # Send to Slack webhook
            ...     slack.post(message)
            >>>
            >>> engine = MetricsEngine(alert_callbacks=[slack_alert])
            >>> engine.trigger_alert('val_loss_spike', 'Loss increased 25%', {...})
        """
        alert = {
            'type': alert_type,
            'message': message,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        with self._lock:
            self._alerts.append(alert)

        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message, metrics)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def has_alerts(self) -> bool:
        """Check if there are any alerts."""
        with self._lock:
            return len(self._alerts) > 0

    def get_alerts(self, clear: bool = True) -> List[Dict[str, Any]]:
        """
        Get all alerts and optionally clear them.

        Args:
            clear: Whether to clear alerts after retrieval (default: True)

        Returns:
            List of alert dictionaries

        Example:
            >>> if engine.has_alerts():
            ...     alerts = engine.get_alerts()
            ...     for alert in alerts:
            ...         print(f"🚨 {alert['message']}")
        """
        with self._lock:
            alerts = self._alerts.copy()
            if clear:
                self._alerts.clear()
        return alerts

    def get_summary(self) -> pd.DataFrame:
        """
        Get all epoch-level metrics as DataFrame.

        Returns:
            DataFrame with one row per epoch, all metric columns

        Example:
            >>> df = engine.get_summary()
            >>> print(df[['epoch', 'train/loss', 'val/loss', 'drift/status']])
        """
        with self._lock:
            return pd.DataFrame(self.metrics_history)

    def get_step_metrics(self) -> pd.DataFrame:
        """
        Get all step-level metrics as DataFrame.

        Returns:
            DataFrame with columns ['step', 'effective_step', 'metric', 'value', 'timestamp']

        Example:
            >>> df = engine.get_step_metrics()
            >>> loss_df = df[df['metric'] == 'train/batch_loss']
            >>> plt.plot(loss_df['effective_step'], loss_df['value'])
        """
        with self._lock:
            df = pd.DataFrame(self._step_metrics)
        return df.sort_values('step') if not df.empty else df

    def get_best_epoch(
        self,
        metric: str = 'val/loss',
        mode: Literal['min', 'max'] = 'min'
    ) -> int:
        """
        Find epoch with best metric value.

        Args:
            metric: Metric name to optimize (default: 'val/loss')
            mode: 'min' to minimize, 'max' to maximize (default: 'min')

        Returns:
            Epoch number with best metric value

        Example:
            >>> best_epoch = engine.get_best_epoch('val/loss', 'min')
            >>> print(f"Best model at epoch {best_epoch}")
        """
        df = self.get_summary()
        if df.empty:
            raise ValueError("No metrics logged yet")

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")

        if mode == 'min':
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        return int(df.loc[best_idx, 'epoch'])

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from cross-entropy loss with overflow protection."""
        clipped_loss = min(loss, 100.0)
        return float(np.exp(clipped_loss))

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage via nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=False,
                timeout=1
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _check_performance_alerts(self, metrics_dict: Dict[str, float], epoch: int) -> None:
        """Check for performance degradation and trigger alerts."""
        if not self._previous_metrics:
            return  # No baseline for comparison

        # Check validation loss spike
        if 'val/loss' in metrics_dict and 'val/loss' in self._previous_metrics:
            prev_loss = self._previous_metrics['val/loss']
            curr_loss = metrics_dict['val/loss']
            loss_change = (curr_loss - prev_loss) / prev_loss

            if loss_change > self.alert_config.val_loss_spike_threshold:
                self.trigger_alert(
                    alert_type='val_loss_spike',
                    message=f"Validation loss spiked by {loss_change*100:.1f}% at epoch {epoch}",
                    metrics={'prev_loss': prev_loss, 'curr_loss': curr_loss, 'change': loss_change}
                )

        # Check accuracy drop
        if 'val/accuracy' in metrics_dict and 'val/accuracy' in self._previous_metrics:
            prev_acc = self._previous_metrics['val/accuracy']
            curr_acc = metrics_dict['val/accuracy']
            acc_drop = prev_acc - curr_acc

            if acc_drop > self.alert_config.accuracy_drop_threshold:
                self.trigger_alert(
                    alert_type='accuracy_drop',
                    message=f"Validation accuracy dropped by {acc_drop*100:.1f}% at epoch {epoch}",
                    metrics={'prev_acc': prev_acc, 'curr_acc': curr_acc, 'drop': acc_drop}
                )

        # Check gradient explosion
        if 'gradient_norm' in metrics_dict:
            grad_norm = metrics_dict['gradient_norm']
            if grad_norm > self.alert_config.gradient_explosion_threshold:
                self.trigger_alert(
                    alert_type='gradient_explosion',
                    message=f"Gradient norm exploded to {grad_norm:.2f} at epoch {epoch}",
                    metrics={'gradient_norm': grad_norm}
                )

    def _log_drift_to_db(self, drift_metrics: DriftMetrics, epoch: int) -> None:
        """Log drift metrics to ExperimentDB."""
        if not self.experiment_db or not self.run_id:
            return

        try:
            # Log drift scores as individual metrics
            for feature, score in drift_metrics.details.items():
                self.experiment_db.log_metric(
                    run_id=self.run_id,
                    metric_name=f'drift/{feature}',
                    value=score,
                    epoch=epoch
                )

            # Log overall drift status
            self.experiment_db.log_metric(
                run_id=self.run_id,
                metric_name='drift/js_divergence',
                value=drift_metrics.js_divergence,
                epoch=epoch
            )

            # Log drift metadata as artifact
            self.experiment_db.log_artifact(
                run_id=self.run_id,
                artifact_type='drift_metrics',
                filepath=f'drift_epoch{epoch:04d}',
                metadata=drift_metrics.to_dict()
            )
        except Exception as e:
            logger.warning(f"Failed to log drift to ExperimentDB: {e}")

    def _log_epoch_to_db(self, metrics_dict: Dict[str, Any], epoch: int) -> None:
        """Log epoch metrics to ExperimentDB."""
        if not self.experiment_db or not self.run_id:
            return

        try:
            for metric_name, value in metrics_dict.items():
                if metric_name == 'epoch':
                    continue
                self.experiment_db.log_metric(
                    run_id=self.run_id,
                    metric_name=metric_name,
                    value=float(value) if isinstance(value, (int, float, np.number)) else 0.0,
                    epoch=epoch
                )
        except Exception as e:
            logger.warning(f"Failed to log epoch metrics to ExperimentDB: {e}")

    @staticmethod
    def _default_alert_callback(alert_type: str, message: str, metrics: Dict[str, Any]) -> None:
        """Default console alert callback."""
        logger.warning(f"🚨 ALERT [{alert_type}]: {message}")


__all__ = [
    'MetricsEngine',
    'DriftMetrics',
    'ConfidenceMetrics',
    'AlertConfig',
    'AlertCallback'
]
