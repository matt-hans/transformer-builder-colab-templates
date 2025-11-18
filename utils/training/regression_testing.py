"""
Regression testing utilities for baseline vs candidate model comparison.

This module provides a helper to run both models through the same evaluation
pipeline and compute metric deltas, with optional logging to ExperimentDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


MetricSummary = Dict[str, float]


@dataclass
class RegressionResult:
    """Container for regression comparison output."""

    metrics: Dict[str, Dict[str, float | str]]
    comparison_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"metrics": self.metrics}
        if self.comparison_id is not None:
            out["comparison_id"] = self.comparison_id
        return out


def _classify_metric_delta(
    metric_name: str,
    baseline_val: float,
    candidate_val: float,
    threshold: float,
) -> Dict[str, float | str]:
    """
    Compute delta and status for a single metric.

    For "loss"-like metrics (name contains 'loss' case-insensitive), lower is
    considered better. For other metrics (accuracy, etc.), higher is better.
    """
    delta = candidate_val - baseline_val

    # Determine if higher or lower is better
    is_loss_like = "loss" in metric_name.lower()

    if abs(delta) < threshold:
        status = "neutral"
    else:
        if is_loss_like:
            # Lower loss is better
            status = "improved" if delta < 0 else "regressed"
        else:
            status = "improved" if delta > 0 else "regressed"

    return {
        "baseline": float(baseline_val),
        "candidate": float(candidate_val),
        "delta": float(delta),
        "status": status,
    }


def compare_models(
    baseline_model: Any,
    candidate_model: Any,
    adapter: Any,
    task_spec: Any,
    eval_cfg: Any,
    db: Any | None = None,
    comparison_name: str | None = None,
    threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Compare baseline and candidate models on a held-out eval set.

    This function is intentionally light on dependencies: it expects callers to
    supply a `run_eval_fn` via eval_cfg (for advanced scenarios) or will fall
    back to the standard `run_evaluation` helper from `eval_runner` by
    importing it lazily.

    Args:
        baseline_model: Baseline nn.Module.
        candidate_model: Candidate nn.Module.
        adapter: ModelAdapter instance.
        task_spec: TaskSpec describing the task.
        eval_cfg: EvalConfig or a structure accepted by run_evaluation.
        db: Optional ExperimentDB instance for logging comparisons.
        comparison_name: Optional human-readable comparison name.
        threshold: Minimum absolute delta to treat as non-neutral.

    Returns:
        Dictionary with structure:
        {
            "metrics": {
                "accuracy": {"baseline": ..., "candidate": ..., "delta": ..., "status": "..."},
                "loss": {...},
            },
            "comparison_id": 123,  # if db provided
        }
    """
    from .eval_runner import run_evaluation  # lazy import to avoid cycles

    # Construct a minimal training_config-like object if needed
    training_config = getattr(eval_cfg, "training_config", None)
    if training_config is None:
        # Fallback namespace with required attributes for shapes/vocab if used
        class _DummyCfg:  # pragma: no cover - trivial container
            pass

        training_config = _DummyCfg()

    # Build dataloader using existing utilities
    from .dataset_utilities import build_dataloader

    dataloader = build_dataloader(task_spec, eval_cfg, training_config)

    # Run evaluation for baseline and candidate
    baseline_metrics: Mapping[str, float] = run_evaluation(
        baseline_model,
        adapter,
        task_spec,
        eval_cfg,
        training_config,
        dataloader,
        metrics_tracker=None,
    )

    # Rebuild dataloader to avoid any exhaustion/iterator state issues
    dataloader_candidate = build_dataloader(task_spec, eval_cfg, training_config)

    candidate_metrics: Mapping[str, float] = run_evaluation(
        candidate_model,
        adapter,
        task_spec,
        eval_cfg,
        training_config,
        dataloader_candidate,
        metrics_tracker=None,
    )

    # Compute per-metric deltas
    metrics_result: Dict[str, Dict[str, float | str]] = {}
    metric_names = set(baseline_metrics.keys()) & set(candidate_metrics.keys())

    for metric_name in sorted(metric_names):
        baseline_val = float(baseline_metrics[metric_name])
        candidate_val = float(candidate_metrics[metric_name])
        metrics_result[metric_name] = _classify_metric_delta(
            metric_name,
            baseline_val,
            candidate_val,
            threshold=threshold,
        )

    comparison_id: Optional[int] = None
    # Optional ExperimentDB logging
    if db is not None and metric_names:
        # Try to obtain run_ids from attached attributes if present
        baseline_run_id = getattr(baseline_model, "run_id", None)
        candidate_run_id = getattr(candidate_model, "run_id", None)

        notes_parts = []
        for name in sorted(metric_names):
            entry = metrics_result[name]
            notes_parts.append(
                f"{name}: {entry['delta']:+.4f} ({entry['status']})"
            )
        notes = "; ".join(notes_parts)
        if comparison_name:
            notes = f"{comparison_name} | {notes}"

        if baseline_run_id is not None and candidate_run_id is not None:
            try:
                comparison_id = db.create_comparison(
                    baseline_run_id=baseline_run_id,
                    candidate_run_id=candidate_run_id,
                    notes=notes,
                )
            except Exception:
                comparison_id = None

    result = RegressionResult(metrics=metrics_result, comparison_id=comparison_id)
    return result.to_dict()

