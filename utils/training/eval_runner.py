"""
Generic evaluation runner.

Provides a simple, adapter-aware evaluation loop that computes task metrics
and logs them to a metrics tracker when provided.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import torch
from torch.utils.data import DataLoader

from .metrics_utils import calculate_perplexity


def _compute_text_metrics(task_type: str, loss_sum: float, count: int, correct: int = 0) -> Dict[str, float]:
    """
    Compute aggregate metrics for text tasks.

    Args:
        task_type: High-level task type ("lm" or "classification").
        loss_sum: Sum of loss values over all batches.
        count: Number of batches.
        correct: Number of correct predictions (only for classification).

    Returns:
        Dictionary containing averaged loss and task-specific metrics.
    """
    avg_loss = loss_sum / max(1, count)
    metrics: Dict[str, float] = {"loss": float(avg_loss)}
    if task_type == "lm":
        metrics["perplexity"] = float(calculate_perplexity(avg_loss))
    if task_type == "classification":
        metrics["accuracy"] = float(correct / max(1, count))
    return metrics


def _compute_vision_metrics(
    loss_sum: float,
    example_count: int,
    top1_correct: int,
    top3_correct: int,
    top5_correct: int,
) -> Dict[str, float]:
    """
    Compute aggregate metrics for vision classification tasks.

    Metrics are aggregated globally across all examples rather than as an
    average of per-batch accuracies.

    Args:
        loss_sum: Sum of loss values over all batches.
        example_count: Total number of evaluated examples.
        top1_correct: Number of examples with correct top-1 prediction.
        top3_correct: Number of examples with correct prediction in top-3.
        top5_correct: Number of examples with correct prediction in top-5.

    Returns:
        Dictionary with loss, accuracy, top-3 accuracy and top-5 accuracy.
    """
    denom = max(1, example_count)
    avg_loss = loss_sum / max(1, denom)
    return {
        "loss": float(avg_loss),
        "accuracy": float(top1_correct / denom),
        "top3_accuracy": float(top3_correct / denom),
        "top5_accuracy": float(top5_correct / denom),
    }


@torch.no_grad()
def run_evaluation(
    model: Any,
    adapter: Any,
    task: Any,
    eval_config: Any,
    training_config: Any,
    dataloader: DataLoader,
    metrics_tracker: Any | None,
) -> Dict[str, float]:
    """
    Runs evaluation loop, logs metrics via metrics_tracker, and returns summary.

    Args:
        model: PyTorch model
        adapter: ModelAdapter instance
        task: TaskSpec
        eval_config: EvalConfig
        training_config: TrainingConfig-like (for shapes/vocab if needed)
        dataloader: PyTorch DataLoader yielding dict batches
        metrics_tracker: Optional metrics tracker with log_scalar/get_summary

    Returns:
        Dict with averaged metrics for the eval set.
    """
    device = next(model.parameters()).device
    model.eval()

    loss_sum = 0.0
    count = 0
    correct_sum = 0

    for batch in dataloader:
        # Move to device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        prepared = adapter.prepare_inputs(batch, task)
        loss, outputs = adapter.forward_for_loss(model, prepared, task)

        if loss is None:
            # Some adapters may not compute loss if labels missing; try to derive if possible
            # Default to zero loss in this degenerate case
            loss_val = torch.tensor(0.0, device=device)
        else:
            loss_val = loss.detach()

        loss_sum += float(loss_val.item())
        count += 1

        # Optional accuracy for CLS or LM token-wise next-token accuracy
        if task.task_type == "classification":
            logits = adapter.get_logits(outputs, task)
            preds = logits.argmax(dim=-1)
            labels = prepared.get("labels")
            if labels is not None:
                correct_sum += int((preds == labels).sum().item())
        elif task.task_type == "lm":
            # Token-level next-token accuracy (rough estimate)
            logits = adapter.get_logits(outputs, task)
            labels = prepared.get("labels")
            if logits is not None and labels is not None and logits.dim() == 3:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                preds = shift_logits.argmax(dim=-1)
                correct_sum += int((preds == shift_labels).float().mean().item() > 0)  # count per batch

    summary = _compute_metrics(task.task_type, loss_sum, count, correct_sum)

    # Log to tracker if provided
    if metrics_tracker is not None:
        for k, v in summary.items():
            try:
                metrics_tracker.log_scalar(f"eval/{k}", float(v))
            except Exception:
                pass

    return summary
