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
    batch_count = 0
    correct_sum = 0

    # Vision-specific aggregation
    vision_top1_correct = 0
    vision_top3_correct = 0
    vision_top5_correct = 0
    vision_example_count = 0

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
        batch_count += 1

        # Optional accuracy for CLS or LM token-wise next-token accuracy
        if getattr(task, "task_type", None) == "classification":
            logits = adapter.get_logits(outputs, task)
            preds = logits.argmax(dim=-1)
            labels = prepared.get("labels")
            if labels is not None:
                correct_sum += int((preds == labels).sum().item())
        elif getattr(task, "task_type", None) == "lm":
            # Token-level next-token accuracy (rough estimate)
            logits = adapter.get_logits(outputs, task)
            labels = prepared.get("labels")
            if logits is not None and labels is not None and logits.dim() == 3:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                preds = shift_logits.argmax(dim=-1)
                correct_sum += int((preds == shift_labels).float().mean().item() > 0)  # count per batch
        elif getattr(task, "modality", None) == "vision" and getattr(task, "task_type", None) == "vision_classification":
            logits = adapter.get_logits(outputs, task)
            labels = prepared.get("labels")
            if labels is not None:
                # Ensure [B, C]
                if logits.dim() > 2:
                    logits = logits.view(logits.size(0), -1)
                _, num_classes = logits.shape
                batch_size = int(labels.shape[0])

                # Top-1
                top1_pred = logits.argmax(dim=-1)
                vision_top1_correct += int((top1_pred == labels).sum().item())

                # Top-k
                k3 = min(3, num_classes)
                k5 = min(5, num_classes)

                top3 = logits.topk(k3, dim=-1).indices
                top5 = logits.topk(k5, dim=-1).indices

                labels_expanded = labels.view(-1, 1)
                vision_top3_correct += int((top3 == labels_expanded).any(dim=-1).sum().item())
                vision_top5_correct += int((top5 == labels_expanded).any(dim=-1).sum().item())

                vision_example_count += batch_size

    # Final metrics routing
    if getattr(task, "modality", None) == "vision" and getattr(task, "task_type", None) == "vision_classification":
        summary = _compute_vision_metrics(
            loss_sum=loss_sum,
            example_count=vision_example_count,
            top1_correct=vision_top1_correct,
            top3_correct=vision_top3_correct,
            top5_correct=vision_top5_correct,
        )
    else:
        summary = _compute_text_metrics(
            getattr(task, "task_type", ""),
            loss_sum,
            batch_count,
            correct_sum,
        )

    # Log to tracker if provided
    if metrics_tracker is not None:
        for k, v in summary.items():
            try:
                metrics_tracker.log_scalar(f"eval/{k}", float(v))
            except Exception:
                pass

    return summary
