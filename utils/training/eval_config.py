"""
Evaluation configuration utilities.

Defines a minimal, serializable EvalConfig that captures the core parameters
needed to evaluate a model on a given task/dataset configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class EvalConfig:
    """
    Evaluation configuration.

    Attributes:
        dataset_id: Dataset preset or identifier (e.g., "lm_tiny_v1", "cls_tiny_v1").
        split: Data split to evaluate on ("train", "validation", "test").
        max_eval_examples: Upper bound on number of examples to evaluate.
        batch_size: Evaluation batch size.
        num_workers: DataLoader worker count.
        max_seq_length: Maximum sequence length for evaluation.
        eval_interval_steps: Interval for running eval during training (steps).
        eval_on_start: Whether to run an eval pass before training begins.
    """

    dataset_id: str
    split: str
    max_eval_examples: int
    batch_size: int
    num_workers: int
    max_seq_length: int
    eval_interval_steps: int
    eval_on_start: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EvalConfig":
        return EvalConfig(
            dataset_id=data["dataset_id"],
            split=data.get("split", "validation"),
            max_eval_examples=int(data.get("max_eval_examples", 512)),
            batch_size=int(data.get("batch_size", 8)),
            num_workers=int(data.get("num_workers", 0)),
            max_seq_length=int(data.get("max_seq_length", 128)),
            eval_interval_steps=int(data.get("eval_interval_steps", 100)),
            eval_on_start=bool(data.get("eval_on_start", True)),
        )


__all__ = ["EvalConfig"]

