"""
Task and evaluation task specification utilities.

Defines a lightweight, serializable TaskSpec that describes the semantics of a
training/evaluation task independently of any specific model implementation.

This enables architecture-agnostic training/evaluation by pairing TaskSpec with
an appropriate ModelAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class TaskSpec:
    """
    Describes a task's semantics and expected model I/O.

    Attributes:
        name: Human-friendly preset name (e.g., "lm_tiny", "cls_tiny").
        task_type: High-level task type ("lm", "classification", "seq2seq").
        model_family: Model family the task expects ("decoder_only", "encoder_only", "encoder_decoder").
        input_fields: Names of input fields a batch will provide (e.g., ["input_ids", "attention_mask"]).
        target_field: Name of the target field in the batch (e.g., "labels"). None if not applicable.
        loss_type: Primary loss to optimize (e.g., "cross_entropy", "mse").
        metrics: List of metric identifiers to compute (e.g., ["loss", "perplexity", "accuracy"]).
        special_tokens: Mapping of token-role to token IDs (if any) used by the task (e.g., {"pad_token_id": 0}).
        additional_config: Freeform extra config used by adapters/datasets (small, JSON-serializable values only).
    """

    name: str
    task_type: str
    model_family: str
    input_fields: List[str]
    target_field: Optional[str]
    loss_type: str
    metrics: List[str]
    special_tokens: Dict[str, int] = field(default_factory=dict)
    additional_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskSpec":
        """Deserialize from a dict (raises KeyError/TypeError on invalid input)."""
        return TaskSpec(
            name=data["name"],
            task_type=data["task_type"],
            model_family=data["model_family"],
            input_fields=list(data.get("input_fields", [])),
            target_field=data.get("target_field"),
            loss_type=data["loss_type"],
            metrics=list(data.get("metrics", [])),
            special_tokens=dict(data.get("special_tokens", {})),
            additional_config=dict(data.get("additional_config", {})),
        )


# Backwards-compatible helper names suggested in the spec
def load_task_spec_from_dict(data: Dict[str, Any]) -> TaskSpec:
    """Alias for TaskSpec.from_dict for clearer callsites."""
    return TaskSpec.from_dict(data)


def get_default_task_specs() -> Dict[str, TaskSpec]:
    """
    Return built-in tiny presets for fast local/Colab validation.

    These are intentionally minimal and are used by notebooks/CLI to provide
    a frictionless starting point. Larger presets can extend these in-place.
    """
    return {
        # Language Modeling (decoder-only)
        "lm_tiny": TaskSpec(
            name="lm_tiny",
            task_type="lm",
            model_family="decoder_only",
            input_fields=["input_ids", "attention_mask"],
            target_field="labels",
            loss_type="cross_entropy",
            metrics=["loss", "perplexity", "accuracy"],
            special_tokens={"pad_token_id": 0},
            additional_config={"shift_labels": True},
        ),

        # Text Classification (encoder-only)
        "cls_tiny": TaskSpec(
            name="cls_tiny",
            task_type="classification",
            model_family="encoder_only",
            input_fields=["input_ids", "attention_mask"],
            target_field="labels",
            loss_type="cross_entropy",
            metrics=["loss", "accuracy"],
            special_tokens={"pad_token_id": 0},
            additional_config={"num_classes": 2},
        ),

        # Seq2Seq (encoder–decoder)
        "seq2seq_tiny": TaskSpec(
            name="seq2seq_tiny",
            task_type="seq2seq",
            model_family="encoder_decoder",
            input_fields=["input_ids", "attention_mask", "decoder_input_ids"],
            target_field="labels",
            loss_type="cross_entropy",
            metrics=["loss"],  # simple default; BLEU/etc. added in metrics_utils later
            special_tokens={"pad_token_id": 0},
            additional_config={"teacher_forcing": True},
        ),
    }


__all__ = [
    "TaskSpec",
    "get_default_task_specs",
    "load_task_spec_from_dict",
]

