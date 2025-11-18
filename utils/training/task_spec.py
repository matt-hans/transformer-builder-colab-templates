"""
Task and evaluation task specification utilities.

Defines a lightweight, serializable TaskSpec that describes the semantics of a
training/evaluation task independently of any specific model implementation.

This enables architecture-agnostic training/evaluation by pairing TaskSpec with
an appropriate ModelAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Literal, TypedDict, cast


TaskModality = Literal["text", "vision", "audio", "tabular"]
TaskType = Literal[
    "lm",
    "classification",  # kept for backwards compatibility
    "seq2seq",
    "text_classification",
    "vision_classification",
    "vision_multilabel",
]


class TaskSchemaDict(TypedDict, total=False):
    """
    Typed mapping used for input/output schema dictionaries.

    This is intentionally loose and only constrains common keys used by
    built-in presets; user code is free to add additional keys.
    """

    # Text tasks
    max_seq_len: int
    vocab_size: int

    # Vision tasks
    image_size: List[int]
    channels_first: bool
    num_classes: int


def _empty_schema() -> "TaskSchemaDict":
    """Return an empty schema mapping for TaskSpec input/output schemas."""
    return {}


@dataclass
class TaskSpec:
    """
    Describes a task's semantics and expected model I/O.

    Attributes:
        name:
            Human-friendly preset name (e.g., "lm_tiny", "cls_tiny").

        task_type:
            High-level task type. For text tasks this is typically one of
            {"lm", "classification", "seq2seq"}; for multimodal extensions it
            will use more explicit values such as "text_classification",
            "vision_classification", or "vision_multilabel".

        model_family:
            Model family the task expects
            ("decoder_only", "encoder_only", "encoder_decoder").

        input_fields:
            Names of input fields a batch will provide
            (e.g., ["input_ids", "attention_mask"] or ["pixel_values"]).

        target_field:
            Name of the target field in the batch (e.g., "labels").
            None if not applicable.

        loss_type:
            Primary loss to optimize (e.g., "cross_entropy", "mse").

        metrics:
            List of metric identifiers to compute
            (e.g., ["loss", "perplexity", "accuracy"]).

        special_tokens:
            Mapping of token-role to token IDs (if any) used by the task
            (e.g., {"pad_token_id": 0}).

        additional_config:
            Freeform extra config used by adapters/datasets
            (small, JSON-serializable values only).

        modality:
            High-level data modality for the task. Defaults to "text".
            Other supported values are "vision", "audio", and "tabular".

        input_schema:
            Free-form schema describing expected model inputs for this task.
            For example, a vision classification task might specify:
                {"image_size": [3, 224, 224], "channels_first": True}
            while a language modeling task might specify:
                {"max_seq_len": 128, "vocab_size": 50257}

        output_schema:
            Schema describing model outputs/targets. For example:
                {"num_classes": 10}  # classification
                {"vocab_size": 50257}  # language modeling

        preprocessing_config:
            Optional configuration for preprocessing/augmentation. The exact
            structure is left to higher-level code (e.g., dataset utilities)
            but typical keys include "normalize", "mean", "std", "augmentations".
    """

    name: str
    task_type: TaskType
    model_family: str
    input_fields: List[str]
    target_field: Optional[str]
    loss_type: str
    metrics: List[str]
    special_tokens: Dict[str, int] = field(default_factory=dict)
    additional_config: Dict[str, Any] = field(default_factory=dict)

    # Multimodal extensions (MM-01)
    modality: TaskModality = "text"
    input_schema: TaskSchemaDict = field(default_factory=_empty_schema)
    output_schema: TaskSchemaDict = field(default_factory=_empty_schema)
    preprocessing_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TaskSpec":
        """
        Deserialize from a dict (raises KeyError/TypeError on invalid input).

        The loader is tolerant of older configs that do not include modality
        or schema fields and will default them appropriately.
        """
        # Backwards compatibility: allow legacy "classification" task type
        raw_task_type = data["task_type"]
        if raw_task_type == "classification":
            task_type: TaskType = "classification"
        else:
            task_type = cast(TaskType, raw_task_type)

        modality = cast(TaskModality, data.get("modality", "text"))

        input_schema = cast(TaskSchemaDict, data.get("input_schema") or {})
        output_schema = cast(TaskSchemaDict, data.get("output_schema") or {})
        preprocessing_config = data.get("preprocessing_config")

        return TaskSpec(
            name=data["name"],
            task_type=task_type,
            model_family=data["model_family"],
            input_fields=list(data.get("input_fields", [])),
            target_field=data.get("target_field"),
            loss_type=data["loss_type"],
            metrics=list(data.get("metrics", [])),
            special_tokens=dict(data.get("special_tokens", {})),
            additional_config=dict(data.get("additional_config", {})),
            modality=modality,
            input_schema=input_schema,
            output_schema=output_schema,
            preprocessing_config=preprocessing_config,
        )

    # ------------------------------------------------------------------
    # Convenience helpers for downstream code (modality-aware queries)
    # ------------------------------------------------------------------

    @property
    def task_name(self) -> str:
        """
        Alias for the task preset name.

        Some documentation refers to this field as ``task_name``; the
        underlying storage is ``name`` for backwards compatibility.
        """
        return self.name

    def is_text(self) -> bool:
        """Return True if this is a text task."""
        return self.modality == "text"

    def is_vision(self) -> bool:
        """Return True if this is a vision task."""
        return self.modality == "vision"

    def is_audio(self) -> bool:
        """Return True if this is an audio task."""
        return self.modality == "audio"

    def is_tabular(self) -> bool:
        """Return True if this is a tabular task."""
        return self.modality == "tabular"

    def get_input_shape(self) -> Optional[List[int]]:
        """
        Return a best-effort static input shape description for the task.

        For text tasks this typically returns [max_seq_len]; for vision
        tasks it returns the image_size field if present.
        """
        # Vision: use explicit image_size when available
        image_size = self.input_schema.get("image_size")
        if isinstance(image_size, list) and all(isinstance(d, int) for d in image_size):
            return image_size

        # Text: approximate using max_seq_len when available
        max_seq_len = self.input_schema.get("max_seq_len")
        if isinstance(max_seq_len, int):
            return [max_seq_len]

        return None


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
            modality="text",
            input_schema={"max_seq_len": 128, "vocab_size": 50257},
            output_schema={"vocab_size": 50257},
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
            modality="text",
            input_schema={"max_seq_len": 128, "vocab_size": 50257},
            output_schema={"num_classes": 2},
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
            modality="text",
            input_schema={"max_seq_len": 128, "vocab_size": 50257},
            output_schema={"vocab_size": 50257},
        ),
        # Vision Classification (encoder-only, tiny example)
        "vision_tiny": TaskSpec(
            name="vision_tiny",
            task_type="vision_classification",
            model_family="encoder_only",
            input_fields=["pixel_values"],
            target_field="labels",
            loss_type="cross_entropy",
            metrics=["loss", "accuracy"],
            special_tokens={},
            additional_config={"num_classes": 4},
            modality="vision",
            input_schema={"image_size": [3, 32, 32], "channels_first": True},
            output_schema={"num_classes": 4},
        ),
    }


__all__ = [
    "TaskSpec",
    "get_default_task_specs",
    "load_task_spec_from_dict",
    "TaskModality",
    "TaskType",
]
