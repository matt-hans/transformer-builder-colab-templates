# Developer Guide: Tasks, Evaluation, and Adapters

## Add a New Task

`TaskSpec` is the single source of truth for task semantics across the training
stack. It now supports multiple modalities via a small set of fields:

- `name` / `task_name`: human-friendly preset identifier (e.g. `"lm_tiny"`).
- `task_type`: high-level task type (e.g. `"lm"`, `"classification"`,
  `"seq2seq"`, `"vision_classification"`).
- `modality`: `"text"`, `"vision"`, `"audio"`, or `"tabular"`.
- `input_fields`: names of batch fields provided to the adapter/model.
- `target_field`: target field in the batch (usually `"labels"`).
- `input_schema`: dictionary describing input shapes/properties.
- `output_schema`: dictionary describing output shapes/properties.
- `preprocessing_config`: optional preprocessing/augmentation config.

To add a new task:

1. Add a `TaskSpec` preset in `utils/training/task_spec.py` (`get_default_task_specs`).
2. Extend `build_dataloader` in `utils/training/dataset_utilities.py` to handle your task.
3. Define metrics in `utils/training/eval_runner.py` if needed.

### Text Task Example (Language Modeling)

```python
from utils.training.task_spec import TaskSpec

lm_task = TaskSpec(
    name="lm_custom",
    task_type="lm",
    model_family="decoder_only",
    input_fields=["input_ids", "attention_mask"],
    target_field="labels",
    loss_type="cross_entropy",
    metrics=["loss", "perplexity"],
    modality="text",
    input_schema={"max_seq_len": 256, "vocab_size": 50257},
    output_schema={"vocab_size": 50257},
)
```

### Vision Task Example (Classification)

```python
from utils.training.task_spec import TaskSpec

vision_task = TaskSpec(
    name="vision_tiny",
    task_type="vision_classification",
    model_family="encoder_only",
    input_fields=["pixel_values"],
    target_field="labels",
    loss_type="cross_entropy",
    metrics=["loss", "accuracy"],
    modality="vision",
    input_schema={"image_size": [3, 64, 64], "channels_first": True},
    output_schema={"num_classes": 10},
    preprocessing_config={
        "normalize": True,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
)
```

Downstream components (datasets, adapters, evaluation, export) can use these
fields to dynamically configure preprocessing, shapes, and metrics without
hard-coding modality-specific logic.

## Implement a New ModelAdapter

- Create a concrete adapter in `utils/adapters/model_adapter.py` implementing:
  - `prepare_inputs`, `forward_for_loss`, `get_logits`, `predict`, and optionally `get_attention_maps`.
- Use the adapter across Tier 1/2/3 by passing `(model, adapter, task_spec)`.

## Extend Tier 2 Analyses

- If your model exposes attention maps, return them from `adapter.get_attention_maps`.
- For custom analyses, add hooks in `utils/tier2_advanced_analysis.py`.

## Evaluation & Metrics

- Use `utils/training/eval_runner.py:run_evaluation` for generic eval logic.
- Log to `MetricsTracker` when available; store to `ExperimentDB` if orchestrated externally.
