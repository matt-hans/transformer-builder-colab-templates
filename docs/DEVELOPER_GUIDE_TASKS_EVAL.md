# Developer Guide: Tasks, Evaluation, and Adapters

## Add a New Task

1. Add a `TaskSpec` preset in `utils/training/task_spec.py` (`get_default_task_specs`).
2. Extend `build_dataloader` in `utils/training/dataset_utilities.py` to handle your task.
3. Define metrics in `utils/training/eval_runner.py` if needed.

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

