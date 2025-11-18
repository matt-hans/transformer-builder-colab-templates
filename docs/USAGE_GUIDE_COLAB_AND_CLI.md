# Usage Guide: Colab and CLI

## Modes & Presets

- In notebooks: `from utils.ui.presets import build_configs_for_mode`
  - FAST_DEV, STANDARD_EXPERIMENT, ABLATION_SWEEP
  - Returns `(training_cfg, task_spec, eval_cfg)` configured for quick starts

## Adapter-First Training + Tiny Eval

- In `training.ipynb`, use the provided cell:
  - Builds `TrainingConfig`, `TaskSpec`, `EvalConfig`
  - Selects `DecoderOnlyLMAdapter` (choose others as needed)
  - Calls `run_training(model, adapter, training_cfg, task_spec, eval_cfg)`
  - Prints `results['eval_summary']`

## Sweeps

- Use `utils/training/sweep_runner.py:run_grid_sweep` with `ExperimentDB`.
- Log runs with `sweep_id` and `sweep_params` for reproducibility.
- See the notebook sweep example cell.

## Repro Bundles

- Use `create_repro_bundle(run_id, training_cfg, task_spec, eval_cfg, env_snapshot, db, dashboard_paths, output_dir)`.
- Produces a zip with configs, env, metrics, and dashboards.

## Gist Loading

- Use `utils.adapters.gist_loader.load_gist_model(gist_id, revision)`.
- Shows owner, files and checksum. Dynamically import `model.py` when present.
- Log `gist_id`, `revision` and `sha256` to `ExperimentDB` for reproducibility.

## CLI

- Run tiers:
  - `python -m cli.run_tiers --config configs/example_tiers.json`
- Run training:
  - `python -m cli.run_training --config configs/example_train.json`
- Config JSON shape (example):

```
{
  "task_name": "lm_tiny",
  "epochs": 1,
  "batch_size": 2,
  "vocab_size": 101,
  "max_seq_len": 16,
  "learning_rate": 0.0005,
  "model_file": "./path/to/model.py",  // or: "gist_id": "...", "gist_revision": "..."
  "eval": {"dataset_id": "lm_tiny_v1", "batch_size": 2},
  "log_to_db": true,
  "run_name": "cli-run-01"
}
```

- The CLI reuses the same internal APIs as notebooks and supports loading `model.py` from a local path or a fetched gist.

