# Platform Architecture Overview (v4.0.0)

## Layers

- Frontend Interfaces
  - `template.ipynb` (verification) and `training.ipynb` (training/eval/sweeps)
  - CLI (`cli/run_tiers.py`, `cli/run_training.py`)

- Core Abstractions
  - `TaskSpec` (task semantics), `EvalConfig` (evaluation config)
  - `TrainingConfig` (hyperparams + metadata)
  - `ModelAdapter` (adapts arbitrary models to task I/O)

- Execution Engine
  - Training loop (Tier 3 utilities) and adapter-first `run_training`
  - `eval_runner.py` (generic evaluation)
  - `sweep_runner.py` (grid sweeps)
  - `experiment_db.py` (SQLite tracking), `metrics_tracker.py`, `dashboard.py`

- Validation Stack
  - Tier 1: shapes/gradients/stability/memory/inference speed
  - Tier 2: attention/attribution/robustness
  - Tier 3: training utilities + light benchmark helpers
  - All parameterized by `(model, adapter, task_spec)`

- Infrastructure & Safety
  - `gist_loader.py` (revision pinning + checksum)
  - `seed_manager.py`, `environment_snapshot.py`

## Data Flow

```
Gist (model/config) → load_gist_model → Tier 1/2/3 validation →
Training (run_training + adapter) → EvalRunner → ExperimentDB + dashboard →
Repro bundle (configs + env + metrics)
```

## Extension Points

- Add a new task: add a `TaskSpec` preset and extend `build_dataloader`.
- Add a new model family: implement a concrete `ModelAdapter`.
- Extend Tier 2 analyses: use adapter.get_attention_maps() or add hooks.

