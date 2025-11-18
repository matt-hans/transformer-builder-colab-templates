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

## Distributed Training (DDP/FSDP)

Distributed training options are exposed via `TrainingConfig` fields and the
CLI JSON configs.

### Config Fields

- `strategy`: Lightning strategy string, e.g. `"auto"`, `"ddp"`, `"fsdp_native"`.
- `devices`: Number of devices (e.g. `2`), `"auto"` for all visible devices, or a list of device IDs.
- `num_nodes`: Number of nodes (default `1`).
- `accumulate_grad_batches`: Gradient accumulation steps; effective batch size is `batch_size * accumulate_grad_batches`.
- `precision`: Precision string passed to Lightning (e.g. `"bf16-mixed"`, `"16-mixed"`, `"32"`).

### Example DDP Config

File: `configs/example_train_ddp.json`

```json
{
  "task_name": "lm_tiny",
  "learning_rate": 5e-5,
  "batch_size": 4,
  "epochs": 1,
  "strategy": "ddp",
  "devices": "auto",
  "num_nodes": 1,
  "precision": "bf16-mixed",
  "accumulate_grad_batches": 2,
  "use_amp": true
}
```

Run:

```bash
python -m cli.run_training --config configs/example_train_ddp.json
```

On single-GPU systems, Lightning will still run but effectively use a single
device. If `pytorch_lightning` is not installed, the CLI falls back to the
adapter-first stub training loop.

### Export Tier (Tier 4)

Tier 4 validates exported models (TorchScript/ONNX) against the PyTorch
reference implementation and reports parity/latency metrics.

1. Create or use the example export config:

```json
{
  "task_name": "lm_tiny",
  "modality": "text",
  "tier": "4",
  "export": {
    "formats": ["torchscript", "onnx", "pytorch"],
    "quantization": null,
    "export_dir": "exports/lm_tiny"
  }
}
```

2. Run the export + validation pipeline:

```bash
python -m cli.run_tiers --config configs/example_tiers_export.json
```

This will:

- Build a `TrainingConfig` and `TaskSpec` for `task_name`.
- Instantiate a stub model (LMStub for text, SimpleCNN for vision) plus the
  appropriate adapter.
- Export the model via `export_model` to the requested formats.
- Run Tier 4 export validation (`run_tier4_export_validation`) and print:
  - Status per format (ok/warn/fail).
  - Max absolute difference and latency in ms.
  - Paths to exported artifacts.

3. JSON output for CI/CD:

```bash
python -m cli.run_tiers --config configs/example_tiers_export.json --json
```

This prints a JSON object containing:

- `export`: mapping of format names to artifact paths.
- `tier4`: structured validation results (status, per-format metrics).

## How to Run Vision Tasks (Tier 1)

Vision tasks use the same CLI entrypoint as text tasks, but with a different
`task_name` and adapter/model wiring under the hood.

1. Ensure you have a working Python environment with `torch` installed.
2. Use the provided example config for the tiny vision preset:

```bash
python -m cli.run_tiers --config configs/example_tiers_vision.json
```

This will:

- Build a `TrainingConfig` with `task_name="vision_tiny"`.
- Construct a `TaskSpec` with `modality="vision"` and image schema
  (e.g., `{"image_size": [3, 32, 32]}`).
- Instantiate a `SimpleCNN` stub model and `VisionClassificationAdapter`.
- Run Tier 1 shape robustness and gradient flow tests via `utils.test_functions`.

You can copy `configs/example_tiers_vision.json` and adjust it for your own
vision tasks (e.g., different `task_name` and `num_classes`) as long as the
corresponding `TaskSpec` and dataset configuration are defined.
