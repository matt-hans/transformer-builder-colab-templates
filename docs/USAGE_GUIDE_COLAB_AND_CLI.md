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
- **Security Warning**: Any external `model.py` (local file or GitHub gist) is
  executed as Python code. The CLI performs a simple static scan to refuse
  obviously dangerous patterns (`os.system`, `subprocess.Popen`, etc.), but you
  **must still review and trust the code** before running it, especially in
  shared or production environments.

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

### Strategies

- **`auto`**:
  - Default and safest option.
  - Works on CPU, single-GPU, and multi-GPU nodes.
  - Lets Lightning pick the right accelerator/strategy.
- **`ddp`**:
  - Data-parallel training across multiple GPUs on a node.
  - Recommended for 2–8 GPUs when your model fits on a single device.
- **`fsdp_native`**:
  - Fully Sharded Data Parallel for very large models.
  - Requires recent PyTorch/Lightning and high-memory GPUs (e.g., A100/H100).

### Config Fields

- `strategy`: Lightning strategy string, as above.
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

For multi-GPU setups, you can also launch via `torchrun` to ensure one process
per GPU:

```bash
torchrun --nproc_per_node=2 -m cli.run_training --config configs/example_train_ddp.json
```

On single-GPU systems, Lightning will still run but effectively use a single
device. If `pytorch_lightning` is not installed, the CLI falls back to the
adapter-first stub training loop.

### Resuming from a Checkpoint

You can resume training from a Lightning checkpoint by specifying
`resume_from_checkpoint` in your training config:

```json
{
  "task_name": "lm_tiny",
  "learning_rate": 5e-5,
  "batch_size": 4,
  "epochs": 5,
  "strategy": "ddp",
  "devices": "auto",
  "resume_from_checkpoint": "training_output/checkpoints/cli-run/epoch=02-val_loss=0.1234.ckpt"
}
```

The CLI will pass this to `TrainingCoordinator`, which in turn passes it to
Lightning’s `Trainer.fit(..., ckpt_path=...)` so that model, optimizer, and
RNG state are restored and training continues from the next epoch.

### Hardware Notes & Safe Defaults

- **Colab Free / Single-GPU**:
  - Use `strategy="auto"`, `devices=1` or omit `devices` and let it default.
  - Keep `precision="16-mixed"` or `"bf16-mixed"` if your GPU supports it.
- **Local Multi-GPU Workstation (2–4 GPUs)**:
  - Use `strategy="ddp"`, `devices=2`/`4` or `"auto"`.
  - Start with `precision="bf16-mixed"` on Ampere+ GPUs, otherwise `"16-mixed"`.
- **Very Large Models / FSDP**:
  - Consider `strategy="fsdp_native"` only on capable hardware (A100/H100).
  - Begin with small batch sizes and enable gradient accumulation.

The coordinator includes guardrails:

- If `strategy="ddp"` but only one device is effectively requested or visible,
  it logs a warning and falls back to `strategy="auto"` (single-device).
- If `strategy="fsdp_native"` is requested without a multi-GPU CUDA setup,
  it logs a warning that training may fail and suggests `ddp`/`auto`.

### Troubleshooting

- **Error: "DDP requires multiple processes/devices"**
  - Check that `devices` is >1 (or a list with length >1) and that
    `torch.cuda.device_count() >= devices`.
  - On Colab Free (single GPU), prefer `strategy="auto"` or `devices=1`.

- **FSDP out-of-memory (OOM)**
  - Reduce `batch_size` and increase `accumulate_grad_batches`.
  - Consider `strategy="ddp"` if the model fits in a single-device memory.

- **Training runs on CPU unexpectedly**
  - Check `use_gpu=True` in your config or coordinator.
  - Confirm that `torch.cuda.is_available()` returns `True` inside your env.

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

## Tier 5 Monitoring & Drift

Tier 5 combines three checks into a single command:

- Evaluation of the current model on a held-out eval set
- Optional baseline vs candidate comparison (regression testing)
- Optional input/output drift analysis relative to a stored reference profile

### CLI: Tier 5 Monitoring

1. Use the example monitoring config:

File: `configs/example_tiers_monitoring.json`

```json
{
  "task_name": "lm_tiny",
  "modality": "text",
  "tier": "5",
  "baseline_run_id": null,
  "reference_profile_id": null,
  "db_path": "experiments.db",
  "eval": {
    "dataset_id": "lm_tiny_v1",
    "split": "validation",
    "max_eval_examples": 32,
    "batch_size": 4
  }
}
```

2. Run Tier 5 from the CLI:

```bash
python -m cli.run_tiers --config configs/example_tiers_monitoring.json --json
```

This will:

- Build a `TrainingConfig` and `TaskSpec` for `task_name`
- Instantiate a stub model (LMStub or SimpleCNN) plus adapter
- Evaluate the model on the specified eval set
- Optionally compare to a baseline run (if `baseline_run_id` is set)
- Optionally compute drift metrics (if `reference_profile_id` points to a run with a stored profile)

The JSON output contains:

- `eval_metrics`: aggregated metrics for the candidate model
- `comparison`: regression comparison (if baseline provided)
- `drift`: drift analysis (if reference profile provided)
- `status`: `"ok"`, `"warn"`, or `"fail"` for CI/CD gates

### CI/CD Example (GitHub Actions)

You can wire Tier 5 into CI to block regressions and severe drift:

```yaml
jobs:
  tier5-monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: |
          python -m venv .venv
          . .venv/bin/activate
          pip install -U pip
          pip install -r requirements.txt
      - name: Run Tier 5 monitoring
        run: |
          . .venv/bin/activate
          python -m cli.run_tiers --config configs/example_tiers_monitoring.json --json > tier5.json
          python - << 'PY'
          import json
          with open("tier5.json") as f:
              data = json.load(f)
          status = data.get("status", "fail")
          if status == "fail":
              raise SystemExit("Tier 5 monitoring failed (regression or drift detected).")
          PY
```

For more advanced workflows, you can generate the Tier 5 config dynamically
to compare the latest experiment against a stored “production” run and adjust
the thresholds inside your model-regression/drift logic accordingly.

### Using ExperimentDB Profiles

To enable drift detection, first log a reference profile for a run using `log_profile_to_db` from `utils.training.drift_metrics`, then supply its `run_id` as `reference_profile_id` in the Tier 5 config.
