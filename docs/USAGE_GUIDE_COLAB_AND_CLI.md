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

## Export Health Checks

The export bundle system includes comprehensive health checks to ensure production readiness. Health checks validate models before, during, and after export to multiple formats.

### Overview

Health checks are automatically run when creating an export bundle via `create_export_bundle()` (enabled by default). The health checker performs three stages of validation:

1. **Pre-Export Checks**: Validate model architecture, parameters, memory requirements, and forward pass
2. **Format-Specific Validation**: Verify ONNX, TorchScript, and PyTorch exports
3. **Post-Export Verification**: Check numerical consistency and performance benchmarks

### Basic Usage

```python
from utils.training.export_utilities import create_export_bundle
from utils.training.training_config import TrainingConfig

# Create export bundle with health checks (default)
export_dir = create_export_bundle(
    model=trained_model,
    config=model_config,
    task_spec=task_spec,
    training_config=TrainingConfig(
        export_bundle=True,
        export_formats=["onnx", "torchscript", "pytorch"]
    )
)

# Health reports are automatically saved to:
# - export_dir/artifacts/health_report.json (structured data)
# - export_dir/health_report.md (human-readable report)
```

### Health Check Categories

#### 1. Pre-Export Checks

**Architecture Validation**:
- Counts and validates model layers
- Detects BatchNorm, Dropout, and other layer types
- Verifies module hierarchy

**Parameter Validation**:
- Checks for NaN/Inf in model parameters
- Computes parameter statistics (mean, std, min, max)
- Reports total and trainable parameter counts

**Input/Output Shape Validation**:
- Validates forward pass with dummy inputs
- Verifies output shapes match expected schema
- Tests multiple batch sizes

**Memory Requirements**:
- Estimates parameter memory footprint
- Measures peak memory usage during inference
- Warns if memory requirements are excessive (>1GB)

**Forward Pass Validation**:
- Tests model with various batch sizes
- Detects NaN/Inf in outputs
- Validates model can handle different input shapes

#### 2. Format-Specific Validation

**ONNX Validation**:
- Verifies ONNX model structure with `onnx.checker`
- Reports opset version and IR version
- Counts model operations
- Checks file size and integrity

**TorchScript Validation**:
- Loads and validates TorchScript model
- Tests forward pass with dummy input
- Verifies serialization/deserialization
- Checks for NaN/Inf in outputs

**PyTorch Validation**:
- Validates state dict completeness
- Checks for NaN/Inf in saved tensors
- Verifies parameter count matches original model
- Reports file size

#### 3. Post-Export Verification

**Numerical Consistency**:
- Compares ONNX outputs with PyTorch (max error threshold: 1e-4)
- Compares TorchScript outputs with PyTorch (max error threshold: 1e-6)
- Reports absolute and relative errors
- Fails if numerical differences exceed thresholds

**Performance Benchmarking**:
- Measures inference latency for PyTorch, ONNX, TorchScript
- Computes speedup ratios (e.g., ONNX is 2.3x faster)
- Runs 50 inference iterations for stable measurements
- Reports mean inference time in milliseconds

### Health Report Format

#### JSON Report Structure

```json
{
  "timestamp": "2025-01-20T14:30:00",
  "model_name": "my-transformer",
  "summary": {
    "total": 12,
    "passed": 10,
    "warnings": 1,
    "failed": 1
  },
  "health_score": 87.5,
  "all_passed": false,
  "checks": [
    {
      "check_name": "architecture_validation",
      "status": "passed",
      "message": "Architecture validated: 42 modules",
      "details": {
        "total_modules": 42,
        "layer_counts": {"Linear": 8, "Embedding": 2, ...}
      },
      "duration_seconds": 0.123
    },
    {
      "check_name": "parameter_validation",
      "status": "failed",
      "message": "Found 2 NaN parameters",
      "details": {
        "total_params": 1250000,
        "nan_params": 2,
        "inf_params": 0
      },
      "duration_seconds": 0.456
    }
  ],
  "recommendations": [
    "Fix NaN parameters: Check training stability and gradient clipping",
    "Review all warnings before production deployment"
  ]
}
```

#### Markdown Report

The Markdown report includes:
- **Summary**: Pass/fail counts and health score
- **Failed Checks**: Detailed error information for failures
- **Warnings**: Non-critical issues requiring attention
- **All Checks**: Tabular summary of all validations
- **Recommendations**: Actionable next steps

### Health Score Calculation

The health score (0-100) is computed as:
- **Passed checks**: Full credit (1.0 weight)
- **Warnings**: Partial credit (0.5 weight)
- **Failed checks**: No credit (0.0 weight)

Formula: `score = (passed + 0.5 * warnings) / total * 100`

Example:
- 10 passed, 2 warnings, 1 failed → Score: `(10 + 0.5*2) / 13 * 100 = 84.6/100`

### Programmatic Usage

#### Run Health Checks Separately

```python
from utils.training.export_health import ExportHealthChecker

# Create health checker
checker = ExportHealthChecker(
    model=trained_model,
    config=model_config,
    task_spec=task_spec
)

# Run only pre-export checks
report = checker.run_all_checks()

# Access results
print(f"Health Score: {report.health_score}/100")
print(f"All Passed: {report.all_passed}")

for check in report.get_failed_checks():
    print(f"Failed: {check.check_name} - {check.message}")

# Run full checks including post-export verification
full_report = checker.run_all_checks(
    export_dir=Path("exports/model_001"),
    formats=["onnx", "torchscript"]
)

# Save reports
full_report.save_json("health_report.json")
full_report.save_markdown("health_report.md")
```

#### Disable Health Checks

```python
# Skip health checks for faster exports
export_dir = create_export_bundle(
    model=trained_model,
    config=model_config,
    task_spec=task_spec,
    training_config=training_config,
    run_health_checks=False  # Disable health validation
)
```

#### Custom Health Check Logic

```python
from utils.training.export_health import CheckResult, ExportHealthReport

# Create custom check
def custom_check(model, config):
    # Your validation logic
    if some_condition:
        return CheckResult(
            check_name="custom_validation",
            status="passed",
            message="Custom check passed",
            details={"metric": 0.95}
        )
    else:
        return CheckResult(
            check_name="custom_validation",
            status="failed",
            message="Custom check failed",
            details={"error": "Some error"}
        )

# Add to report
report = ExportHealthReport(
    timestamp=datetime.now().isoformat(),
    model_name="my-model"
)
report.add_check(custom_check(model, config))
```

### Production Deployment Guidelines

#### Critical Checks (Must Pass)

Before production deployment, ensure these checks **pass**:
1. **Parameter Validation**: No NaN/Inf in model weights
2. **Forward Pass Validation**: Model produces valid outputs
3. **Numerical Consistency**: Export formats match PyTorch outputs
4. **Format Validation**: All export formats load successfully

#### Warnings (Review Required)

Warnings may be acceptable depending on use case:
- **Large Memory Footprint**: Consider quantization or model compression
- **Moderate Numerical Differences**: Acceptable for some applications (e.g., ONNX may have slight differences due to operator implementations)

#### Failed Checks Remediation

**NaN Parameters**:
```python
# Check training stability
# - Reduce learning rate
# - Enable gradient clipping
# - Check for exploding gradients

from utils.training.training_config import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-5,  # Reduce LR
    max_grad_norm=1.0,   # Enable gradient clipping
    use_amp=True         # Use mixed precision
)
```

**Numerical Inconsistency**:
```python
# For ONNX exports, ensure proper precision and opset
from utils.training.export_utilities import ONNXExporter

exporter = ONNXExporter(
    opset_version=16,     # Use newer opset
    optimize=True,        # Enable optimizations
    validate=True         # Validate outputs
)
```

**Memory Issues**:
```python
# Apply quantization to reduce memory footprint
from utils.training.export_utilities import export_model

paths = export_model(
    model=model,
    adapter=adapter,
    task_spec=task_spec,
    export_dir="exports/quantized",
    formats=["torchscript", "onnx", "pytorch"],
    quantization="dynamic"  # Apply dynamic quantization
)
```

### CI/CD Integration

Integrate health checks into your CI/CD pipeline:

```yaml
# .github/workflows/model_export.yml
name: Model Export and Validation

on:
  push:
    branches: [main]

jobs:
  export-and-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Export model with health checks
        run: |
          python - << 'PY'
          from utils.training.export_utilities import create_export_bundle

          export_dir = create_export_bundle(
              model=trained_model,
              config=model_config,
              task_spec=task_spec,
              training_config=training_config,
              run_health_checks=True
          )

          # Load health report
          import json
          with open(export_dir / "artifacts" / "health_report.json") as f:
              report = json.load(f)

          # Fail CI if health checks failed
          if not report["all_passed"]:
              print("Health checks failed!")
              print(f"Health Score: {report['health_score']}/100")
              for check in report["checks"]:
                  if check["status"] == "failed":
                      print(f"Failed: {check['check_name']} - {check['message']}")
              exit(1)

          print(f"Health checks passed! Score: {report['health_score']}/100")
          PY
      - name: Upload health report
        uses: actions/upload-artifact@v3
        with:
          name: health-report
          path: exports/*/health_report.md
```

### Best Practices

1. **Always run health checks** before production deployment
2. **Archive health reports** with exported models for audit trails
3. **Set thresholds** for acceptable warnings in your deployment pipeline
4. **Monitor health scores** over time to detect model degradation
5. **Review recommendations** in health reports for optimization opportunities
6. **Test on target hardware** - health checks use current device, may differ from production
7. **Validate multiple formats** - ONNX and TorchScript may have different behaviors
8. **Document exceptions** - if deploying with warnings, document why they're acceptable

### Troubleshooting

**Health checks fail with "torch not available"**:
- Ensure PyTorch is installed: `pip install torch>=2.0`
- Use virtual environment with proper dependencies

**ONNX validation shows "module not found"**:
- Install ONNX dependencies: `pip install onnx onnxruntime onnxscript`
- Or disable ONNX export in `export_formats` list

**Numerical consistency fails for ONNX**:
- Some operators may have slight differences between PyTorch and ONNX
- Check opset version compatibility
- Review operator implementations in ONNX Runtime docs
- Consider increasing tolerance threshold if differences are acceptable

**Performance benchmark shows no speedup**:
- ONNX/TorchScript speedup varies by model architecture
- Smaller models may not benefit from optimization
- Try different ONNX optimization levels
- Test on target deployment hardware (CPU vs GPU)

**Memory requirements estimation fails**:
- Requires CUDA for accurate GPU memory profiling
- CPU estimates are approximate (parameter memory × 2)
- Test on actual deployment hardware for accurate measurements
