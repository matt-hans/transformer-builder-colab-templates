# Usage Guide: Colab and CLI

## Modes & Presets

- In notebooks: `from utils.ui.presets import build_configs_for_mode`
  - FAST_DEV, STANDARD_EXPERIMENT, ABLATION_SWEEP
  - Returns `(training_cfg, task_spec, eval_cfg)` configured for quick starts

## TrainingConfig Builder Pattern (Recommended)

The **TrainingConfigBuilder** provides a fluent API for creating training configurations with progressive validation and preset templates. This is the recommended way to create configs in modern code.

### Quick Start with Presets

Choose from 5 battle-tested presets:

```python
from utils.training.training_config import TrainingConfigBuilder

# 1. Quick Prototype - Fast iteration, debugging (3 epochs, 12M params)
config = TrainingConfigBuilder.quick_prototype().build()

# 2. Baseline - Standard config, balanced settings (10 epochs, 125M params)
config = TrainingConfigBuilder.baseline().build()

# 3. Production - High quality, reproducible (20 epochs, export enabled)
config = TrainingConfigBuilder.production().build()

# 4. Distributed - Multi-GPU training (DDP/FSDP, 4 GPUs default)
config = TrainingConfigBuilder.distributed().build()

# 5. Low Memory - Colab free tier, small GPUs (2 batch, 8x accumulation)
config = TrainingConfigBuilder.low_memory().build()
```

### Customize Presets

All presets support customization via method chaining:

```python
# Start with baseline, customize for your needs
config = (TrainingConfigBuilder.baseline()
    .with_training(epochs=30, batch_size=8)  # Longer training, bigger batch
    .with_optimizer(gradient_accumulation_steps=4)  # Effective batch = 32
    .with_logging(run_name="extended-baseline-v1", notes="Extended training")
    .build()
)

# Low memory config for even tighter constraints
config = (TrainingConfigBuilder.low_memory()
    .with_training(batch_size=1, max_train_samples=5000)
    .with_optimizer(gradient_accumulation_steps=16)  # Effective batch = 16
    .with_model(max_seq_len=64)  # Shorter sequences
    .build()
)

# Production with custom export
config = (TrainingConfigBuilder.production()
    .with_export(export_formats=["onnx", "torchscript"], export_dir="./model_v1")
    .with_logging(run_name="final-model-v1.0")
    .build()
)
```

### Build from Scratch

Use the fluent API to build custom configurations:

```python
config = (TrainingConfigBuilder()
    # Model architecture
    .with_model(
        model_name="custom-gpt",
        model_type="gpt",
        d_model=512,
        num_layers=6,
        num_heads=8,
        vocab_size=50257
    )
    # Training hyperparameters
    .with_training(
        learning_rate=1e-4,
        batch_size=16,
        epochs=20,
        validation_split=0.1,
        max_train_samples=50000  # Limit for quick experiments
    )
    # Optimizer settings
    .with_optimizer(
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2
    )
    # Hardware configuration
    .with_hardware(
        use_amp=True,
        compile_mode="default",  # 10-15% speedup
        devices=1,
        precision="bf16-mixed"
    )
    # Experiment tracking
    .with_logging(
        wandb_project="transformer-experiments",
        run_name="custom-config-v1",
        notes="Testing custom configuration"
    )
    # Checkpointing
    .with_checkpointing(
        checkpoint_dir="./checkpoints",
        save_every_n_epochs=2,
        keep_best_only=True
    )
    # Reproducibility
    .with_reproducibility(
        random_seed=42,
        deterministic=False  # Fast mode
    )
    .build()  # Creates validated TrainingConfig
)
```

### Progressive Validation

The builder validates parameters as you set them, catching errors early:

```python
# Error caught immediately, not at build()
try:
    config = (TrainingConfigBuilder()
        .with_model(d_model=768, num_heads=5)  # 768 % 5 != 0
    )
except ValueError as e:
    print(f"Error: {e}")  # "d_model (768) must be divisible by num_heads (5)"

# Negative learning rate caught early
try:
    config = (TrainingConfigBuilder()
        .with_training(learning_rate=-0.001)
    )
except ValueError as e:
    print(f"Error: {e}")  # "learning_rate must be positive"
```

### Preset Comparison

| Preset | Epochs | Model Size | Use Case | Runtime |
|--------|--------|------------|----------|---------|
| `quick_prototype()` | 3 | 12M params (6L, 512d) | Debugging, CI/CD | ~5-10 min |
| `baseline()` | 10 | 125M params (12L, 768d) | Standard experiments | ~2-4 hours |
| `production()` | 20 | 125M params (12L, 768d) | Final runs, deployment | ~8-12 hours |
| `distributed()` | 15 | 350M params (24L, 1024d) | Multi-GPU training | Variable |
| `low_memory()` | 10 | 6M params (6L, 384d) | Colab free, small GPUs | ~1-2 hours |

### Method Reference

The builder provides 11 configuration methods:

- `with_model()` - Architecture (d_model, layers, heads, vocab_size)
- `with_training()` - Hyperparameters (learning_rate, batch_size, epochs)
- `with_optimizer()` - Optimizer settings (weight_decay, warmup, grad_accumulation)
- `with_scheduler()` - LR schedule configuration
- `with_hardware()` - GPU, AMP, compilation, distributed strategies
- `with_logging()` - W&B project, run names, notes
- `with_checkpointing()` - Save frequency, best-only mode
- `with_export()` - ONNX/TorchScript export configuration
- `with_reproducibility()` - Random seed, deterministic mode
- `with_dataset()` - Dataset selection, task configuration
- `build()` - Construct final validated TrainingConfig

### Migration from Direct Construction

**Old way (still supported):**
```python
config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    d_model=768,
    num_layers=12,
    num_heads=12,
    use_amp=True,
    compile_mode="default"
)
```

**New way (recommended):**
```python
config = (TrainingConfigBuilder()
    .with_training(learning_rate=5e-5, batch_size=4, epochs=10)
    .with_model(d_model=768, num_layers=12, num_heads=12)
    .with_hardware(use_amp=True, compile_mode="default")
    .build()
)
```

**Benefits:**
- Organized by concern (model, training, hardware)
- Progressive validation (errors caught early)
- Preset templates for common scenarios
- Method chaining for readability
- Immutable builder (thread-safe)

### Examples

**Quick prototyping:**
```python
# Fast iteration on new architecture
config = (TrainingConfigBuilder.quick_prototype()
    .with_model(d_model=256, num_layers=4, num_heads=4)  # Smaller model
    .with_training(epochs=1)  # Just 1 epoch
    .build()
)
```

**Hyperparameter search:**
```python
# Test different learning rates
for lr in [1e-5, 5e-5, 1e-4]:
    config = (TrainingConfigBuilder.baseline()
        .with_training(learning_rate=lr, epochs=5, max_train_samples=10000)
        .with_logging(run_name=f"hp-search-lr{lr}")
        .build()
    )
    results = test_fine_tuning(model, config, n_epochs=config.epochs)
```

**Production deployment:**
```python
# Final model for deployment
config = (TrainingConfigBuilder.production()
    .with_export(
        export_bundle=True,
        export_formats=["onnx", "torchscript"],
        export_dir="./final_model"
    )
    .with_reproducibility(random_seed=42, deterministic=True)
    .with_logging(
        wandb_project="production-models",
        run_name="v1.0-release",
        notes="Final production model for deployment"
    )
    .build()
)
```

**Multi-GPU training:**
```python
# 8 GPU DDP training
config = (TrainingConfigBuilder.distributed()
    .with_hardware(devices=8, strategy="ddp")
    .with_model(d_model=1024, num_layers=24)
    .with_optimizer(gradient_accumulation_steps=4)  # Effective batch = 256
    .build()
)

# WARNING: Use CLI for distributed training, not notebooks:
# python -m cli.run_training --config config.json
```

See `examples/config_builder_demo.py` for comprehensive examples of all features.

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

## Automated Retraining Triggers (Phase 2 - Production Hardening)

The retraining trigger system provides automated detection of when models need retraining based on multiple criteria: data drift, performance degradation, time elapsed, and data volume changes.

### Quick Start

```python
from utils.training.retraining_triggers import RetrainingTriggerManager
from utils.training.engine.metrics import MetricsEngine
from utils.training.experiment_db import ExperimentDB

# Initialize infrastructure
engine = MetricsEngine(use_wandb=False)
db = ExperimentDB('experiments.db')

# Create trigger manager
manager = RetrainingTriggerManager(
    metrics_engine=engine,
    experiment_db=db
)

# Register triggers (balanced configuration)
manager.register_drift_trigger(threshold=0.15, severity='warning')
manager.register_performance_trigger(threshold=0.05, metric_name='val_loss', severity='warning')
manager.register_time_trigger(interval_hours=168, severity='info')  # 1 week
manager.register_data_volume_trigger(threshold_percentage=0.20, severity='info')

# Check if retraining needed
report = manager.check_retraining_needed()

if report.triggered:
    print(f"🔴 Retraining recommended (severity: {report.severity})")
    for rec in report.recommendations:
        print(f"  - {rec}")
else:
    print("🟢 Model health OK - no retraining needed")
```

### Individual Triggers

#### 1. Drift Trigger

Monitors data distribution changes via Jensen-Shannon divergence:

```python
from utils.training.retraining_triggers import DriftTrigger

trigger = DriftTrigger(
    threshold=0.15,           # JS divergence threshold (0-1)
    metric_name='js_divergence',  # Metric from MetricsEngine
    severity='warning',
    name='drift_monitor'
)

result = trigger.evaluate(drift_metrics={'js_divergence': 0.18})
print(f"Triggered: {result.triggered}")  # True (0.18 > 0.15)
```

**Thresholds:**
- Conservative: 0.20 (production systems)
- Balanced: 0.15 (recommended default)
- Aggressive: 0.10 (rapid iteration)

#### 2. Performance Trigger

Detects model performance degradation:

```python
from utils.training.retraining_triggers import PerformanceTrigger

trigger = PerformanceTrigger(
    threshold=0.05,           # 5% degradation
    metric_name='val_loss',
    mode='min',               # 'min' for loss, 'max' for accuracy
    severity='warning'
)

result = trigger.evaluate(
    current_metrics={'val_loss': 0.45},
    baseline_metrics={'val_loss': 0.40}
)
print(f"Triggered: {result.triggered}")  # True (12.5% increase)
```

**Thresholds:**
- Conservative: 10% degradation
- Balanced: 5% degradation
- Aggressive: 3% degradation

#### 3. Time Trigger

Scheduled retraining based on time elapsed:

```python
from utils.training.retraining_triggers import TimeTrigger
from datetime import datetime, timedelta

trigger = TimeTrigger(
    interval_hours=168,  # 1 week
    severity='info'
)

last_training = datetime.now() - timedelta(days=10)
result = trigger.evaluate(
    metadata={'last_training_time': last_training.isoformat()}
)
print(f"Triggered: {result.triggered}")  # True (10 days > 7 days)
```

**Intervals:**
- Conservative: 336 hours (2 weeks)
- Balanced: 168 hours (1 week)
- Aggressive: 48 hours (2 days)

#### 4. Data Volume Trigger

Triggers when sufficient new data collected:

```python
from utils.training.retraining_triggers import DataVolumeTrigger

trigger = DataVolumeTrigger(
    threshold_samples=1000,      # Absolute threshold (OR)
    threshold_percentage=0.20,   # Percentage threshold (OR)
    severity='info'
)

result = trigger.evaluate(
    metadata={'current_count': 6000, 'baseline_count': 5000}
)
print(f"Triggered: {result.triggered}")  # True (1000 new samples, 20% increase)
```

**Thresholds:**
- Conservative: 30% increase
- Balanced: 20% increase
- Aggressive: 10% increase

### Composite Triggers

Combine multiple triggers with AND/OR logic:

```python
from utils.training.retraining_triggers import CompositeTrigger

drift_trigger = DriftTrigger(threshold=0.15)
perf_trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss')
time_trigger = TimeTrigger(interval_hours=168)

# OR logic: Trigger if ANY condition met
composite_or = CompositeTrigger(
    triggers=[drift_trigger, perf_trigger],
    logic='OR',
    name='drift_or_perf'
)

# AND logic: Trigger if ALL conditions met
composite_and = CompositeTrigger(
    triggers=[drift_trigger, perf_trigger],
    logic='AND',
    name='drift_and_perf'
)

# Nested logic: (Drift OR Perf) AND Time
policy = CompositeTrigger(
    triggers=[
        CompositeTrigger([drift_trigger, perf_trigger], logic='OR'),
        time_trigger
    ],
    logic='AND',
    name='complex_policy'
)
```

### Configuration Presets

Use predefined configurations for common scenarios:

```python
from utils.training.retraining_triggers import (
    get_conservative_config,
    get_aggressive_config,
    get_balanced_config
)

# Conservative: High thresholds, infrequent retraining (production)
conservative = get_conservative_config()
# - Drift: 0.20
# - Performance: 10% degradation
# - Time: 2 weeks
# - Data volume: 30% increase

# Balanced: Medium thresholds, weekly retraining (recommended)
balanced = get_balanced_config()
# - Drift: 0.15
# - Performance: 5% degradation
# - Time: 1 week
# - Data volume: 20% increase

# Aggressive: Low thresholds, frequent retraining (rapid iteration)
aggressive = get_aggressive_config()
# - Drift: 0.10
# - Performance: 3% degradation
# - Time: 2 days
# - Data volume: 10% increase

# Apply to manager
manager = RetrainingTriggerManager()
manager.register_drift_trigger(
    threshold=balanced['drift'].threshold,
    severity=balanced['drift'].severity
)
# ... register other triggers
```

### Integration with MetricsEngine

Automatic drift detection and trigger evaluation:

```python
from utils.training.engine.metrics import MetricsEngine
from utils.training.retraining_triggers import RetrainingTriggerManager
from utils.training.drift_metrics import compute_dataset_profile

# Initialize engine and manager
engine = MetricsEngine(
    use_wandb=True,
    drift_threshold_warning=0.1,
    drift_threshold_critical=0.2
)
manager = RetrainingTriggerManager(metrics_engine=engine)
manager.register_drift_trigger(threshold=0.15)

# During training
ref_profile = compute_dataset_profile(train_dataset, task_spec)

for epoch in range(n_epochs):
    # ... training loop ...
    
    # Profile current data
    curr_profile = compute_dataset_profile(val_dataset, task_spec)
    
    # Log epoch with drift detection
    drift_metrics = engine.log_epoch(
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        learning_rate=scheduler.get_last_lr()[0],
        gradient_norm=grad_norm,
        epoch_duration=epoch_time,
        reference_profile=ref_profile,
        current_profile=curr_profile
    )
    
    # Check if retraining needed
    report = manager.evaluate(drift_metrics={'js_divergence': drift_metrics.js_divergence})
    if report.triggered:
        logger.warning(f"⚠️ Retraining trigger fired: {report.severity}")
```

### Integration with ModelRegistry

Automated baseline performance tracking:

```python
from utils.training.model_registry import ModelRegistry
from utils.training.retraining_triggers import RetrainingTriggerManager

# Register model in registry
registry = ModelRegistry('models.db')
model_id = registry.register_model(
    name='gpt-small-v1',
    version='1.0.0',
    checkpoint_path='checkpoints/epoch_10.pt',
    task_type='language_modeling',
    config_hash=ModelRegistry.compute_config_hash(config.to_dict()),
    metrics={'val_loss': 0.40, 'val_accuracy': 0.85}
)

# Promote to production
registry.promote_model(model_id, 'production')

# Later: Check if production model needs retraining
manager = RetrainingTriggerManager(model_registry=registry)
manager.register_performance_trigger(threshold=0.05, metric_name='val_loss')

# Automatic baseline retrieval from registry
report = manager.check_retraining_needed(model_id=model_id)

if report.triggered:
    print("Production model needs retraining!")
    print(f"Recommendations: {report.recommendations}")
```

### Report Generation

Generate human-readable reports:

```python
# Evaluate triggers
report = manager.evaluate(
    drift_metrics={'js_divergence': 0.18},
    current_metrics={'val_loss': 0.45},
    baseline_metrics={'val_loss': 0.40}
)

# Save as JSON (for automation)
report.to_json('retraining_report.json')

# Save as Markdown (for humans)
markdown = report.to_markdown()
with open('retraining_report.md', 'w') as f:
    f.write(markdown)

# Access structured data
print(f"Triggered: {report.triggered}")
print(f"Severity: {report.severity}")
print(f"Timestamp: {report.timestamp}")

for detail in report.trigger_details:
    print(f"\n{detail.trigger_name}:")
    print(f"  Triggered: {detail.triggered}")
    print(f"  Reason: {detail.reason}")
    if detail.threshold and detail.actual_value:
        print(f"  Threshold: {detail.threshold:.4f}")
        print(f"  Actual: {detail.actual_value:.4f}")
```

Example Markdown report:

```markdown
# Retraining Trigger Report

**Status:** 🔴 TRIGGERED
**Severity:** ⚠️ WARNING
**Timestamp:** 2025-11-20T17:41:48.323817

## Trigger Details

### drift_trigger - ✅ Fired

- **Severity:** warning
- **Reason:** Drift detected: js_divergence=0.1800 exceeds threshold 0.1500
- **Threshold:** 0.1500
- **Actual Value:** 0.1800

### performance_trigger - ✅ Fired

- **Severity:** warning
- **Reason:** Performance degradation: val_loss changed by +12.50% (baseline=0.4000, current=0.4500)
- **Threshold:** 0.0500
- **Actual Value:** 0.1250

## Recommendations

1. Data drift detected: Review new data sources and consider retraining with expanded dataset
2. Performance degradation detected: Retrain model with recent data or investigate root cause
```

### Logging to ExperimentDB

Automatic trigger event logging:

```python
from utils.training.experiment_db import ExperimentDB
from utils.training.retraining_triggers import RetrainingTriggerManager

db = ExperimentDB('experiments.db')
manager = RetrainingTriggerManager(experiment_db=db)

manager.register_drift_trigger(threshold=0.15)
manager.register_performance_trigger(threshold=0.05, metric_name='val_loss')

# Evaluate (automatically logs if triggered)
report = manager.evaluate(
    drift_metrics={'js_divergence': 0.18},
    current_metrics={'val_loss': 0.45},
    baseline_metrics={'val_loss': 0.40}
)

# Query trigger history
runs = db.list_runs(limit=10)
trigger_runs = runs[runs['run_name'].str.contains('trigger_event')]
print(f"Trigger events logged: {len(trigger_runs)}")

# Get metrics for specific trigger event
for _, run in trigger_runs.iterrows():
    metrics = db.get_metrics(run['run_id'])
    print(f"\nTrigger event {run['run_id']}:")
    for _, metric in metrics.iterrows():
        print(f"  {metric['metric_name']}: {metric['value']}")
```

### Best Practices

1. **Start with Balanced Configuration**
   ```python
   # Use balanced config for initial deployment
   balanced = get_balanced_config()
   manager.register_drift_trigger(
       threshold=balanced['drift'].threshold,
       severity=balanced['drift'].severity
   )
   ```

2. **Monitor Trigger History**
   ```python
   # Review recent evaluations
   history = manager.get_trigger_history(limit=10)
   triggered_count = sum(1 for h in history if h.triggered)
   print(f"Triggers fired: {triggered_count}/{len(history)}")
   ```

3. **Combine Multiple Triggers**
   ```python
   # Use composite triggers for complex policies
   # Example: Retrain if (drift OR performance) AND time elapsed
   policy = CompositeTrigger(
       triggers=[
           CompositeTrigger([drift_trigger, perf_trigger], logic='OR'),
           time_trigger
       ],
       logic='AND'
   )
   ```

4. **Adjust Thresholds Based on Domain**
   ```python
   # Critical applications: Conservative thresholds
   if domain == 'healthcare':
       drift_threshold = 0.20  # High threshold
       perf_threshold = 0.10   # 10% degradation
   
   # Rapid iteration: Aggressive thresholds
   elif domain == 'development':
       drift_threshold = 0.10  # Low threshold
       perf_threshold = 0.03   # 3% degradation
   ```

5. **Integrate with CI/CD Pipeline**
   ```python
   # Example CI/CD integration
   import sys
   
   report = manager.check_retraining_needed()
   
   if report.triggered and report.severity == 'critical':
       print(f"🚨 Critical trigger: {report.recommendations}")
       sys.exit(1)  # Fail CI/CD pipeline
   elif report.triggered:
       print(f"⚠️ Warning trigger: {report.recommendations}")
       # Log but don't fail
   ```

### Troubleshooting

**Triggers fire too frequently:**
- Increase thresholds (use conservative config)
- Check for data quality issues
- Review drift detection window size
- Consider composite triggers with AND logic

**Triggers never fire:**
- Decrease thresholds (use aggressive config)
- Verify baseline metrics are set correctly
- Check that drift metrics are being computed
- Ensure metadata (timestamps, counts) is provided

**False positives:**
- Review affected features in drift reports
- Adjust threshold for specific metrics
- Use composite triggers to require multiple conditions
- Add time trigger to prevent too-frequent retraining

**Missing drift metrics:**
- Ensure MetricsEngine is initialized with drift detection
- Provide both reference and current profiles to log_epoch()
- Check that dataset profiles are computed correctly
- Verify task_spec.modality matches dataset type

### See Also

- **MetricsEngine**: `/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/utils/training/engine/metrics.py`
- **Drift Metrics**: `/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/utils/training/drift_metrics.py`
- **Model Registry**: `/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/utils/training/model_registry.py`
- **ExperimentDB**: `/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/utils/training/experiment_db.py`
- **Example Demo**: `/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/examples/retraining_trigger_demo.py`
- **Tests**: `/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/tests/training/test_retraining_triggers.py`

