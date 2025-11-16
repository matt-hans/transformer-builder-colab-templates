# Environment Snapshot Usage Guide

## Overview

The environment snapshot utility captures your complete Python environment (pip freeze) at training time, enabling exact environment recreation for reproducing results months or years later.

**Problem it solves**: Training succeeds today but fails to reproduce in 6 months because package versions changed (e.g., PyTorch 2.0 → 2.5 breaking changes).

**Value**: Future-proof reproducibility. Anyone can recreate your exact environment and reproduce your results, even years later.

## Quick Start

### 1. Capture Environment at Training Start

Add this at the beginning of your training notebook:

```python
from utils.training.environment_snapshot import (
    capture_environment,
    save_environment_snapshot,
    log_environment_to_wandb
)

# Capture environment
print("📸 Capturing environment snapshot...")
env_info = capture_environment()

# Save locally
req_path, env_path, repro_path = save_environment_snapshot(env_info, "./environment")

# Log to W&B (optional)
import wandb
if wandb.run:
    log_environment_to_wandb(req_path, env_path, repro_path, env_info)
```

This creates 3 files:
- `environment/requirements.txt` - pip freeze format
- `environment/environment.json` - full metadata
- `environment/REPRODUCE.md` - setup instructions

### 2. Reproduce Environment Later

When you need to recreate the environment (e.g., 6 months later):

```bash
# Download environment files from W&B or your saved copy

# Check Python version requirement
cat environment/REPRODUCE.md

# Install exact versions
pip install -r environment/requirements.txt

# Verify
python -c "import torch; print(torch.__version__)"
```

## Complete Integration Example

### Colab Notebook Training Cell

```python
import wandb
from utils.training.seed_manager import set_random_seed
from utils.training.environment_snapshot import (
    capture_environment,
    save_environment_snapshot,
    log_environment_to_wandb
)
from utils.training.metrics_tracker import MetricsTracker

# Initialize W&B
wandb.init(
    project="transformer-training",
    name="my-experiment-v1",
    config={
        "learning_rate": 5e-5,
        "batch_size": 4,
        "n_epochs": 10,
    }
)

# Set random seed for reproducibility
set_random_seed(42, deterministic=True)

# Capture environment BEFORE training starts
print("📸 Capturing environment snapshot...")
env_info = capture_environment()
req_path, env_path, repro_path = save_environment_snapshot(env_info, "./environment")
log_environment_to_wandb(req_path, env_path, repro_path, env_info)

# Now proceed with training
tracker = MetricsTracker(use_wandb=True)

for epoch in range(10):
    # ... training loop ...
    tracker.log_epoch(
        epoch=epoch,
        train_metrics={'loss': train_loss, 'accuracy': train_acc},
        val_metrics={'loss': val_loss, 'accuracy': val_acc},
        learning_rate=current_lr,
        gradient_norm=max_grad,
        epoch_duration=epoch_time
    )

wandb.finish()
```

## Environment Comparison

Compare two training runs to identify version differences:

```python
from utils.training.environment_snapshot import compare_environments

# Compare two runs
diff = compare_environments(
    'run1/environment.json',
    'run2/environment.json'
)

# Output:
# 🔍 Environment Differences:
#
#   Python: 3.10.12 → 3.10.12
#   CUDA: 12.2 → 12.2
#
#   📦 Changed packages (1):
#     - torch: 2.0.1 → 2.1.0
#
#   ➕ Added packages: 1
#     - transformers==4.36.0
```

Access diff programmatically:

```python
# Check if environments differ
if diff['changed']:
    print(f"Version changes: {len(diff['changed'])}")
    for pkg, old_ver, new_ver in diff['changed']:
        print(f"  {pkg}: {old_ver} → {new_ver}")

if diff['python_version_changed']:
    print("⚠️ Python version changed - may affect results")

if diff['cuda_version_changed']:
    print("⚠️ CUDA version changed - may affect numerical precision")
```

## Captured Information

The environment snapshot includes:

### System Info
- Python version (full and X.Y.Z short format)
- Platform (OS, architecture)
- OS name and release

### Package Versions
- Complete pip freeze output
- Parsed package → version mapping
- PyTorch version
- Transformers version (if installed)
- NumPy version (if installed)

### Hardware Info
- CUDA availability
- CUDA version (if GPU available)
- cuDNN version (if GPU available)
- GPU device name (e.g., "Tesla T4")
- GPU count

## W&B Integration

When logged to W&B, environment snapshots are:

1. **Artifacts**: Stored as versioned artifacts with metadata
2. **Config**: Key versions added to run config for easy filtering
3. **Downloadable**: Can download environment files from any run

### Download Environment from W&B

```python
import wandb

# Initialize API
api = wandb.Api()

# Get run
run = api.run("my-entity/my-project/my-run-id")

# Download environment artifact
artifact = run.use_artifact('my-run-environment:latest')
artifact_dir = artifact.download()

# Now you have the environment files
# artifact_dir/requirements.txt
# artifact_dir/environment.json
# artifact_dir/REPRODUCE.md
```

## Best Practices

### 1. Capture at Training Start
Always capture environment BEFORE training starts (not after), to ensure environment matches the actual training conditions.

### 2. Use Deterministic Mode
For publishable results, use deterministic seed management:

```python
from utils.training.seed_manager import set_random_seed

# Deterministic mode (slower but bit-exact reproducibility)
set_random_seed(42, deterministic=True)
```

### 3. Save Multiple Copies
Save environment snapshot to:
- Local disk (`./environment/`)
- W&B artifacts (via `log_environment_to_wandb`)
- Google Drive (for Colab): `save_environment_snapshot(env_info, "/content/drive/MyDrive/experiments/run1/environment")`

### 4. Compare Before Debugging
If results don't reproduce, first compare environments:

```python
diff = compare_environments('old_run/environment.json', 'new_run/environment.json')

if diff['changed']:
    print("⚠️ Environment changed - likely cause of different results")
```

### 5. Document Non-Determinism
Some operations are non-deterministic even with same environment:
- GPU atomicAdd operations (minor floating point differences)
- Multi-threaded data loading (different order)
- Dropout (if seed not set correctly)

Document these in your REPRODUCE.md:
```markdown
## Known Non-Determinism

- Results may vary by ±0.01% due to GPU atomicAdd non-determinism
- Use deterministic=True mode for bit-exact reproduction (20% slower)
```

## Troubleshooting

### Issue: Different Python Version

**Symptom**: Have Python 3.11.x but environment requires 3.10.12

**Solution**:
```bash
# Option 1: Use pyenv to install exact version
pyenv install 3.10.12
pyenv local 3.10.12

# Option 2: Try with your version (may work for minor differences)
pip install -r requirements.txt

# Option 3: Use Docker with exact Python version
docker run -it python:3.10.12 bash
```

### Issue: CUDA Version Mismatch

**Symptom**: Environment has CUDA 12.2 but you have 11.8

**Solution**:
```bash
# Install PyTorch for your CUDA version
# See: https://pytorch.org/get-started/locally/

# For CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Or use CPU-only:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Package Conflicts

**Symptom**: pip install fails with dependency conflicts

**Solution**:
```bash
# Create fresh virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in order (PyTorch first, then others)
pip install -r requirements.txt

# If still fails, try legacy resolver
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

### Issue: Results Still Don't Reproduce

**Checklist**:
1. ✅ Exact same environment (compare with `compare_environments`)
2. ✅ Same random seed (`set_random_seed(42, deterministic=True)`)
3. ✅ Deterministic mode enabled
4. ✅ Same data order (use `seed_worker` in DataLoader)
5. ✅ Same model initialization order
6. ⚠️ GPU differences (T4 vs A100) may have minor numerical differences

If all above pass but results differ:
- Check for date/time dependencies in code
- Check for external API calls (e.g., downloading data)
- Check for file system dependencies (different file order)

## API Reference

### `capture_environment() -> Dict`

Captures complete Python environment snapshot.

**Returns**: Dict with keys:
- `python_version`: Full version string
- `python_version_short`: "X.Y.Z" format
- `platform`: Full platform string
- `pip_freeze`: Raw pip freeze output
- `packages`: Dict mapping package → version
- `torch_version`: PyTorch version
- `cuda_available`: Bool
- `cuda_version`: CUDA version or None
- `gpu_name`: GPU name or None

### `save_environment_snapshot(env_info, output_dir) -> Tuple[str, str, str]`

Saves environment to 3 files.

**Args**:
- `env_info`: Dict from `capture_environment()`
- `output_dir`: Directory to save files (created if missing)

**Returns**: Tuple of (requirements_path, environment_json_path, reproduce_md_path)

### `compare_environments(env1_path, env2_path) -> Dict`

Compares two environment snapshots.

**Args**:
- `env1_path`: Path to first environment.json
- `env2_path`: Path to second environment.json

**Returns**: Dict with keys:
- `added`: List of (package, version) for new packages
- `removed`: List of (package, version) for removed packages
- `changed`: List of (package, old_ver, new_ver) for changed packages
- `python_version_changed`: Bool
- `cuda_version_changed`: Bool

### `log_environment_to_wandb(req_path, env_path, repro_path, env_info)`

Logs environment snapshot to W&B as artifact.

**Args**:
- `req_path`: Path to requirements.txt
- `env_path`: Path to environment.json
- `repro_path`: Path to REPRODUCE.md
- `env_info`: Dict from `capture_environment()`

**Raises**:
- `ImportError`: If wandb not installed
- `RuntimeError`: If no active W&B run

## Example Output Files

### requirements.txt
```
torch==2.1.0+cu121
transformers==4.36.0
numpy==1.24.3
pandas==2.0.3
...
```

### environment.json
```json
{
  "python_version": "3.10.12 (main, Nov 20 2023, 15:14:05) ...",
  "python_version_short": "3.10.12",
  "platform": "Linux-5.15.0-1051-gcp-x86_64-with-glibc2.35",
  "torch_version": "2.1.0+cu121",
  "cuda_version": "12.1",
  "gpu_name": "Tesla T4",
  "packages": {
    "torch": "2.1.0+cu121",
    "transformers": "4.36.0",
    ...
  }
}
```

### REPRODUCE.md
```markdown
# Environment Reproduction Guide

## Quick Setup

\`\`\`bash
python --version  # Should be 3.10.12
pip install -r requirements.txt
\`\`\`

## System Information

- **Python**: 3.10.12
- **Platform**: Linux-5.15.0-1051-gcp-x86_64-with-glibc2.35
- **PyTorch**: 2.1.0+cu121
- **CUDA**: 12.1
- **GPU**: Tesla T4

...
```

## Related Documentation

- [Seed Management Guide](./SEED_MANAGEMENT.md) - Random seed best practices
- [Metrics Tracking Guide](./METRICS_TRACKING.md) - W&B integration
- [Training Reproducibility](./REPRODUCIBILITY.md) - End-to-end reproducibility
