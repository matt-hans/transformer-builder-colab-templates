# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository provides Colab-ready notebooks and test utilities for validating transformer models exported from [Transformer Builder](https://transformer-builder.com). The main workflow: users build a transformer visually, export to Colab, and the template automatically loads and validates their model through a 3-tier testing suite.

## Requirements Files Strategy

This repository uses a **three-file requirements strategy** to support different use cases:

### 1. `requirements.txt` - Local Development
**Purpose**: Reproducible local development environments with exact version pins.

**Use for**:
- Setting up virtual environments (`python -m venv .venv`)
- Running tests locally (`pytest`)
- Developing `utils/` test functions
- Debugging with exact package versions

**Installation**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2. `requirements-training.txt` - Training Notebook Only
**Purpose**: Exact version pins for `training.ipynb` (Tier 3 Training Utilities).

**Use for**:
- Reproducible fine-tuning experiments in `training.ipynb`
- Hyperparameter search with Optuna
- Metrics tracking with W&B
- Installing training dependencies in fresh Colab runtime

**Installation** (in `training.ipynb`):
```python
# Cell 1 of training.ipynb
!pip install -r requirements-training.txt
```

### 3. `requirements-colab-v3.4.0.txt` - Documentation & Training Notebook Reference
**Purpose**: Documents Colab's zero-installation strategy and provides training dependencies.

**CRITICAL - Two Distinct Sections**:
1. **Template Section (Lines 22-36)**: Documentation ONLY - DO NOT INSTALL
   - `template.ipynb` uses zero-installation strategy (Colab pre-installed packages)
   - Installing packages in template.ipynb causes NumPy corruption
   - This section exists purely for reference/documentation

2. **Training Section (Lines 38-50)**: Install in `training.ipynb` ONLY
   - `training.ipynb` runs in fresh Colab runtime
   - Installs pytorch-lightning, optuna, torchmetrics for Tier 3 tests
   - Safe to install because training notebook uses separate runtime

**Version Strategy**:
- Uses range pins (`>=`) for Colab compatibility (evolving runtime)
- For exact reproducibility, use `requirements-training.txt` (exact pins `==`)
- Documents version deviations and intentional package omissions (see file footer)

### Architecture Decision Rationale

**Why three files?**
1. **Local Dev** (`requirements.txt`): Developers need exact versions for reproducibility
2. **Training** (`requirements-training.txt`): Training experiments need consistent training stack
3. **Template** (`requirements-colab-v3.4.0.txt`): Zero-installation strategy prevents dependency corruption

**Why exact pins (`==`)?**
- Reproducibility across environments
- Prevent transitive dependency conflicts
- Enable precise bug reproduction
- Match tested configurations

## Common Development Commands

### Local Development Setup
```bash
# Recommended: Use requirements.txt for exact versions
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# Alternative: Manual installation (may differ from tested versions)
pip install torch numpy pandas matplotlib seaborn scipy jupyter
```

### Running the Notebook
```bash
# Launch Jupyter to work with template.ipynb
jupyter lab template.ipynb
# OR
jupyter notebook template.ipynb
```

### Using Test Functions Programmatically
```python
from types import SimpleNamespace
from utils.test_functions import test_shape_robustness, test_gradient_flow

# Create a model config
config = SimpleNamespace(vocab_size=50257, max_seq_len=128, max_batch_size=8)

# Run individual tests
model = ...  # your PyTorch nn.Module
results = test_shape_robustness(model, config)
print(results)

# Run all Tier 1 tests
from utils.test_functions import run_all_tier1_tests
run_all_tier1_tests(model, config)
```

### Using TrainingConfig for Reproducible Experiments
```python
from utils.training.training_config import TrainingConfig, compare_configs
from utils.training.seed_manager import set_random_seed

# Create versioned configuration
config = TrainingConfig(
    # Hyperparameters
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,

    # Model architecture
    vocab_size=50257,
    d_model=768,
    num_layers=12,

    # Reproducibility
    random_seed=42,
    deterministic=False,  # Fast mode

    # Experiment tracking
    wandb_project="transformer-training",
    run_name="baseline-exp",
    notes="Baseline configuration"
)

# Validate before training
config.validate()  # Raises ValueError if invalid

# Save for reproducibility (auto-generates timestamped filename)
config_path = config.save()  # config_20250115_143022.json

# Set seed from config
set_random_seed(config.random_seed, config.deterministic)

# Log to W&B
import wandb
wandb.init(project=config.wandb_project, config=config.to_dict())

# Later: Load to reproduce experiment
loaded_config = TrainingConfig.load(config_path)

# Compare configurations
diff = compare_configs(config_v1, config_v2)
# Prints: learning_rate: 5e-5 → 1e-4, batch_size: 4 → 8
```

### Reproducibility: Deterministic vs. Fast Mode

The codebase supports two reproducibility modes with different performance trade-offs:

**Fast Mode (Default)**: `deterministic=False`
- Enables cuDNN benchmark auto-tuning for ~20% speedup
- Seeds all random number generators (Python, NumPy, PyTorch CPU/GPU)
- DataLoader workers seeded for reproducible batch ordering
- May have minor GPU non-determinism (<0.1% variation) from cuDNN algorithms
- **Recommended for**: Iterative development, experimentation, quick prototyping

**Deterministic Mode**: `deterministic=True`
- Fully bit-exact reproducibility across runs
- Disables cuDNN optimizations: `cudnn.deterministic=True`, `cudnn.benchmark=False`
- Enables PyTorch deterministic algorithms
- **Performance impact**: ~5-10% slower training (acceptable for final experiments)
- **Recommended for**: Publication results, debugging, A/B testing

**Usage Example:**
```python
from utils.training.seed_manager import set_random_seed
from utils.tier3_training_utilities import test_fine_tuning

# Fast mode for development (default)
set_random_seed(42, deterministic=False)
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    random_seed=42,
    deterministic=False  # Fast mode
)

# Deterministic mode for reproducible experiments
set_random_seed(42, deterministic=True)
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    random_seed=42,
    deterministic=True  # Bit-exact reproducibility
)

# Verify reproducibility: run twice with same seed
losses_run1 = results['loss_history']
results2 = test_fine_tuning(model, config, n_epochs=10, random_seed=42, deterministic=True)
losses_run2 = results2['loss_history']
assert losses_run1 == losses_run2  # Bit-identical in deterministic mode
```

**What Gets Seeded:**
1. **Python random module**: `random.seed(seed)`
2. **NumPy RNG**: `np.random.seed(seed)`
3. **PyTorch CPU**: `torch.manual_seed(seed)`
4. **PyTorch GPU**: `torch.cuda.manual_seed_all(seed)`
5. **DataLoader workers**: Each worker seeded via `worker_init_fn=seed_worker`
6. **DataLoader shuffling**: Seeded generator ensures reproducible batch order

**Performance Comparison:**
```python
import time
from utils.training.training_config import TrainingConfig

# Fast mode benchmark
config_fast = TrainingConfig(random_seed=42, deterministic=False)
set_random_seed(config_fast.random_seed, config_fast.deterministic)
start = time.time()
results_fast = test_fine_tuning(model, config, n_epochs=5, deterministic=False)
fast_time = time.time() - start

# Deterministic mode benchmark
config_det = TrainingConfig(random_seed=42, deterministic=True)
set_random_seed(config_det.random_seed, config_det.deterministic)
start = time.time()
results_det = test_fine_tuning(model, config, n_epochs=5, deterministic=True)
det_time = time.time() - start

print(f"Fast mode: {fast_time:.1f}s")
print(f"Deterministic mode: {det_time:.1f}s")
print(f"Slowdown: {(det_time / fast_time - 1) * 100:.1f}%")
# Expected: ~5-10% slower in deterministic mode
```

**Best Practices:**
- **Development**: Use `deterministic=False` for 100s of experiments (20% faster)
- **Final experiments**: Use `deterministic=True` for publication-ready results
- **Debugging**: Use `deterministic=True` to ensure bugs are reproducible
- **A/B testing**: Use `deterministic=True` to isolate changes from randomness
- **Colab timeout**: If hitting 12-hour limit, use `deterministic=False` to save time

**Limitations:**
- Deterministic mode covers 99% of PyTorch operations
- Some exotic operations (e.g., scatter_add on GPU) may still have minor non-determinism
- Multi-GPU distributed training may have edge cases even in deterministic mode
- See [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html) for details

### Using MetricsTracker for Training with W&B
```python
from utils.training.metrics_tracker import MetricsTracker
from utils.tier3_training_utilities import test_fine_tuning

# Initialize W&B (optional)
import wandb
wandb.init(project="transformer-training", name="my-experiment")

# Run training with metrics tracking
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    use_wandb=True  # Log to W&B
)

# Access metrics summary
df = results['metrics_summary']
print(df[['epoch', 'train/loss', 'val/loss', 'val/perplexity']])

# Get best epoch for early stopping
best_epoch = results['best_epoch']
print(f"Best model at epoch {best_epoch}")

# Or use MetricsTracker standalone
tracker = MetricsTracker(use_wandb=True)

# In your training loop
for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = train_batch(model, batch, optimizer)

        # Log per-batch metrics (NEW in v3.4.0)
        global_step = epoch * len(dataloader) + batch_idx
        tracker.log_scalar('train/batch_loss', loss.item(), step=global_step)
        tracker.log_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step=global_step)
        tracker.log_scalar('train/gradient_norm', grad_norm, step=global_step)

    # Log per-epoch metrics (existing)
    tracker.log_epoch(
        epoch=epoch,
        train_metrics={'loss': train_loss, 'accuracy': train_acc},
        val_metrics={'loss': val_loss, 'accuracy': val_acc},
        learning_rate=current_lr,
        gradient_norm=max_grad,
        epoch_duration=epoch_time
    )

# Export metrics for analysis
summary_df = tracker.get_summary()  # Epoch-level metrics
step_df = tracker.get_step_metrics()  # Per-batch metrics (NEW)

summary_df.to_csv('training_metrics.csv', index=False)
step_df.to_csv('batch_metrics.csv', index=False)
```

### Using ExperimentDB for Local Experiment Tracking

Track experiments locally with SQLite as a lightweight alternative to W&B (or use both for redundancy).

```python
from utils.training.experiment_db import ExperimentDB
from utils.training.training_config import TrainingConfig

# Initialize local database
db = ExperimentDB('experiments.db')

# Create run
config = TrainingConfig(learning_rate=5e-5, batch_size=4, epochs=10)
run_id = db.log_run('baseline-v1', config.to_dict(), notes='Initial baseline')

# Training loop with dual logging (W&B + SQLite)
for epoch in range(10):
    train_loss = train_epoch(model, dataloader)
    val_loss = validate(model, val_dataloader)

    # Log to SQLite
    db.log_metric(run_id, 'train/loss', train_loss, epoch=epoch)
    db.log_metric(run_id, 'val/loss', val_loss, epoch=epoch)

    # Log per-batch metrics (optional)
    for step, batch_loss in enumerate(batch_losses):
        global_step = epoch * len(dataloader) + step
        db.log_metric(run_id, 'train/batch_loss', batch_loss, step=global_step, epoch=epoch)

# Log artifacts
db.log_artifact(run_id, 'checkpoint', 'checkpoints/best.pt',
                metadata={'epoch': 5, 'val_loss': 0.38})

# Mark run complete
db.update_run_status(run_id, 'completed')

# Compare multiple runs
comparison = db.compare_runs([1, 2, 3])
print(comparison[['run_name', 'best_val_loss', 'best_epoch']])

# Find best run
best = db.get_best_run('val/loss', mode='min')
print(f"Best: {best['run_name']} (loss={best['best_value']:.4f} at epoch {best['best_epoch']})")

# Query metrics
metrics = db.get_metrics(run_id, 'train/loss')
print(metrics[['epoch', 'value']])

# Export for analysis
import pandas as pd
all_runs = db.list_runs(limit=10)
all_runs.to_csv('experiment_summary.csv', index=False)
```

**Key Features:**
- **Zero dependencies**: Uses built-in SQLite (no internet required)
- **Dual logging**: Works alongside W&B for redundancy
- **Epoch + step metrics**: Matches MetricsTracker granularity
- **Artifact tracking**: Store checkpoint paths with metadata
- **SQL queries**: Direct database access for complex analysis
- **Portable**: Single `.db` file, easy to backup/share

**Example: Hyperparameter Search with ExperimentDB**
```python
from utils.training.experiment_db import ExperimentDB

db = ExperimentDB('hyperparam_search.db')

for lr in [1e-5, 5e-5, 1e-4]:
    for bs in [4, 8, 16]:
        config = TrainingConfig(learning_rate=lr, batch_size=bs)
        run_id = db.log_run(f'lr{lr}_bs{bs}', config.to_dict())

        # Train and log
        results = test_fine_tuning(model, config, n_epochs=5)
        for epoch, loss in enumerate(results['loss_history']):
            db.log_metric(run_id, 'val/loss', loss, epoch=epoch)

        db.update_run_status(run_id, 'completed')

# Find best hyperparameters
best = db.get_best_run('val/loss', mode='min')
print(f"Best config: {best['config']}")
```

### Learning Rate Warmup + Cosine Decay

Enable industry-standard LR scheduling with linear warmup (10% steps) followed by cosine decay to 0.

```python
from utils.tier3_training_utilities import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    learning_rate=5e-5,
    use_lr_schedule=True,   # default True
    use_wandb=True
)

# LR is logged each epoch as 'train/learning_rate' in W&B and summary
```

### Gradient Clipping

Prevent gradient explosions by clipping gradients to a maximum norm.

```python
from utils.tier3_training_utilities import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=5,
    learning_rate=5e-5,
    gradient_clip_norm=1.0  # default 1.0; set None to disable
)

# Logs (per epoch):
#  - gradients/pre_clip_norm
#  - gradients/post_clip_norm
```

### Logging Best Practices

Use Python's `logging` instead of `print()` for production-friendly diagnostics.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Optional file output
        # logging.FileHandler('training.log')
    ]
)

from utils.tier3_training_utilities import test_fine_tuning
results = test_fine_tuning(model, config, n_epochs=3)

# To increase verbosity during debugging
logging.getLogger('utils').setLevel(logging.DEBUG)
```

### Static Type Checking (mypy)

Run mypy to validate type hints and catch issues early:

```bash
mypy utils/ --config-file mypy.ini
```

### GPU Metrics Tracking

Track GPU memory, utilization, and temperature during training.

- Memory: `gpu/memory_allocated_mb`, `gpu/memory_reserved_mb` (always when CUDA is available)
- Utilization & Temperature: `gpu/utilization_percent`, `gpu/temperature_celsius` (requires `pynvml` or `nvidia-smi`)

Install optional dependency in Colab for full metrics:

```python
!pip install pynvml
```

Metrics appear in W&B under the `gpu/` namespace and are logged once per epoch.

The config (`mypy.ini`) enables strict checks while ignoring missing stubs for heavy third‑party libs (torch, transformers, datasets). Public functions in `utils/` include type hints to improve IDE support and reliability.

### Padding Token Handling in Training

**All training functions (`test_fine_tuning`, `test_hyperparameter_search`) automatically exclude padding tokens from loss calculation.** This ensures accurate metrics and prevents the model from learning to predict padding.

**How it works:**
1. **Automatic Detection**: `pad_token_id` is detected from `config.pad_token_id` or `config.tokenizer.pad_token_id`
2. **Fallback**: Defaults to `pad_token_id=0` if not found (with warning)
3. **Loss Masking**: All `F.cross_entropy` calls use `ignore_index=pad_token_id`
4. **Consistent Application**: Applied to both training and validation loops

**Example with custom padding:**
```python
from types import SimpleNamespace
from utils.tier3_training_utilities import test_fine_tuning

# Config with custom pad_token_id (e.g., GPT-2 uses EOS token as padding)
config = SimpleNamespace(
    vocab_size=50257,
    max_seq_len=128,
    pad_token_id=50256  # GPT-2 EOS token
)

# Training automatically uses ignore_index=50256
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    batch_size=4
)

# Loss and perplexity exclude padding tokens
print(f"Final loss (excl. padding): {results['final_loss']:.4f}")
print(f"Perplexity: {results['metrics_summary']['val/perplexity'].iloc[-1]:.2f}")
```

**Expected behavior:**
- With padding: Loss values are ~20-40% lower than without masking (padding excluded)
- Without padding attribute: Warning logged: "⚠️  No pad_token_id found in config/tokenizer, defaulting to 0"
- Perplexity correctly computed as `exp(masked_loss)`

**Why this matters:**
- **Correct metrics**: Loss/perplexity reflect actual language modeling performance, not padding prediction
- **Training efficiency**: Gradients focus on real tokens, not wasted capacity on padding
- **Baseline compatibility**: Matches HuggingFace transformers' default behavior

## Architecture & Code Organization

### Three-Tier Testing Architecture

The codebase implements a progressive testing suite with increasing complexity:

1. **Tier 1: Critical Validation** (`utils/tier1_critical_validation.py`)
   - Fast (~1 minute), mandatory tests that verify core functionality
   - Tests: shape robustness, gradient flow, numerical stability, parameter initialization, memory profiling, inference speed
   - Must pass before proceeding to advanced analysis

2. **Tier 2: Advanced Analysis** (`utils/tier2_advanced_analysis.py`)
   - Moderate-time (~4 minutes) diagnostic tests for model behavior
   - Tests: attention pattern analysis, feature attribution (Integrated Gradients), input perturbation sensitivity
   - Optional but recommended for understanding model internals

3. **Tier 3: Training Utilities** (`utils/tier3_training_utilities.py`)
   - Time-intensive (5-120 minutes) training and optimization tests
   - Tests: fine-tuning loop with metrics tracking, hyperparameter search (Optuna), GLUE benchmarks
   - Includes `MetricsTracker` for comprehensive W&B logging (loss, perplexity, accuracy, LR, gradients, GPU metrics)
   - For production training workflows

### Module Facade Pattern

`utils/test_functions.py` serves as a **unified import facade** that re-exports all tier functions for backward compatibility. New code should prefer direct imports from tier modules:

```python
# Legacy (still works)
from test_functions import test_shape_robustness

# Preferred for clarity
from tier1_critical_validation import test_shape_robustness
from tier2_advanced_analysis import test_attention_patterns
from tier3_training_utilities import test_fine_tuning
```

### Architecture-Agnostic Design

All test functions use helper utilities that detect model characteristics at runtime:

- **`_detect_vocab_size()`**: Introspects model/config to find vocabulary size (checks config.vocab_size → embedding layers → default 50257)
- **`_safe_get_model_output()` / `_extract_output_tensor()`**: Handle multiple output formats (raw tensors, tuples, dicts, HuggingFace ModelOutput objects)
- **`_get_device()`**: Detects if model is on GPU/CPU to ensure test inputs match

This design allows tests to work with:
- Custom architectures from Transformer Builder
- Standard HuggingFace models
- Arbitrary PyTorch nn.Module subclasses

### Notebook Structure (`template.ipynb`)

The Colab notebook follows a strict cell organization pattern:

1. **Dependency Installation** (Cells 1-2): Install PyTorch, transformers, captum, optuna
2. **Model Loading** (Cells 3-10): URL-based Gist loading with fallback to example model
3. **Code Display** (Cells 5-6): Show loaded model code for transparency
4. **Dynamic Dependency Detection** (Cells 7-8): Parse imports and auto-install missing packages
5. **Model Instantiation** (Cells 9-10): Load config, instantiate model, move to GPU
6. **Tier 1 Tests** (Cells 11-13): Critical validation with detailed output
7. **Tier 2 Tests** (Cells 14-15): Advanced analysis (optional)
8. **Tier 3 Tests** (Cells 16-17): Training utilities (optional, compute-intensive)

Key architectural patterns:
- **Idempotent cells**: Each cell can be re-run without side effects
- **Progressive disclosure**: Tests organized by complexity, skippable sections clearly marked
- **Graceful degradation**: Missing dependencies or unsupported architectures fall back with warnings

### URL-Based Model Loading

The notebook uses a sophisticated URL parameter extraction system:

1. **Primary**: JavaScript reads URL hash from parent frame (`window.parent.location.href`)
2. **Fallback**: Reads `document.referrer` or `document.baseURI`
3. **Override**: Environment variable `GIST_ID`
4. **Final fallback**: Colab form inputs (`@param` cells)

Expected URL format: `https://colab.research.google.com/...#gist_id=abc123&name=CustomTransformer`

The notebook fetches `model.py` and `config.json` from the Gist and validates before execution.

## Test Function Return Conventions

All test functions follow consistent patterns:

- **Return type**: `pandas.DataFrame` or `dict` with structured results
- **Side effects**: Print diagnostics but return data for programmatic use
- **Device handling**: Automatically move test inputs to model's device
- **Error handling**: Catch common failures (wrong input shapes, missing attention weights) and return graceful error messages

Example return structure:
```python
# test_shape_robustness returns DataFrame:
#    Input Shape    Output Shape    Status
# 0  (1, 8)         (1, 8, 50257)   ✅ Pass
# 1  (4, 32)        (4, 32, 50257)  ✅ Pass

# test_gradient_flow returns dict:
# {
#     'max_gradient': 0.0234,
#     'min_gradient': 0.0001,
#     'has_vanishing_gradients': False,
#     'has_exploding_gradients': False
# }
```

## Important Constraints & Gotchas

### Model Assumptions
- Models must be PyTorch `nn.Module` subclasses
- Models should accept `input_ids` as primary input (can be positional or keyword arg)
- Output can be tensor, tuple, dict, or HuggingFace ModelOutput (tests handle all formats)

### Vocabulary Size Detection Priority
1. `config.vocab_size` (explicit attribute)
2. First `nn.Embedding` layer found in model
3. Default to 50257 (GPT-2 tokenizer size)

Always verify vocab_size in config when working with custom tokenizers.

### Attention Pattern Analysis
- Only works if model has attention mechanism with weights accessible via `.attn_weights` or similar attributes
- Tests gracefully skip if attention weights cannot be extracted
- For custom attention, ensure weights are stored and accessible during forward pass

### Memory & GPU Considerations
- Tests automatically detect and use GPU if available via `torch.cuda.is_available()`
- Memory footprint tests scale batch size/sequence length to measure memory growth
- Large models (>1B parameters) may OOM on Colab free tier during Tier 3 tests

## Coding Conventions

Following conventions from AGENTS.md:

- **Style**: PEP 8, 4-space indentation, type hints where practical
- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes
- **Test functions**: Prefix with `test_*`, accept `(model, config)` parameters
- **Commits**: Use Conventional Commits format (`feat:`, `fix:`, `chore:`)

## Security Notes

- The template fetches arbitrary code from GitHub Gists—review before execution
- **Never commit config_*.json files**—they may contain API keys (auto-ignored via .gitignore)
- Use environment variables for credentials in production: `os.getenv('WANDB_API_KEY')`
- For offline/airgapped environments, copy `utils/test_functions.py` locally instead of downloading from remote URLs

### Pre-commit Secret Scanning Hook (Task T050)

Add a lightweight pre-commit hook that blocks commits when common secrets are detected in staged files.

Setup (one-time per clone):
```bash
# From repository root
cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

What it detects:
- `WANDB_API_KEY=...`, `hf_...` (Hugging Face), `sk-...` (OpenAI), `ghp_...` (GitHub), and AWS secret keys

Behavior:
- Blocks commit and prints remediation steps when a secret is found
- For exceptional cases, you may bypass once with `git commit --no-verify` (use sparingly)

Notes:
- Hook is versioned at `.github/hooks/pre-commit`; Git does not track `.git/hooks` so collaborators must copy it locally
- Designed to be portable (Bash), no external dependencies
