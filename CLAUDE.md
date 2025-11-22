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

### Core Training Concepts

**For detailed training examples, see `NEW_API_QUICK_REFERENCE.md`.**

#### TrainingConfig - Versioned Configuration System

```python
from utils.training.training_config import TrainingConfig
from utils.training.engine.trainer import Trainer

# Create configuration
config = TrainingConfig(
    epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    random_seed=42,
    use_wandb=True,
    wandb_project="my-project"
)

# Train with modern API
trainer = Trainer(model, config, training_config, task_spec, tokenizer=tokenizer)
results = trainer.train(train_data, val_data)
```

**Key Features:**
- Reproducibility via random seeds and deterministic mode
- W&B integration with automatic metrics tracking
- Checkpoint management with configurable frequency
- Export bundles for production deployment
- Validation before training (fail-fast)

**See `NEW_API_QUICK_REFERENCE.md` for:**
- Complete migration guide from legacy API
- TrainingConfigBuilder patterns
- Data module selection
- Hyperparameter search strategies

### Data Quality Validation Architecture (v4.0)

**Three-Layer Validation Strategy** for robust training data quality:

#### Layer 1: Preprocessing Validation (Permissive Warnings - v4.1+)
**Where**: `training.ipynb` Cell 22 (before training starts)
**Purpose**: Provide data quality guidance without blocking training
**Performance**: 1× filter (runs once during preprocessing)
**Philosophy**: Guidance, not gatekeeping - users make final decision

```python
from utils.training.validation import SequenceLengthValidator
from utils.training.data_quality import filter_short_sequences

# Step 1: Validate dataset quality (permissive)
validator = SequenceLengthValidator(
    min_seq_len=2,  # For causal LM
    max_filter_rate=0.20  # Advisory threshold (not blocking)
)

result = validator.validate(train_data)

# v4.1+: Severity-based warnings (not blocking)
if result.severity == 'excellent':
    print(f"✅ {result.message}")
elif result.severity == 'good':
    print(f"ℹ️ {result.message}")
elif result.severity == 'high':
    print(f"⚠️ {result.message}")  # Normal for WikiText (25-40%)
elif result.severity == 'very_high':
    print(f"🔶 {result.message}")  # Review recommended
else:  # critical
    print(f"🚨 {result.message}")
    # Only critical severity requires user confirmation
    user_input = input("Continue? (yes/no): ")
    if user_input.lower() != 'yes':
        raise ValueError("Training aborted due to data quality concerns")

# Step 2: Filter short sequences
train_data = filter_short_sequences(train_data, min_length=2)
val_data = filter_short_sequences(val_data, min_length=2)
```

**Severity Zones (v4.1+):**
- ✅ **Excellent** (0-10%): No warning - excellent data quality
- ℹ️ **Good** (10-20%): Info only - moderate filtering is normal
- ⚠️ **High** (20-40%): Warning - **NORMAL for WikiText-raw** (25-40% expected)
- 🔶 **Very High** (40-60%): Strong warning - review recommended
- 🚨 **Critical** (60-100%): Critical warning - user confirmation required

**Key Features:**
- Statistical sampling (1000 examples) for performance
- Multi-level warnings based on severity
- No false positives blocking valid datasets (WikiText, etc.)
- Dataset-agnostic flexibility for diverse use cases
- Works with HuggingFace datasets, PyTorch datasets, and lists

#### Layer 2: Trainer Validation (Pre-Training)
**Where**: `utils/training/engine/trainer.py:796-889` (`_validate_data_quality()`)
**Purpose**: Verify preprocessing wasn't skipped
**Performance**: Samples first 10 batches only

```python
# Automatic - called by Trainer before training loop
trainer = Trainer(model, config, training_config, task_spec, tokenizer=tokenizer)
results = trainer.train(train_data, val_data)
# Raises ValueError if >1% sequences are too short
```

**Detection Logic:**
- Samples first 10 batches (early detection)
- Strict threshold: >1% short sequences = preprocessing skipped
- Clear error messages guiding users to Layer 1 filtering

#### Layer 3: Collator Safety Net (Runtime)
**Where**: `utils/tokenization/data_collator.py:101-108` (`LanguageModelingDataCollator`)
**Purpose**: Minimal empty batch check only
**Performance**: Negligible overhead

```python
# Automatic - part of DataLoader collation
# Only checks for completely empty batches (should never happen)
```

**Architecture Benefits:**
- **49x speedup**: Preprocessing (1× filter) vs Runtime (12,500× filter)
- **Statistical validity**: Dataset-level thresholds vs unreliable batch-level
- **Single Responsibility**: Collator only batches, doesn't validate
- **Fail-fast**: Errors caught before GPU allocation

**Centralized Constants:**
```python
from utils.training.constants import TASK_MIN_SEQ_LEN

# Task-specific minimum sequence lengths
TASK_MIN_SEQ_LEN = {
    'lm': 2,                    # Causal LM (token shifting requirement)
    'classification': 1,         # Classification (single token valid)
    'vision_classification': 0,  # Vision (no text sequences)
}
```

**See also:**
- `docs/plans/2025-11-21-validation-layer-implementation.md` - Full architecture design
- `tests/validation/test_performance.py` - Performance benchmarks
- `tests/validation/test_integration.py` - Integration tests

### Training Pipeline Features (v3.5, v3.6, v4.0)

**For comprehensive training documentation, see `NEW_API_QUICK_REFERENCE.md`.**

#### Performance Features (v3.5)

1. **torch.compile Integration**: 10-20% speedup with PyTorch 2.0+ compilation
   - Modes: `"default"`, `"reduce-overhead"`, `"max-autotune"`
   - Set via `TrainingConfig(compile_mode="default")`

2. **Gradient Accumulation Tracking**: 75% W&B log reduction
   - Tracks both micro-batch steps and effective optimizer updates
   - Set via `TrainingConfig(gradient_accumulation_steps=4)`

3. **VisionDataCollator**: 2-5% faster vision task batching
   - Automatic selection for `TaskSpec.modality="vision"`
   - ImageNet normalization by default

4. **Production Export Bundles**: Complete deployment artifacts
   - ONNX, TorchScript, Docker, TorchServe configs
   - Set via `TrainingConfig(export_bundle=True, export_formats=["onnx"])`

#### Safety & Visualization Features (v3.6)

1. **Distributed Training Guardrails**: Prevents DDP/FSDP deadlocks in notebooks
   - Automatic detection of Colab/Jupyter environments
   - Forces safe `strategy='auto'` for single-GPU training

2. **Flash Attention Support**: 2-4x attention speedup
   - Automatic for PyTorch 2.0+ with CUDA
   - Uses `torch.nn.functional.scaled_dot_product_attention`

3. **Drift Visualization Dashboard**: 4-panel drift analysis
   - JS distance tracking with threshold zones
   - Distribution histograms, heatmaps, timeseries

#### Data Collation Requirements (v4.0)

**BREAKING CHANGE**: Text tasks now require explicit tokenizer/collator.

```python
from utils.training.engine.trainer import Trainer

# Text tasks: tokenizer required
trainer = Trainer(model, config, training_config, task_spec, tokenizer=tokenizer)

# Vision tasks: no tokenizer needed
trainer = Trainer(model, config, training_config, TaskSpec.vision_tiny())
```

**Why**: Prevents runtime collation errors with variable-length sequences.

**See `NEW_API_QUICK_REFERENCE.md` section "Tokenizer Integration" for:**
- Migration guide from v3.x
- Manual collator override
- Collator priority system
- Common errors and fixes

#### Additional Training Utilities

**Reproducibility**: Fast mode (default, 20% faster) vs Deterministic mode (bit-exact)
```python
config = TrainingConfig(random_seed=42, deterministic=False)  # Fast mode
```

**Metrics Tracking**: W&B integration + local SQLite via ExperimentDB
```python
tracker = MetricsTracker(use_wandb=True)
tracker.log_epoch(epoch, train_metrics, val_metrics)
```

**LR Scheduling**: Linear warmup + cosine decay (automatic)
```python
config = TrainingConfig(warmup_ratio=0.1)  # 10% warmup
```

**Gradient Clipping**: Prevent explosions
```python
config = TrainingConfig(gradient_clip_norm=1.0)
```

**GPU Metrics**: Memory, utilization, temperature (automatic)

**Padding Handling**: Automatic `ignore_index=pad_token_id` in loss calculation

#### Checkpoint Recovery (v4.0+)

**NEW**: Recover training results from saved checkpoints for interrupted training, analysis, or resume workflows.

**Quick Example** (in training.ipynb):
```python
from utils.training.engine.recovery import recover_training_results

# Recover from best checkpoint
results = recover_training_results(checkpoint_dir='./checkpoints')

# Use exactly like Trainer.train() return value
print(f"Train Loss: {results['loss_history'][-1]:.4f}")
print(f"Val Loss: {results['val_loss_history'][-1]:.4f}")
```

**Use Cases**:
1. **Interrupted Training**: Runtime disconnected? Recover your 3-hour training run
2. **Analysis**: Examine training history without re-running training
3. **Resume Training**: Load checkpoint and continue from any epoch
4. **Comparison**: Load multiple checkpoints to compare experiments

**Recovery Cell** (training.ipynb Cell 33):
- Lists all checkpoints with metrics
- Recovers best checkpoint automatically
- Logs to ExperimentDB
- Provides results in same format as `Trainer.train()`

**Load Model Weights Cell** (training.ipynb Cell 34):
- Loads `model_state_dict` from checkpoint into model
- **Intelligent checkpoint discovery**: reuses `ckpt_dir` from Cell 33, or searches common locations
- Checkpoint search paths: `./checkpoints`, `./training_output/checkpoints`, `./tmp_training_output/checkpoints`, `/content/workspace/checkpoints`
- Displays available checkpoints with metrics
- Defaults to Cell 33's checkpoint selection if available
- Auto-selects best checkpoint by val_loss otherwise
- Migrates model to GPU if available
- Sets model to eval() mode (ready for inference)
- Shows comprehensive info (checkpoint metadata, model summary, architecture preview)
- Provides error handling for architecture mismatches
- Lists searched locations in error message for debugging

**Variable Extraction Cell** (training.ipynb Cell 35):
- Extracts `workspace_root`, `run_name`, `metrics_df` from results dict
- Works for both training (Cell 32) and recovery (Cell 33) workflows
- Always prints extraction status for verification

**Display Metrics Table Cell** (training.ipynb Cell 40):
- Displays training metrics summary table
- Exports metrics to CSV: `{workspace_root}/results/{run_name}_metrics.csv`
- Requires `metrics_df` from Cell 32 (Training) or Cell 33 (Recovery)
- Provides fallback values: `run_name='training_run'`, `workspace_root='./workspace'`
- Graceful error handling for missing variables or export failures

**What's Saved in Checkpoints** (automatic):
- Full metrics history (`metrics_tracker.metrics_history`)
- Model state dict
- Optimizer & scheduler states
- RNG states (reproducibility)
- Training configuration
- Git commit hash
- **Session metadata** (v4.0+): `workspace_root`, `run_name` for reliable recovery

**API Reference**:
```python
# Recover from specific checkpoint
results = recover_training_results(
    checkpoint_path='checkpoint_epoch0009_step000009_20251122_065455.pt'
)

# Recover best checkpoint in directory
results = recover_training_results(
    checkpoint_dir='./checkpoints',
    monitor='val_loss',
    mode='min'
)

# List all checkpoints
from utils.training.engine.recovery import list_checkpoints
checkpoints = list_checkpoints('./checkpoints')
for ckpt in checkpoints:
    print(f"Epoch {ckpt['epoch']}: val_loss={ckpt['val_loss']:.4f}")
```

**Return Format** (backward-compatible):
```python
{
    'metrics_summary': pd.DataFrame,  # Modern API: per-epoch metrics
    'best_epoch': int,
    'final_loss': float,
    'checkpoint_path': str,
    'training_time': float,
    'workspace_root': str,            # v4.0+: Base directory for results/checkpoints
    'run_name': str,                  # v4.0+: Unique run identifier
    'loss_history': List[float],      # v3.x compatibility
    'val_loss_history': List[float]   # v3.x compatibility
}
```

**Checkpoint Schema v4.0**:

Starting with v4.0, checkpoints store **session metadata** for reliable recovery:
- `workspace_root`: Base directory for results/exports (extracted from checkpoint, not inferred from filesystem)
- `run_name`: Unique run identifier (extracted from checkpoint, not parsed from filename)

**Recovery Workflow**:
1. Load checkpoint → extract `metrics_history` + `workspace_root` + `run_name` from `custom_state`
2. Compute derived fields → `loss_history`, `best_epoch`
3. Return `results` dict → ready for downstream notebook cells

**Legacy Checkpoint Support**:
- v3.x checkpoints (without session metadata) are still supported
- Recovery falls back to path parsing with a warning message
- Ensures backward compatibility while encouraging v4.0+ format

**Error Handling**:
- Missing checkpoint → Clear FileNotFoundError with path
- No metrics_history → Explains v4.0 requirement
- Corrupted file → Suggests recovery from earlier checkpoint

**See Also**: `tests/training/engine/test_recovery.py` for comprehensive test examples

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

### Data Module Selection Guide

**Two data modules available** (as of v4.0):

1. **`SimpleDataModule`** (`utils/tokenization/data_module.py`):
   - For **pre-tokenized** datasets
   - Requires PyTorch Lightning
   - Lightweight wrapper over DataLoader
   - **Use when**: Data is already tokenized, need simple Lightning integration

2. **`UniversalDataModule`** (`utils/training/engine/data.py`):
   - For **any dataset type** (HuggingFace, PyTorch, List[Tensor])
   - Framework-agnostic (no Lightning requirement)
   - Auto train/val split, reproducibility, collator registry
   - **Use when**: Need full training engine features, reproducibility, flexibility

**Training notebook usage**:
```python
# Cell 8: Import both
from utils.training.engine.data import UniversalDataModule
from utils.tokenization.data_module import SimpleDataModule

# Section 6: Use SimpleDataModule for pre-tokenized data
data_module = SimpleDataModule(
    train_dataset=final_train_data,
    val_dataset=final_val_data,
    task_spec=task_spec,
    batch_size=training_config.batch_size,
    num_workers=2,
    tokenizer=tokenizer
)
```

**Migration path** (optional):
- v4.x: Both data modules supported
- v5.0: `SimpleDataModule` may be deprecated in favor of `UniversalDataModule`

### Training API Architecture

**Two Training APIs Available** (as of v4.0):

1. **Modern API** (`utils/training/engine/trainer.Trainer`):
   - Modular v4.0 engine architecture
   - Framework-agnostic (no Lightning required)
   - Recommended for new code
   - Full control over training loop components

2. **Legacy API** (`utils/training/training_core.TrainingCoordinator`):
   - PyTorch Lightning-based orchestration
   - Available for backward compatibility
   - Used in training.ipynb Section 6
   - Simpler interface for quick prototyping

**Import Requirements**:
```python
# Modern API
from utils.training.engine.trainer import Trainer

# Legacy API
from utils.training.training_core import TrainingCoordinator
```

**Training notebook usage** (Section 6):
```python
# Cell 8: Import both APIs
from utils.training.engine.trainer import Trainer
from utils.training.training_core import TrainingCoordinator

# Section 6: Uses TrainingCoordinator (Lightning-based)
coordinator = TrainingCoordinator(
    output_dir=training_config.checkpoint_dir,
    use_gpu=torch.cuda.is_available(),
    max_epochs=training_config.epochs,
    precision_str='32-true'
)
```

**When to Use**:
- **Trainer**: New projects, production pipelines, framework-agnostic workflows, microservices
- **TrainingCoordinator**: Existing notebooks, Lightning integration, quick prototyping, research

**Migration Path**: v5.0 may deprecate `TrainingCoordinator` in favor of unified `Trainer` API.

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
