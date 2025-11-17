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
    # ... training code ...
    tracker.log_epoch(
        epoch=epoch,
        train_metrics={'loss': train_loss, 'accuracy': train_acc},
        val_metrics={'loss': val_loss, 'accuracy': val_acc},
        learning_rate=current_lr,
        gradient_norm=max_grad,
        epoch_duration=epoch_time
    )

# Export metrics for analysis
summary_df = tracker.get_summary()
summary_df.to_csv('training_metrics.csv', index=False)
```

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
