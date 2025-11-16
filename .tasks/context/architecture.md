# Technical Architecture

## Tech Stack

**Language**: Python 3.10+
**Target Environment**: Google Colab (12GB GPU, 12-hour timeout, ephemeral storage)

**Core Dependencies**:
- PyTorch 2.6+ (pre-installed in Colab)
- NumPy 2.3.4 (pre-installed, corruption prevention via zero-installation strategy)
- Pandas, Matplotlib, Seaborn, SciPy (pre-installed)

**Training Dependencies** (training.ipynb only):
- pytorch-lightning >= 2.4.0 (training framework)
- optuna >= 3.0.0 (hyperparameter optimization)
- torchmetrics >= 1.3.0 (metrics computation)

**MLOps Stack** (to be added):
- wandb (experiment tracking)
- huggingface_hub (model registry)
- datasets (real data integration)

## System Architecture

### Two-Notebook Strategy (v3.4.0)

**template.ipynb** (Validation):
- Zero pip installs to prevent NumPy corruption
- Tier 1 + 2 tests (~4 minutes total)
- Fast feedback on model correctness

**training.ipynb** (Training):
- Fresh runtime with training dependencies
- Tier 3 utilities (fine-tuning, hyperparameter search)
- Isolated environment prevents dependency conflicts

### Three-Tier Testing Framework

**Tier 1: Critical Validation** (`utils/tier1_critical_validation.py`)
- Shape robustness, gradient flow, numerical stability
- Parameter initialization, memory profiling, inference speed
- ~1 minute, mandatory before training

**Tier 2: Advanced Analysis** (`utils/tier2_advanced_analysis.py`)
- Attention pattern analysis, feature attribution
- Input perturbation sensitivity
- ~3 minutes, optional diagnostics

**Tier 3: Training Utilities** (`utils/tier3_training_utilities.py`)
- Fine-tuning loop, hyperparameter search, benchmarking
- 10-120 minutes, production training workflows

### Module Organization

```
utils/
├── test_functions.py          # Unified facade (backward compatibility)
├── tier1_critical_validation.py
├── tier2_advanced_analysis.py
├── tier3_training_utilities.py
├── adapters/                  # Architecture-agnostic model introspection
│   ├── model_adapter.py       # Detect vocab_size, model_type, output format
│   └── __init__.py
├── tokenization/              # Tokenizer utilities
│   ├── adaptive_tokenizer.py  # Auto-select tokenizer for vocab_size
│   ├── bpe_trainer.py         # Custom BPE training
│   └── data_module.py         # DataLoader with collation
├── training/                  # Training infrastructure
│   ├── training_core.py       # Main training loop
│   ├── checkpoint_manager.py  # Google Drive checkpointing
│   ├── dataset_utilities.py   # HF datasets integration
│   └── export_utilities.py    # Multi-format model export
└── ui/                        # Setup wizards
    ├── setup_wizard.py        # Interactive training config
    └── presets.py             # Common hyperparameter presets
```

## Design Patterns

**Architecture-Agnostic Design**: All utilities detect model characteristics at runtime via introspection (_detect_vocab_size, _safe_get_model_output, _detect_model_type).

**Module Facade Pattern**: test_functions.py re-exports all tier functions for backward compatibility while allowing direct tier imports.

**Progressive Disclosure**: Notebooks organized by complexity with clear skip markers for advanced sections.

**Graceful Degradation**: Missing dependencies or unsupported architectures fall back with warnings rather than errors.

## Critical Paths

1. **Model Loading**: URL param extraction → Gist fetch → Code validation → Instantiation
2. **Validation**: Tier 1 tests → Optional Tier 2 → Pass/fail report
3. **Training**: Data loading → Checkpointing setup → Training loop → Export
4. **Session Recovery**: Detect checkpoint → Restore model/optimizer → Resume training
