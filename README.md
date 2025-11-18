# Transformer Builder - Colab Testing Templates

Advanced testing and training infrastructure for transformer models built with [Transformer Builder](https://transformer-builder.com).

## Quick Start (v3.4.0)

### Step 1: Model Validation
1. Build a transformer in [Transformer Builder](https://transformer-builder.com)
2. Click "Open in Colab" in the export panel
3. The notebook automatically loads your model and runs validation tests

**Zero installation required** - uses only pre-installed Colab packages!

### Step 2: Training (Optional)
1. Open `training.ipynb` in Colab
2. Restart runtime (Runtime → Restart runtime)
3. Paste your same Gist ID
4. Run training and optimization tests

**Why two notebooks?** Training dependencies (pytorch-lightning, optuna) require NumPy version changes. Separating them prevents dependency conflicts and keeps validation fast.

## What's Included

### 📓 template.ipynb - Tier 1 & 2 Tests

#### Tier 1: Critical Validation (~1 minute)
- ✅ Multi-input shape verification across edge cases
- ✅ Gradient flow analysis (detect vanishing/exploding gradients)
- ✅ Numerical stability checks (NaN/Inf detection)
- ✅ Parameter initialization validation
- ✅ Memory footprint profiling
- ✅ Inference speed benchmarks

#### Tier 2: Advanced Analysis (~3 minutes)
- 🔬 Attention pattern analysis (multi-head attention support)
- 🔬 Robustness testing under input perturbations

### 📓 training.ipynb - Tier 3 Training

#### Tier 3: Training & Fine-Tuning (10-20 minutes)
- 🚀 Fine-tuning loop with loss tracking
- 🚀 Hyperparameter optimization using Optuna
- 🚀 Benchmark comparison against baselines

## Repository Structure

```
transformer-builder-colab-templates/
├── template.ipynb                 # Testing & validation (Tier 1 + 2)
├── training.ipynb                 # Training utilities (Tier 3) + modes/sweeps
├── cli/                           # CLI entrypoints (run_tiers, run_training)
│   ├── __init__.py
│   ├── run_tiers.py
│   └── run_training.py
├── docs/                          # Platform docs (v4.0.0)
│   ├── ARCHITECTURE_OVERVIEW_v4.0.0.md
│   ├── USAGE_GUIDE_COLAB_AND_CLI.md
│   └── DEVELOPER_GUIDE_TASKS_EVAL.md
├── examples/
│   └── datasets/                  # Tiny datasets for quick eval
│       ├── lm_tiny.txt
│       ├── cls_tiny.csv
│       └── seq2seq_tiny.jsonl
├── utils/
│   ├── test_functions.py          # Unified test facade
│   ├── tier1_critical_validation.py
│   ├── tier2_advanced_analysis.py
│   ├── tier3_training_utilities.py
│   ├── adapters/                  # Model introspection + ModelAdapter + gist_loader
│   ├── tokenization/              # BPE training & validation
│   ├── training/                  # Dataset, checkpoints, eval_runner, export, sweeps, ExperimentDB
│   └── ui/                        # Setup wizard & mode presets
├── requirements-colab.txt         # Dependency documentation
└── README.md
```

## Manual Usage

If you have model code outside Transformer Builder:

1. Open `template.ipynb` in Colab
2. Modify Cell 3 to include your model code
3. Update config in Cell 4
4. Run all cells

## Requirements

- Google account (Colab free tier is sufficient)
- Generated model must be a PyTorch `nn.Module`

## Examples

See `examples/` directory for pre-populated notebooks demonstrating common architectures.

## Docs (v4.0.0)

- Architecture overview: `docs/ARCHITECTURE_OVERVIEW_v4.0.0.md`
- Usage guide (Colab + CLI): `docs/USAGE_GUIDE_COLAB_AND_CLI.md`
- Developer guide (Tasks/Adapters/Eval): `docs/DEVELOPER_GUIDE_TASKS_EVAL.md`

## CLI Quick Start

Run quick validation (Tier 1) with a tiny stub model:

```
python -m cli.run_tiers --config configs/example_tiers.json  # optional config
```

Run training + tiny evaluation:

```
python -m cli.run_training --config configs/example_train.json
```

Example training config JSON:

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

Notes:
- `model_file` can be a directory (containing `model.py`) or a file path; the CLI tries `build_model()` then `Model` class.
- If `gist_id` is provided, the CLI fetches the gist (best effort in restricted environments) and tries to import `model.py`.
- Without a model provided, the CLI uses a tiny LM stub with the requested `vocab_size`.

## Support

Issues? Report at [transformer-builder/issues](https://github.com/your-org/transformer-builder/issues)

## License

MIT License - see LICENSE file
