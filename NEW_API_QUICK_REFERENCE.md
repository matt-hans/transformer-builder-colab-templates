# New Training API - Quick Reference Guide

**Updated:** 2025-11-20
**For:** training.ipynb v3.6.1

---

## Overview

The training notebook now uses a **modular engine API** instead of the legacy function-based approach. This guide helps you migrate your code and understand the new patterns.

---

## Quick Comparison

### Training

#### Old API (Removed)
```python
from utils.tier3_training_utilities import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    use_wandb=True
)
```

#### New API
```python
from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig

# 1. Create config
training_config = TrainingConfig(
    epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    wandb_project="my-project",
    use_wandb=True
)

# 2. Create trainer
trainer = Trainer(
    model=model,
    config=config,  # model architecture config
    training_config=training_config,
    task_spec=task_spec
)

# 3. Train
results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset
)
```

### Hyperparameter Search

#### Old API (Removed)
```python
from utils.tier3_training_utilities import test_hyperparameter_search

hp_results = test_hyperparameter_search(
    model=model,
    config=config,
    train_data=train_data,
    val_data=val_data,
    n_trials=10,
    timeout=3600
)
```

#### New API (Job Queue Pattern)
```python
from utils.training.training_config import TrainingConfigBuilder

# Define search space
search_space = {
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'batch_size': [4, 8, 16]
}

# Generate trials
import itertools
trials = list(itertools.product(*search_space.values()))

# Run each trial
for trial_params in trials:
    # Build config for this trial
    trial_config = (
        TrainingConfigBuilder.from_config(base_config)
        .with_training(
            learning_rate=trial_params[0],
            batch_size=trial_params[1]
        )
        .build()
    )

    # Create trainer
    trainer = Trainer(model, config, trial_config, task_spec)

    # Train
    results = trainer.train(train_data, val_data)
```

#### Recommended for Production
```bash
# Use CLI for parallel search
python cli/run_training.py --sweep-config sweep.yaml

# Or use external tools
# - Optuna: Bayesian optimization
# - Ray Tune: Distributed parallel search
# - W&B Sweeps: Cloud-based with agents
```

---

## Common Patterns

### 1. Basic Training
```python
# Minimal setup
from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec

# Config
config = TrainingConfig(
    epochs=10,
    learning_rate=5e-5,
    batch_size=4
)

# Task
task_spec = TaskSpec.language_modeling(
    name="gpt-training",
    vocab_size=50257,
    max_seq_len=128
)

# Train
trainer = Trainer(model, model_config, config, task_spec)
results = trainer.train(train_data, val_data)
```

### 2. With W&B Logging
```python
config = TrainingConfig(
    epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    use_wandb=True,
    wandb_project="my-project",
    run_name="baseline-v1"
)

trainer = Trainer(model, model_config, config, task_spec)
results = trainer.train(train_data, val_data)
```

### 3. With Checkpointing
```python
config = TrainingConfig(
    epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    checkpoint_dir="./checkpoints",
    checkpoint_frequency=2  # Save every 2 epochs
)

trainer = Trainer(model, model_config, config, task_spec)
results = trainer.train(train_data, val_data)

# Checkpoints saved to:
# ./checkpoints/epoch_2.pt
# ./checkpoints/epoch_4.pt
# etc.
```

### 4. With All v3.5/v3.6 Features
```python
config = TrainingConfig(
    # Training
    epochs=10,
    learning_rate=5e-5,
    batch_size=4,

    # v3.5: torch.compile (10-20% speedup)
    compile_mode="default",

    # v3.5: Gradient accumulation
    gradient_accumulation_steps=4,

    # v3.5: Export bundle
    export_bundle=True,
    export_formats=["onnx", "torchscript"],

    # Logging
    use_wandb=True,
    wandb_project="my-project",

    # Checkpointing
    checkpoint_dir="./checkpoints"
)

trainer = Trainer(model, model_config, config, task_spec)
results = trainer.train(train_data, val_data)

# v3.6: Flash Attention automatically enabled (2-4x attention speedup)
# v3.6: Distributed guardrails active in notebooks
```

### 5. Using TrainingConfigBuilder
```python
from utils.training.training_config import TrainingConfigBuilder

# Start with preset
config = (
    TrainingConfigBuilder.baseline()  # or .fast() or .quality()
    .with_training(epochs=10, learning_rate=5e-5)
    .with_logging(wandb_project="my-project", run_name="exp-1")
    .with_checkpointing(checkpoint_dir="./checkpoints", frequency=2)
    .with_optimization(compile_mode="default", gradient_accumulation_steps=4)
    .build()
)

trainer = Trainer(model, model_config, config, task_spec)
results = trainer.train(train_data, val_data)
```

---

## Results Format

Both old and new APIs return similar results:

```python
results = {
    'loss_history': [0.5, 0.45, 0.42, ...],  # Loss per epoch
    'metrics_summary': pd.DataFrame({        # All metrics
        'epoch': [0, 1, 2, ...],
        'train/loss': [...],
        'val/loss': [...],
        'val/perplexity': [...]
    }),
    'best_epoch': 5,                         # Epoch with best val loss
    'checkpoint_paths': ['./checkpoints/epoch_5.pt'],
    'final_model_state': {...}               # Final state dict
}

# Access metrics
final_loss = results['loss_history'][-1]
best_epoch = results['best_epoch']
metrics_df = results['metrics_summary']
```

---

## Migration Checklist

When migrating existing code:

- [ ] Replace `test_fine_tuning()` with `Trainer.train()`
- [ ] Replace `test_hyperparameter_search()` with job queue or external tool
- [ ] Update imports:
  - `from utils.training.engine.trainer import Trainer`
  - `from utils.training.training_config import TrainingConfig`
- [ ] Create `TrainingConfig` object instead of passing kwargs
- [ ] Create `TaskSpec` object if not already created
- [ ] Update checkpoint loading (if using custom logic)
- [ ] Update results processing (format unchanged, but verify)

---

## Troubleshooting

### Import Error: "No module named 'utils.training.engine'"

**Cause:** Utils package not downloaded or outdated

**Fix:**
```python
# Run setup cell in Section 1 of notebook
!rm -rf utils/
!git clone --depth 1 https://github.com/matt-hans/transformer-builder-colab-templates.git temp_repo
!cp -r temp_repo/utils ./
!rm -rf temp_repo
```

### AttributeError: "Trainer object has no attribute 'train'"

**Cause:** Old version of Trainer class

**Fix:** Re-run setup cells to download latest utils package

### RuntimeError: "Expected train_data to be Dataset or DataLoader"

**Cause:** Incorrect data format

**Fix:**
```python
# Ensure data is PyTorch Dataset
from torch.utils.data import Dataset, DataLoader

# If data is list/array, wrap it
from utils.training.engine.data import UniversalDataModule

data_module = UniversalDataModule(
    train_dataset=train_data,
    val_dataset=val_data,
    task_spec=task_spec,
    batch_size=config.batch_size
)

trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(
    train_data=data_module.train_dataloader(),
    val_data=data_module.val_dataloader()
)
```

---

## FAQ

### Q: Why was the old API removed?

**A:** The function-based API (`test_fine_tuning()`) became too large and monolithic (1000+ lines). The new modular engine provides:
- Better separation of concerns
- Easier testing and debugging
- More extensibility
- Type safety throughout

### Q: Is the old API still available?

**A:** No. `tier3_training_utilities.py` has been removed. All code must use the new `Trainer` API.

### Q: Can I still use the notebook the same way?

**A:** Yes! The notebook cells have been updated to use the new API, but the workflow is identical:
1. Load model and data
2. Configure training
3. Run training
4. Analyze results

### Q: What about hyperparameter search?

**A:** Three options:
1. **Notebook:** Simple job queue (sequential, Cell 44)
2. **CLI:** Grid/random search (parallel, recommended)
3. **External:** Optuna, Ray Tune, W&B Sweeps (production)

### Q: Do I need to change my training configs?

**A:** Mostly no. The `TrainingConfig` object accepts the same parameters as before. Just use the builder or direct instantiation instead of function kwargs.

### Q: Will my old checkpoints still work?

**A:** Yes! Checkpoint format is unchanged. You can load old checkpoints with new code.

### Q: What about W&B integration?

**A:** Fully supported. Just set `use_wandb=True` in `TrainingConfig`.

---

## Additional Resources

- **Full Migration Guide:** `TRAINING_NOTEBOOK_UPDATE_SUMMARY.md`
- **Engine Documentation:** `/utils/training/engine/README.md`
- **Training Config Docs:** `/utils/training/training_config.py` (docstrings)
- **Trainer API Docs:** `/utils/training/engine/trainer.py` (docstrings)

---

## Support

If you encounter issues not covered here:

1. Check Cell 8 prerequisite validation in notebook
2. Review error messages (they include troubleshooting steps)
3. Compare with `training.ipynb.backup` (original notebook)
4. File an issue with full error traceback

**Happy Training!** 🚀
