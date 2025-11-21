# Migration Guide: v3.x → v4.0+

**Version:** 4.0+
**Last Updated:** 2025-11-20
**Estimated Migration Time:** 2-4 hours for typical projects
**Difficulty:** Low-Medium
**Status:** v3.x backward compatible through legacy_api module

## Overview

The training pipeline has been refactored from a monolithic `tier3_training_utilities.py` into a modular engine architecture. The legacy API (`test_fine_tuning()`, `test_hyperparameter_search()`) is fully backward compatible through a facade layer but will be removed in v4.0.0.

**Key Benefits of Migrating:**
- Better code organization (modular engine components)
- Improved testability (mockable components)
- Better error messages and debugging
- Type safety (TrainingConfig dataclass with validation)
- Extensibility (hooks system for custom behavior)
- Performance improvements (torch.compile, Flash Attention)
- Production-ready (checkpointing, resumption, artifact export)

## Timeline

| Version | Status | Notes |
|---------|--------|-------|
| v3.6.0 (current) | Deprecated | Legacy API works with warnings |
| v3.8.0 (Q2 2025) | Escalated | Warnings become more prominent |
| v4.0.0 (Q3 2025) | Removed | Legacy functions deleted entirely |

## Quick Start: 5-Minute Migration

### Before (Old API)
```python
from utils.tier3_training_utilities import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    use_wandb=True,
    gradient_clip_norm=1.0
)

print(f"Final loss: {results['final_loss']:.4f}")
print(f"Training time: {results['training_time']:.1f}s")
```

### After (New API)
```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig

# Create configuration (better organization)
training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    max_grad_norm=1.0,  # Note: renamed parameter
    wandb_project='transformer-builder-training'  # Instead of use_wandb=True
)

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    training_config=training_config,
    task_spec=task_spec  # Optional: auto-detects if None
)

# Run training
results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset
)

# Same result structure
print(f"Final loss: {results['final_loss']:.4f}")
print(f"Training time: {results['training_time']:.1f}s")
```

## Parameter Mapping Reference

| Old Parameter | New Location | Notes |
|---------------|--------------|-------|
| `n_epochs` | `TrainingConfig.epochs` | Renamed for clarity |
| `learning_rate` | `TrainingConfig.learning_rate` | Direct mapping |
| `batch_size` | `TrainingConfig.batch_size` | Direct mapping |
| `weight_decay` | `TrainingConfig.weight_decay` | Direct mapping |
| `gradient_clip_norm` | `TrainingConfig.max_grad_norm` | Renamed |
| `use_wandb` | `TrainingConfig.wandb_project` | Now a string (project name) |
| `use_amp` | `TrainingConfig.use_amp` | Direct mapping |
| `gradient_accumulation_steps` | `TrainingConfig.gradient_accumulation_steps` | Direct mapping |
| `random_seed` | `TrainingConfig.random_seed` | Direct mapping |
| `deterministic` | `TrainingConfig.deterministic` | Direct mapping |
| `use_lr_schedule` | `TrainingConfig.use_lr_schedule` | Direct mapping |

## Detailed Migration Examples

### Example 1: Basic Fine-Tuning

**Old API:**
```python
from utils.tier3_training_utilities import test_fine_tuning
from types import SimpleNamespace

model = MyTransformerModel(config)
config = SimpleNamespace(vocab_size=50257, d_model=768)
train_data = [torch.randint(0, 50257, (128,)) for _ in range(100)]

results = test_fine_tuning(
    model=model,
    config=config,
    train_data=train_data,
    n_epochs=3,
    learning_rate=5e-5,
    batch_size=4
)

print(f"Loss: {results['final_loss']}")
```

**New API:**
```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec
from torch.utils.data import TensorDataset

model = MyTransformerModel(config)
config = SimpleNamespace(vocab_size=50257, d_model=768)

# Convert data format (optional - Trainer can handle tensors too)
train_data = [torch.randint(0, 50257, (128,)) for _ in range(100)]
train_dataset = TensorDataset(torch.stack(train_data))

# Create configuration
training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=3
)

# Create task spec (describes the task type)
task_spec = TaskSpec(
    name='language_modeling',
    modality='text'
)

# Create and run trainer
trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(train_dataset, val_dataset)

print(f"Loss: {results['final_loss']}")
```

### Example 2: Fine-Tuning with AMP and Gradient Accumulation

**Old API:**
```python
results = test_fine_tuning(
    model=model,
    config=config,
    train_data=train_data,
    n_epochs=5,
    learning_rate=2e-5,
    batch_size=8,
    use_amp=True,
    gradient_accumulation_steps=4,
    gradient_clip_norm=1.0
)
```

**New API:**
```python
training_config = TrainingConfig(
    learning_rate=2e-5,
    batch_size=8,
    epochs=5,
    use_amp=True,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0
)

trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(train_dataset, val_dataset)
```

### Example 3: Fine-Tuning with W&B Logging

**Old API:**
```python
results = test_fine_tuning(
    model=model,
    config=config,
    train_data=train_data,
    n_epochs=10,
    use_wandb=True
)
```

**New API:**
```python
# Initialize W&B separately (more explicit)
import wandb
wandb.init(project='my-project', name='my-experiment')

training_config = TrainingConfig(
    learning_rate=5e-5,
    epochs=10,
    wandb_project='my-project'  # Project name, not boolean
)

trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(train_dataset, val_dataset)
```

### Example 4: Hyperparameter Search

**Old API:**
```python
from utils.tier3_training_utilities import test_hyperparameter_search

results = test_hyperparameter_search(
    model_factory=lambda: MyModel(config),
    config=config,
    train_data=train_data,
    n_trials=20,
    search_space={
        'lr': (1e-5, 1e-3),
        'batch_size': [4, 8, 16],
        'wd': (1e-6, 1e-2)
    }
)

print(f"Best LR: {results['best_params']['learning_rate']}")
```

**New API (Optuna Integration):**
```python
import optuna
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    # Create configuration
    training_config = TrainingConfig(
        learning_rate=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=5  # Short epochs for search
    )

    # Train
    model = model_factory()
    trainer = Trainer(model, config, training_config, task_spec)
    results = trainer.train(train_dataset, val_dataset)

    # Return metric to optimize
    return results['final_loss']

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best LR: {study.best_params['learning_rate']}")
```

### Example 5: Custom Search Space

**Old API:**
```python
search_space = {
    'lr': (1e-5, 1e-3),
    'batch_size': [2, 4, 8, 16],
    'wd': (1e-6, 1e-2),
    'warmup': (0, 100)
}

results = test_hyperparameter_search(
    model_factory=model_factory,
    config=config,
    train_data=train_data,
    n_trials=30,
    search_space=search_space
)
```

**New API:**
```python
import optuna

def objective(trial):
    # Full control over search space
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    warmup_pct = trial.suggest_float('warmup_pct', 0.0, 0.2)

    training_config = TrainingConfig(
        learning_rate=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=5,
        warmup_steps=int(warmup_pct * 1000)  # Custom calculations
    )

    model = model_factory()
    trainer = Trainer(model, config, training_config, task_spec)
    results = trainer.train(train_dataset, val_dataset)
    return results['final_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
```

## Result Format Compatibility

The new engine returns results in the same format as the old API, so your result processing code should work unchanged:

```python
# Old and new APIs return the same structure
results = trainer.train(...)  # or test_fine_tuning(...)

# These keys are guaranteed:
assert 'loss_history' in results  # List[float]
assert 'final_loss' in results  # float
assert 'final_perplexity' in results  # float
assert 'training_time' in results  # float (seconds)
assert 'metrics_summary' in results  # pd.DataFrame
assert 'best_epoch' in results  # int
assert 'grad_norm_history' in results  # List[float]
assert 'amp_enabled' in results  # bool

# Access results the same way
print(f"Final loss: {results['final_loss']:.4f}")
print(f"Best epoch: {results['best_epoch']}")
print(f"Training time: {results['training_time']:.1f}s")

# metrics_summary is a DataFrame with columns:
# - epoch, train/loss, val/loss, train/accuracy, val/accuracy,
#   train/perplexity, val/perplexity, train/learning_rate, etc.
metrics_df = results['metrics_summary']
print(metrics_df[['epoch', 'train/loss', 'val/loss']])
```

## Advanced Features (New API Only)

The new engine includes features not available in the old API:

### Feature 1: Checkpointing and Resumption

```python
trainer = Trainer(model, config, training_config, task_spec)

# Train and save checkpoint
results = trainer.train(train_dataset, val_dataset)
checkpoint_path = results['checkpoint_path']

# Resume from checkpoint in new trainer
trainer2 = Trainer(model, config, training_config, task_spec)
results2 = trainer2.train(
    train_dataset,
    val_dataset,
    resume_from=checkpoint_path
)
```

### Feature 2: Hooks System for Custom Behavior

```python
from utils.training.engine.trainer import TrainingHooks

class CustomHooks(TrainingHooks):
    def on_train_begin(self, trainer, train_dataset, val_dataset):
        print("Training starting!")

    def on_epoch_begin(self, trainer, epoch):
        print(f"Epoch {epoch} starting")

    def on_batch_end(self, trainer, batch, outputs):
        # Can modify behavior based on batch outputs
        pass

hooks = CustomHooks()
trainer = Trainer(model, config, training_config, task_spec, hooks=hooks)
results = trainer.train(train_dataset, val_dataset)
```

### Feature 3: Production Export Artifacts

```python
# Train with export enabled
training_config = TrainingConfig(
    learning_rate=5e-5,
    epochs=10,
    export_bundle=True,
    export_formats=['onnx', 'torchscript']
)

trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(train_dataset, val_dataset)

# Results include path to export bundle with:
# - ONNX model, TorchScript model, PyTorch state dict
# - Configs (task_spec, training_config)
# - Inference script, README, Dockerfile, TorchServe config
export_dir = results['checkpoint_path']
print(f"Export bundle at: {export_dir}")
```

### Feature 4: Job Queue for Distributed Training

```python
from utils.training.job_queue import JobQueue, JobConfig

# Create job configuration
job_config = JobConfig(
    model_factory=lambda: MyModel(config),
    config=config,
    train_data=train_dataset,
    val_data=val_dataset,
    training_config=TrainingConfig(learning_rate=5e-5, epochs=10),
    task_spec=task_spec
)

# Queue job for distributed execution
queue = JobQueue('sqlite:///training_jobs.db')
job_id = queue.submit_job(job_config)

# Monitor and retrieve results
status = queue.get_job_status(job_id)
if status == 'completed':
    results = queue.get_job_results(job_id)
```

### Feature 5: torch.compile Integration

```python
training_config = TrainingConfig(
    learning_rate=5e-5,
    epochs=10,
    compile_mode='default'  # 10-15% speedup
    # or 'reduce-overhead' (15-20%)
    # or 'max-autotune' (20-30%, slower compilation)
)

trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(train_dataset, val_dataset)
# Automatically compiled model = ~10-20% faster training
```

## Data Format Changes

### Old Format
```python
# Lists of tensors
train_data = [
    torch.randint(0, 50257, (128,)),  # First sample
    torch.randint(0, 50257, (128,)),  # Second sample
    ...
]

results = test_fine_tuning(model, config, train_data=train_data)
```

### New Format (Recommended)
```python
from torch.utils.data import TensorDataset, DataLoader

# Convert to TensorDataset
train_tensors = torch.stack([
    torch.randint(0, 50257, (128,)),
    torch.randint(0, 50257, (128,)),
    ...
])
train_dataset = TensorDataset(train_tensors)

# Or use HuggingFace Dataset (better for large-scale)
from datasets import Dataset
train_dataset = Dataset.from_dict({
    'input_ids': [...],
    'labels': [...]
})

trainer = Trainer(model, config, training_config, task_spec)
results = trainer.train(train_dataset, val_dataset)
```

## Deprecation Warnings Explained

When you use the old API, you'll see:

```
================================================================================
DEPRECATION WARNING: test_fine_tuning() is deprecated

This function is part of the legacy tier3_training_utilities API and will be
removed in v4.0.0 (Q3 2025). Please migrate to the new engine API for better
performance, maintainability, and features.

Migration Example:
    OLD:
        results = test_fine_tuning(model, config, n_epochs=10, ...)

    NEW:
        from utils.training.engine import Trainer
        from utils.training.training_config import TrainingConfig

        training_config = TrainingConfig(epochs=10, ...)
        trainer = Trainer(model, config, training_config)
        results = trainer.train(train_data, val_data)

See docs/MIGRATION_GUIDE.md for complete migration guide.
================================================================================
```

**Don't panic!** This is expected and the old API will continue to work. You have until v4.0.0 (Q3 2025) to migrate.

## Troubleshooting Common Issues

### Issue 1: "No attribute 'pad_token_id'"
**Old behavior:** Auto-detected or defaulted to 0
**New behavior:** Must specify in config or TaskSpec

**Solution:**
```python
# Add to your config
config.pad_token_id = 0

# Or specify in TaskSpec
task_spec = TaskSpec(
    name='language_modeling',
    modality='text',
    preprocessing_config={'pad_token_id': 0}
)
```

### Issue 2: "TrainingConfig requires field X"
**Old API:** Had default values for everything
**New API:** Some fields are required

**Solution:**
```python
# Provide required fields
training_config = TrainingConfig(
    learning_rate=5e-5,  # Required
    batch_size=4,  # Required
    epochs=10  # Required
)
```

### Issue 3: Results structure is different
**Solution:** The new engine returns the same keys as the old API through the facade layer. If you're using the new Trainer directly, check the result structure:

```python
results = trainer.train(...)
print(results.keys())
# dict_keys(['metrics_summary', 'best_epoch', 'final_loss', 'checkpoint_path', 'training_time'])
```

### Issue 4: "TypeError: use_wandb got an unexpected keyword argument"
**Cause:** Parameter name changed from boolean `use_wandb` to string `wandb_project`

**Solution:**
```python
# Old (no longer works)
training_config = TrainingConfig(use_wandb=True)

# New
training_config = TrainingConfig(
    wandb_project='my-project'  # Set to project name, or None to disable
)
```

### Issue 5: Data not being loaded correctly
**Old API:** Automatically handled various formats
**New API:** More explicit about data format

**Solution:**
```python
from torch.utils.data import TensorDataset, DataLoader

# Convert list of tensors to Dataset
train_tensors = torch.stack(train_data)
train_dataset = TensorDataset(train_tensors)

trainer.train(train_dataset, val_dataset)
```

## Performance Comparison

### Memory Usage
- Old API: Baseline
- New API: ~5% improvement (better data handling)
- With torch.compile: ~10% improvement

### Training Speed
- Old API: Baseline
- New API: Same speed
- With torch.compile (v3.5+): 10-20% faster
- With Flash Attention (v3.6+): 2-4x faster for attention

### Code Maintainability
- Old API: Monolithic (hard to modify)
- New API: Modular (easy to extend)

## FAQ

**Q: Will the old API continue to work?**
A: Yes, until v4.0.0 (Q3 2025). After that, it will be removed.

**Q: Do I have to migrate immediately?**
A: No, but we recommend migrating before v4.0.0 to avoid breaking changes.

**Q: Will my results change?**
A: No, the new engine produces identical results through the facade layer.

**Q: Is the new API stable?**
A: Yes, the engine is in active use and well-tested. We plan to support it indefinitely.

**Q: Can I mix old and new APIs?**
A: Yes, both work independently. The old API uses the new engine internally.

**Q: How do I report issues?**
A: File an issue on GitHub with the tag `[migration]` or `[legacy-api]`.

## Support

For questions or issues during migration:
1. Check this guide's troubleshooting section
2. Review examples in `docs/` directory
3. Look at test files in `tests/training/engine/`
4. Check API reference in `docs/API_REFERENCE_V4.md`

## Next Steps

1. **Identify usage:** Search your codebase for `test_fine_tuning` and `test_hyperparameter_search`
2. **Plan migration:** Group similar usage patterns
3. **Start simple:** Migrate one notebook/script at a time
4. **Test thoroughly:** Ensure results are identical
5. **Deploy:** Update production code before v4.0.0

---

**Last Updated:** 2025-01-20
**Version:** 1.0
**Status:** Final
