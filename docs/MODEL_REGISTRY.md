# Model Registry Guide

## Overview

The Model Registry provides a production-grade system for versioning, organizing, and tracking trained models. It uses SQLite for local storage and integrates seamlessly with CheckpointManager, ExperimentDB, and export utilities.

## Key Features

- **Semantic Versioning**: Track models with major.minor.patch versions
- **Tag-Based Organization**: Organize models with tags (production, staging, experimental)
- **Model Lineage**: Track parent-child relationships for fine-tuned models
- **Performance Metrics**: Store and compare model performance
- **Export Tracking**: Track available export formats (ONNX, TorchScript, PyTorch)
- **Metadata Storage**: Store arbitrary metadata (model size, memory requirements, etc.)
- **Query and Filter**: Powerful filtering by task type, tags, metrics, status
- **CLI Tool**: Command-line interface for all registry operations

## Architecture

### Database Schema

```
models
├── model_id (PRIMARY KEY)
├── name (TEXT, e.g., "transformer-v1")
├── version (TEXT, e.g., "1.0.0")
├── checkpoint_path (TEXT)
├── task_type (TEXT, e.g., "language_modeling")
├── config_hash (TEXT, SHA-256 of config)
├── training_run_id (INTEGER, link to ExperimentDB)
├── parent_model_id (INTEGER, for lineage)
├── created_at (TIMESTAMP)
├── metrics (JSON, performance metrics)
├── export_formats (JSON, available formats)
├── model_size_mb (REAL)
├── memory_req_gb (REAL)
├── metadata (JSON, additional info)
└── status (TEXT, active/retired)

model_tags (many-to-many)
├── tag_id (PRIMARY KEY)
├── model_id (FOREIGN KEY)
├── tag_name (TEXT, e.g., "production")
└── created_at (TIMESTAMP)

model_exports
├── export_id (PRIMARY KEY)
├── model_id (FOREIGN KEY)
├── export_format (TEXT, e.g., "onnx")
├── export_path (TEXT)
├── created_at (TIMESTAMP)
└── metadata (JSON)
```

## Usage Examples

### Basic Registration

```python
from utils.training.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry('models.db')

# Register a model
model_id = registry.register_model(
    name="gpt-small",
    version="1.0.0",
    checkpoint_path="checkpoints/epoch_10.pt",
    task_type="language_modeling",
    config_hash=ModelRegistry.compute_config_hash(config),
    metrics={"val_loss": 0.38, "perplexity": 1.46, "accuracy": 0.92}
)

print(f"Registered model {model_id}")
```

### Registration with Full Metadata

```python
model_id = registry.register_model(
    name="gpt-small",
    version="1.0.0",
    checkpoint_path="checkpoints/epoch_10.pt",
    task_type="language_modeling",
    config_hash=config_hash,
    metrics={"val_loss": 0.38, "perplexity": 1.46},
    export_formats=["onnx", "torchscript"],
    model_size_mb=256.5,
    memory_req_gb=4.0,
    training_run_id=42,  # Link to ExperimentDB
    parent_model_id=None,  # Base model (no parent)
    metadata={
        "notes": "Baseline model trained on WikiText-103",
        "epochs": 10,
        "dataset": "wikitext-103-v1",
        "num_parameters": 124_000_000
    },
    tags=["baseline", "experimental"]
)
```

### Retrieving Models

```python
# Get by ID
model = registry.get_model(model_id=1)

# Get by name and version
model = registry.get_model(name="gpt-small", version="1.0.0")

# Get production model
model = registry.get_model(tag="production")

print(f"Model: {model['name']} v{model['version']}")
print(f"Metrics: {model['metrics']}")
print(f"Checkpoint: {model['checkpoint_path']}")
```

### Tag Management (Promotion)

```python
# Promote model to production (removes tag from other models)
registry.promote_model(model_id=5, tag="production")

# Add staging tag without removing from others
registry.promote_model(
    model_id=5,
    tag="staging",
    remove_from_others=False
)

# Get all production models
prod_model = registry.get_model(tag="production")
```

### Listing and Filtering

```python
# List all active models
models = registry.list_models()

# List by task type
lm_models = registry.list_models(task_type="language_modeling")

# List by tag
prod_models = registry.list_models(tag="production")

# List by status
retired = registry.list_models(status="retired")

# Combine filters
models = registry.list_models(
    task_type="classification",
    tag="staging",
    status="active",
    limit=10
)

print(models[['model_id', 'name', 'version', 'task_type']])
```

### Model Comparison

```python
# Compare multiple models
comparison = registry.compare_models([1, 2, 3])
print(comparison[['name', 'version', 'val_loss', 'perplexity']])

# Compare specific metrics only
comparison = registry.compare_models(
    [1, 2, 3],
    metrics=["val_loss", "accuracy"]
)

# Find best model
best_row = comparison.loc[comparison['val_loss'].idxmin()]
print(f"Best model: {best_row['name']} v{best_row['version']}")
```

### Model Lineage (Fine-tuning)

```python
# Create base model
base_id = registry.register_model(
    name="base-model",
    version="1.0.0",
    checkpoint_path="checkpoints/base.pt",
    task_type="language_modeling",
    config_hash=config_hash,
    metrics={"val_loss": 0.50}
)

# Create fine-tuned model (child)
finetuned_id = registry.register_model(
    name="finetuned-model",
    version="1.1.0",
    checkpoint_path="checkpoints/finetuned.pt",
    task_type="language_modeling",
    config_hash=config_hash,
    metrics={"val_loss": 0.42},
    parent_model_id=base_id  # Link to parent
)

# Get lineage (oldest to newest)
lineage = registry.get_model_lineage(finetuned_id)
for model in lineage:
    print(f"{model['name']} v{model['version']} -> loss={model['metrics']['val_loss']}")
```

### Export Format Tracking

```python
# Add ONNX export
registry.add_export_format(
    model_id=1,
    export_format="onnx",
    export_path="exports/model.onnx",
    metadata={"opset_version": 14, "quantized": False}
)

# Add TorchScript export
registry.add_export_format(
    model_id=1,
    export_format="torchscript",
    export_path="exports/model.torchscript.pt",
    metadata={"jit_mode": "trace"}
)

# Check available formats
model = registry.get_model(model_id=1)
print(f"Available formats: {model['export_formats']}")
```

### Model Lifecycle Management

```python
# Retire old model (mark as retired, don't delete)
registry.retire_model(model_id=3)

# Delete experimental model (with confirmation)
registry.delete_model(model_id=7)

# Force delete tagged model (use with caution)
registry.delete_model(model_id=7, force=True)
```

## Integration Examples

### Integration with CheckpointManager

```python
from utils.training.engine.checkpoint import CheckpointManager
from utils.training.model_registry import ModelRegistry

# Initialize managers
checkpoint_mgr = CheckpointManager(
    checkpoint_dir="checkpoints",
    monitor="val_loss",
    mode="min"
)
registry = ModelRegistry("models.db")

# Training loop
for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_loss = validate_epoch(...)

    # Save checkpoint
    checkpoint_path = checkpoint_mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics={"val_loss": val_loss, "train_loss": train_loss}
    )

    # Register model in registry (every N epochs or at end)
    if epoch % 5 == 0 or epoch == epochs - 1:
        model_id = registry.register_model(
            name="training-run",
            version=f"1.{epoch}.0",
            checkpoint_path=checkpoint_path,
            task_type="language_modeling",
            config_hash=ModelRegistry.compute_config_hash(config.to_dict()),
            metrics={"val_loss": val_loss, "train_loss": train_loss},
            training_run_id=run_id  # Link to ExperimentDB
        )
```

### Integration with ExperimentDB

```python
from utils.training.experiment_db import ExperimentDB
from utils.training.model_registry import ModelRegistry

# Initialize databases
exp_db = ExperimentDB("experiments.db")
registry = ModelRegistry("models.db")

# Create experiment run
run_id = exp_db.log_run(
    run_name="baseline-v1",
    config=config.to_dict(),
    notes="Initial baseline"
)

# Training...
for epoch in range(epochs):
    # Log metrics to ExperimentDB
    exp_db.log_metric(run_id, "train/loss", train_loss, epoch=epoch)
    exp_db.log_metric(run_id, "val/loss", val_loss, epoch=epoch)

# Register final model with link to experiment
model_id = registry.register_model(
    name="baseline-model",
    version="1.0.0",
    checkpoint_path="checkpoints/final.pt",
    task_type="language_modeling",
    config_hash=config_hash,
    metrics={"val_loss": val_loss},
    training_run_id=run_id  # Link to experiment
)

# Query relationship
model = registry.get_model(model_id=model_id)
run = exp_db.get_run(model['training_run_id'])
print(f"Model trained in run: {run['run_name']}")
```

### Integration with Export Bundle

```python
from utils.training.export_utilities import create_export_bundle
from utils.training.model_registry import ModelRegistry

# Create export bundle
export_dir = create_export_bundle(
    model=trained_model,
    config=model_config,
    task_spec=task_spec,
    training_config=training_config
)

# Register model with export information
registry = ModelRegistry("models.db")
model_id = registry.register_model(
    name="production-model",
    version="2.0.0",
    checkpoint_path=f"{export_dir}/artifacts/model.pytorch.pt",
    task_type="vision_classification",
    config_hash=config_hash,
    metrics=final_metrics,
    export_formats=["onnx", "torchscript"],
    tags=["production"]
)

# Add export format details
registry.add_export_format(
    model_id=model_id,
    export_format="onnx",
    export_path=f"{export_dir}/artifacts/model.onnx",
    metadata={"opset_version": 14}
)
```

## CLI Tool Usage

The `scripts/manage_models.py` CLI provides full registry functionality:

### Register Model

```bash
python scripts/manage_models.py register \
    --name gpt-small \
    --version 1.0.0 \
    --checkpoint checkpoints/epoch_10.pt \
    --task-type language_modeling \
    --metrics '{"val_loss": 0.38, "perplexity": 1.46}' \
    --config '{"d_model": 768, "num_layers": 12}' \
    --tags baseline,experimental
```

### List Models

```bash
# List all models
python scripts/manage_models.py list

# List with filters
python scripts/manage_models.py list --task-type language_modeling --tag production

# Verbose output
python scripts/manage_models.py list --verbose
```

### Get Model Details

```bash
# By ID
python scripts/manage_models.py get --model-id 5

# By name and version
python scripts/manage_models.py get --name gpt-small --version 1.0.0

# By tag
python scripts/manage_models.py get --tag production
```

### Promote Model

```bash
# Promote to production (removes from other models)
python scripts/manage_models.py promote --model-id 5 --tag production

# Add tag without removing from others
python scripts/manage_models.py promote --model-id 5 --tag staging --keep-others
```

### Compare Models

```bash
# Compare all metrics
python scripts/manage_models.py compare --model-ids 1,2,3

# Compare specific metrics
python scripts/manage_models.py compare --model-ids 1,2,3 --metrics val_loss,accuracy
```

### View Lineage

```bash
python scripts/manage_models.py lineage --model-id 5
```

### Retire/Delete Model

```bash
# Retire (mark as retired)
python scripts/manage_models.py retire --model-id 3

# Delete (with confirmation)
python scripts/manage_models.py delete --model-id 7

# Force delete tagged model
python scripts/manage_models.py delete --model-id 7 --force
```

## Best Practices

### Semantic Versioning

Use semantic versioning (major.minor.patch) for models:

- **Major**: Breaking changes (incompatible architectures, different tokenizers)
- **Minor**: New features (fine-tuned versions, new export formats)
- **Patch**: Bug fixes (re-trained with bug fix, hyperparameter tweaks)

```python
# Base model
registry.register_model(name="gpt", version="1.0.0", ...)

# Fine-tuned version
registry.register_model(name="gpt", version="1.1.0", parent_model_id=base_id, ...)

# Bug fix re-train
registry.register_model(name="gpt", version="1.1.1", ...)

# Architecture change
registry.register_model(name="gpt", version="2.0.0", ...)
```

### Tag Strategy

Recommended tags:

- `production`: Production-ready models
- `staging`: Models ready for staging/pre-production testing
- `experimental`: Experimental models under development
- `baseline`: Baseline models for comparison
- `deprecated`: Models scheduled for retirement

```python
# Promote through stages
registry.register_model(..., tags=["experimental"])
registry.promote_model(model_id, "staging")
registry.promote_model(model_id, "production")
```

### Config Hash

Always compute config hash from complete model architecture:

```python
config = {
    "vocab_size": 50257,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "max_seq_len": 128,
    "dropout": 0.1
}

config_hash = ModelRegistry.compute_config_hash(config)
```

### Metrics Storage

Store comprehensive metrics for comparison:

```python
metrics = {
    # Loss metrics
    "val_loss": 0.38,
    "train_loss": 0.42,

    # Task-specific metrics
    "perplexity": 1.46,
    "accuracy": 0.92,
    "f1_score": 0.91,

    # Performance metrics
    "inference_time_ms": 12.5,
    "tokens_per_second": 8000,

    # Training metrics
    "best_epoch": 8,
    "total_epochs": 10,
    "training_time_hours": 4.2
}

registry.register_model(..., metrics=metrics)
```

### Model Lineage

Track lineage for fine-tuned models:

```python
# Base model
base_id = registry.register_model(
    name="base-gpt",
    version="1.0.0",
    ...
)

# Domain-specific fine-tune
domain_id = registry.register_model(
    name="medical-gpt",
    version="1.0.0",
    parent_model_id=base_id,
    ...
)

# Task-specific fine-tune
task_id = registry.register_model(
    name="medical-qa-gpt",
    version="1.0.0",
    parent_model_id=domain_id,
    ...
)

# View full lineage
lineage = registry.get_model_lineage(task_id)
# Shows: base-gpt -> medical-gpt -> medical-qa-gpt
```

## Performance Considerations

### Query Performance

- Queries: **<10ms** for typical operations
- Writes: **<100ms** for registration
- Indexes on: `task_type`, `status`, `config_hash`, `tag_name`

### Optimization Tips

1. **Use tags for frequent queries**: `get_model(tag="production")` is indexed
2. **Limit results**: Use `limit` parameter for large registries
3. **Filter early**: Combine filters to reduce result set
4. **Batch operations**: Use transactions for multiple operations

```python
# Good: Single query with filters
models = registry.list_models(
    task_type="language_modeling",
    tag="production",
    limit=10
)

# Avoid: Multiple queries
all_models = registry.list_models(limit=1000)
filtered = [m for m in all_models if m['task_type'] == 'language_modeling']
```

## Troubleshooting

### Duplicate Version Error

```
ValueError: Model 'gpt-small' version '1.0.0' already exists
```

Solution: Use a different version number or update existing model.

### Missing Checkpoint File

```
FileNotFoundError: Checkpoint not found: /path/to/checkpoint.pt
```

Solution: Verify checkpoint path exists before registration.

### Tagged Model Delete Error

```
ValueError: Model 5 has 2 tag(s). Use force=True to delete anyway
```

Solution: Either remove tags first or use `force=True` to delete.

### Database Locked Error

```
sqlite3.OperationalError: database is locked
```

Solution: SQLite doesn't support concurrent writes. Use separate registries for parallel processes or implement locking.

## Migration and Maintenance

### Backup Registry

```bash
# Backup SQLite database
cp model_registry.db model_registry.db.backup

# Or use SQLite backup
sqlite3 model_registry.db ".backup model_registry.db.backup"
```

### Export Registry to CSV

```python
import pandas as pd

registry = ModelRegistry("models.db")
models = registry.list_models(limit=1000)
models.to_csv("registry_export.csv", index=False)
```

### Clean Up Retired Models

```python
retired = registry.list_models(status="retired")
for _, model in retired.iterrows():
    # Delete models retired over 90 days ago
    created = pd.to_datetime(model['created_at'])
    if (pd.Timestamp.now() - created).days > 90:
        registry.delete_model(model['model_id'], force=True)
```

## API Reference

See inline docstrings in `utils/training/model_registry.py` for complete API documentation.

### Core Methods

- `register_model()`: Register new model
- `get_model()`: Retrieve model by ID, name+version, or tag
- `list_models()`: List models with filtering
- `promote_model()`: Assign tag to model
- `retire_model()`: Mark model as retired
- `delete_model()`: Delete model from registry
- `compare_models()`: Compare metrics across models
- `get_model_lineage()`: Get parent-child chain
- `add_export_format()`: Add export format tracking
- `compute_config_hash()`: Compute SHA-256 of config

## Future Enhancements

Planned features for future versions:

- [ ] Multi-registry support (distributed registries)
- [ ] S3/GCS integration for checkpoint storage
- [ ] Model approval workflows (pending -> approved -> production)
- [ ] Automated model monitoring (drift detection)
- [ ] Model cards auto-generation
- [ ] Integration with HuggingFace Hub
- [ ] Web UI for registry visualization
- [ ] REST API for remote access
