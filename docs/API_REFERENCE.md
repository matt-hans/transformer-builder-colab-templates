# API Reference - Training Engine v4.0+

**Version:** 4.0+
**Last Updated:** 2025-11-20
**Status:** Production
**Target Audience:** Developers integrating the modular training engine

---

## Table of Contents

1. [Overview](#overview)
2. [Core Engine Components](#core-engine-components)
   - [CheckpointManager](#checkpointmanager)
   - [LossStrategy & Implementations](#lossstrategy)
   - [GradientMonitor](#gradientmonitor)
   - [GradientAccumulator](#gradientaccumulator)
   - [DataLoaderFactory & CollatorRegistry](#dataloaderfactory)
   - [TrainingLoop & ValidationLoop](#training-and-validation-loops)
   - [MetricsEngine](#metricsengine)
   - [Trainer Orchestrator](#trainer)
3. [Configuration System](#configuration-system)
   - [TrainingConfig](#trainingconfig)
   - [TrainingConfigBuilder](#trainingconfigbuilder)
   - [TaskSpec](#taskspec)
4. [Production Features](#production-features)
   - [ModelRegistry](#modelregistry)
   - [JobQueue & Scheduler](#jobqueue)
   - [ExportBundle](#export-bundle)
   - [RetrainingTriggers](#retraining-triggers)
5. [Data Loading](#data-loading)
   - [UniversalDataModule](#universaldatamodule)
   - [CollatorRegistry](#collatorregistry)
6. [Utilities](#utilities)
   - [MetricsTracker](#metricstracker)
   - [ExperimentDB](#experimentdb)
   - [SeedManager](#seedmanager)
7. [Common Workflows](#common-workflows)
8. [API Stability & Deprecation](#api-stability)

---

## Overview

The v4.0+ training engine is a **modular, composable architecture** built on design patterns:

### Architecture Principles

| Pattern | Component | Benefit |
|---------|-----------|---------|
| **Strategy Pattern** | `LossStrategy` with 5 implementations | Task-specific loss without hardcoding |
| **Registry Pattern** | `CollatorRegistry`, `LossStrategyRegistry` | Extensible without core changes |
| **Builder Pattern** | `TrainingConfigBuilder` | Fluent, chainable configuration API |
| **Protocol-Based** | `TrainingHooks`, `DataModuleProtocol` | Framework-agnostic interfaces |
| **Composition** | `Trainer` delegates to specialists | Single responsibility, easy testing |

### Performance Characteristics

- **Overhead per step:** <5ms (measured across 100 epochs)
- **Memory overhead:** ~50MB for tracking metadata
- **Checkpoint I/O:** ~500ms per 100M parameter model
- **Drift detection:** ~10ms per epoch (parallel-friendly)

---

## Core Engine Components

### CheckpointManager

**Location:** `utils.training.engine.checkpoint`

Comprehensive checkpoint management with atomic state persistence and recovery.

#### Class: `CheckpointManager`

```python
from utils.training.engine import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    keep_best_k=3,
    keep_last_n=5,
    monitor='val_loss',
    mode='min',
    save_interval_epochs=1,
    drive_backup=False
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | `str` | Required | Directory for checkpoints (created if missing) |
| `keep_best_k` | `int` | `3` | Keep top K checkpoints by monitor metric |
| `keep_last_n` | `int` | `5` | Keep N most recent checkpoints |
| `monitor` | `str` | `'val_loss'` | Metric to track (e.g., 'val_loss', 'val_accuracy') |
| `mode` | `str` | `'min'` | 'min' (lower is better) or 'max' |
| `save_interval_epochs` | `int` | `1` | Save every N epochs |
| `drive_backup` | `bool` | `False` | Enable Google Drive backup (Colab) |
| `drive_backup_path` | `Optional[str]` | `None` | Path in Drive for backups |

#### Key Methods

**`save_checkpoint(epoch, metrics, model, optimizer, scheduler=None, metadata=None)`**

Save checkpoint with automatic retention policy.

```python
manager.save_checkpoint(
    epoch=10,
    metrics={'val_loss': 0.38, 'val_accuracy': 0.87},
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metadata={'best_epoch': 5, 'notes': 'Added attention dropout'}
)
# Returns: CheckpointMetadata(path, is_best, epoch, metrics, ...)
```

**`load_checkpoint(path, model, optimizer=None, scheduler=None)`**

Resume training from checkpoint.

```python
checkpoint = manager.load_checkpoint(
    path='./checkpoints/epoch_010_best.pt',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)
# Returns: CheckpointMetadata with state_dict loaded into model/optimizer
```

**`get_best_checkpoint()`**

Retrieve path to best checkpoint by monitored metric.

```python
best_path = manager.get_best_checkpoint()
print(f"Best: {best_path}")  # ./checkpoints/epoch_005_best.pt
```

**`get_checkpoint_history()`**

Get DataFrame of all saved checkpoints with metrics.

```python
history = manager.get_checkpoint_history()
print(history[['epoch', 'val_loss', 'is_best']])
#    epoch  val_loss  is_best
# 0      1       0.62    False
# 1      2       0.51    False
# 2      3       0.42     True
```

#### Data Structures

**`CheckpointMetadata`**

```python
@dataclass
class CheckpointMetadata:
    path: str                          # Full path to checkpoint
    epoch: int                         # Training epoch
    metrics: Dict[str, float]          # Validation metrics
    is_best: bool                      # Is this the best so far?
    timestamp: str                     # ISO 8601 timestamp
    model_size_mb: float               # Checkpoint file size
    optimizer_state_size_mb: float     # Optimizer state size
```

#### Design Considerations

- **Atomic saves:** Model + optimizer + metadata saved together
- **Automatic cleanup:** Maintains `keep_best_k` and `keep_last_n` invariants
- **Drive backup:** Optional Colab integration for durability
- **Metric-aware:** Tracks which metric justified "best" label

---

### LossStrategy

**Location:** `utils.training.engine.loss`

Protocol-based loss computation with 5 task-specific implementations.

#### Protocol: `LossStrategy`

```python
from typing import Protocol

class LossStrategy(Protocol):
    """Loss computation strategy interface."""

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: Optional[LossInputs] = None
    ) -> torch.Tensor:
        """Compute scalar loss tensor."""
        ...

    def validate_inputs(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate input shapes and types. Raise ValueError if invalid."""
        ...
```

#### Registry: `LossStrategyRegistry`

```python
from utils.training.engine import LossStrategyRegistry, get_loss_strategy

# Get built-in strategy
strategy = get_loss_strategy('language_modeling')

# Or use registry directly
registry = LossStrategyRegistry()
strategy = registry.get('classification')

# Register custom strategy
class CustomLoss:
    def compute_loss(self, logits, labels, inputs=None):
        # Custom implementation
        return F.cross_entropy(logits, labels)

    def validate_inputs(self, logits, labels):
        pass

registry.register('custom', CustomLoss())
```

#### Built-in Implementations

**1. `LanguageModelingLoss`**

For causal LM and next-token prediction tasks.

```python
from utils.training.engine import LanguageModelingLoss

strategy = LanguageModelingLoss()
loss = strategy.compute_loss(
    logits=model_output,      # [batch, seq_len, vocab_size]
    labels=input_ids,         # [batch, seq_len]
    inputs=LossInputs(
        pad_token_id=0,       # Exclude padding from loss
        attention_mask=mask   # Optional: further refinement
    )
)
```

**Properties:**
- Excludes padding tokens from loss computation
- Reshapes [batch, seq, vocab] → [batch*seq, vocab] for efficiency
- Returns mean loss (reduced over non-padding tokens)

**2. `ClassificationLoss`**

For single-label classification tasks.

```python
from utils.training.engine import ClassificationLoss

strategy = ClassificationLoss()
loss = strategy.compute_loss(
    logits=model_output,      # [batch, num_classes]
    labels=class_ids,         # [batch]
    inputs=LossInputs(
        class_weights=weights # Optional: per-class weighting
    )
)
```

**Properties:**
- Supports weighted cross-entropy
- Validates single-label assumption
- Efficient for fixed output dimension

**3. `PEFTAwareLoss`**

For parameter-efficient fine-tuning (LoRA, adapters).

```python
from utils.training.engine import PEFTAwareLoss

strategy = PEFTAwareLoss(
    base_model_frozen=True,  # Assume base parameters frozen
    adapter_scale=1.0
)
loss = strategy.compute_loss(logits, labels)
```

**Properties:**
- Handles adapter-specific gradient flows
- Prevents adaptation to frozen base parameters
- Useful for LoRA/adapter tuning

**4. `QuantizationSafeLoss`**

For models with quantized weights or activations.

```python
from utils.training.engine import QuantizationSafeLoss

strategy = QuantizationSafeLoss(
    bits=8,                   # 8-bit quantization
    symmetric=True
)
loss = strategy.compute_loss(logits, labels)
```

**Properties:**
- Clips gradients to prevent quantization artifacts
- Supports int8, int4 quantization schemes
- Preserves numerical stability

**5. `VisionLoss`**

For vision tasks (image classification, segmentation).

```python
from utils.training.engine import VisionLoss

strategy = VisionLoss(task='classification')
loss = strategy.compute_loss(
    logits=model_output,      # [batch, num_classes]
    labels=labels,            # [batch]
    inputs=LossInputs(
        pixel_values=images   # Optional: for metrics
    )
)
```

**Properties:**
- Task-aware: classification, detection, segmentation
- Handles image-specific preprocessing
- Integrates with vision metrics

#### Type-Safe Inputs: `LossInputs`

```python
from typing_extensions import TypedDict

class LossInputs(TypedDict, total=False):
    logits: torch.Tensor           # Model logits
    labels: torch.Tensor           # Ground truth
    attention_mask: Optional[torch.Tensor]
    pad_token_id: Optional[int]
    pixel_values: Optional[torch.Tensor]
    class_weights: Optional[torch.Tensor]
```

#### Model Output Parsing: `ModelOutput`

```python
from utils.training.engine.loss import ModelOutput

# Automatic parsing from various formats
output = ModelOutput.from_raw(model(input_ids))

# Now type-safe access
loss = strategy.compute_loss(
    logits=output.logits,
    labels=labels
)
```

---

### GradientMonitor

**Location:** `utils.training.engine.gradient_monitor`

Monitor gradient health: detect vanishing/exploding gradients, track norms.

#### Class: `GradientMonitor`

```python
from utils.training.engine import GradientMonitor, check_gradient_health

monitor = GradientMonitor(
    check_interval=10,              # Check every N batches
    norm_threshold_warning=1.0,     # Warn if norm > 1.0
    norm_threshold_critical=10.0,   # Fail if norm > 10.0
    log_distribution=True           # Log grad norm histogram
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_interval` | `int` | `10` | Check gradients every N batches |
| `norm_threshold_warning` | `float` | `1.0` | Warn if max gradient norm > value |
| `norm_threshold_critical` | `float` | `10.0` | Fail if max gradient norm > value |
| `log_distribution` | `bool` | `True` | Track gradient norm histogram |

#### Key Methods

**`check_step(model)`**

Perform health check at current training step.

```python
health = monitor.check_step(model)
if health.has_exploding_gradients:
    print(f"⚠️ Gradient explosion detected: {health.max_norm:.4f}")
if health.has_vanishing_gradients:
    print(f"⚠️ Gradient vanishing: {health.min_norm:.6f}")
```

**`get_statistics()`**

Retrieve gradient statistics over time.

```python
stats = monitor.get_statistics()
print(stats['mean_norm_history'])    # List of mean norms
print(stats['max_norm_history'])     # List of max norms
print(stats['check_count'])          # Total checks performed
```

#### Data Structures

**`GradientHealth`**

```python
@dataclass
class GradientHealth:
    max_norm: float
    min_norm: float
    mean_norm: float
    has_exploding_gradients: bool
    has_vanishing_gradients: bool
    num_zero_gradients: int
    timestamp: str
```

#### Utility Function

**`check_gradient_health(model, threshold_max=1.0, threshold_min=1e-7)`**

Standalone health check without monitoring loop.

```python
from utils.training.engine import check_gradient_health

health = check_gradient_health(
    model=model,
    threshold_max=1.0,
    threshold_min=1e-7
)

if health.has_exploding_gradients:
    # Apply gradient clipping, reduce LR, etc.
    pass
```

---

### GradientAccumulator

**Location:** `utils.training.engine.gradient_accumulator`

Manage gradient accumulation with step tracking and statistics.

#### Class: `GradientAccumulator`

```python
from utils.training.engine import GradientAccumulator

accumulator = GradientAccumulator(
    accumulation_steps=4,
    max_gradient_norm=1.0,
    clip_by_global_norm=True
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accumulation_steps` | `int` | `1` | Accumulate N batches before optimizer step |
| `max_gradient_norm` | `Optional[float]` | `1.0` | Max L2 norm for gradient clipping |
| `clip_by_global_norm` | `bool` | `True` | Clip by global norm (vs per-parameter) |

#### Key Methods

**`should_update_optimizer(step)`**

Check if optimizer update should occur.

```python
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()

    if accumulator.should_update_optimizer(step):
        accumulator.clip_gradients(model)
        optimizer.step()
        optimizer.zero_grad()

        # Log effective step for W&B
        effective_step = accumulator.get_effective_step(step)
        logger.log({'loss': loss, 'step': effective_step})
```

**`clip_gradients(model)`**

Apply gradient clipping with accumulated stats.

```python
accumulator.clip_gradients(model)
stats = accumulator.get_statistics()
print(f"Pre-clip norm: {stats['norm_before']:.4f}")
print(f"Post-clip norm: {stats['norm_after']:.4f}")
```

**`get_effective_step(global_step)`**

Map global step to effective optimizer step.

```python
# With accumulation_steps=4:
# global_step=0 → effective_step=0
# global_step=4 → effective_step=1
effective = accumulator.get_effective_step(global_step)
```

#### Data Structures

**`AccumulationStats`**

```python
@dataclass
class AccumulationStats:
    global_step: int              # Backward pass count
    effective_step: int           # Optimizer update count
    accumulated_loss: float       # Sum of accumulated losses
    norm_before_clip: float       # L2 norm before clipping
    norm_after_clip: float        # L2 norm after clipping
    num_parameters: int           # Total trainable parameters
```

---

### DataLoaderFactory

**Location:** `utils.training.engine.data`

Factory pattern for creating optimized DataLoaders with task-aware collation.

#### Class: `DataLoaderFactory`

```python
from utils.training.engine import DataLoaderFactory, DataLoaderConfig

factory = DataLoaderFactory(
    task_spec=task_spec,
    seed=42,
    pin_memory=True,
    num_workers=4
)

train_loader = factory.create_loader(
    dataset=train_dataset,
    config=DataLoaderConfig(
        batch_size=32,
        shuffle=True,
        drop_last=True
    )
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_spec` | `TaskSpec` | Required | Task specification for collation strategy |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `pin_memory` | `bool` | `True` | Pin memory for GPU transfer efficiency |
| `num_workers` | `int` | `4` | DataLoader workers (0 for CPU-only) |

#### Key Methods

**`create_loader(dataset, config, is_eval=False)`**

Create DataLoader with optimal settings for task.

```python
loader = factory.create_loader(
    dataset=dataset,
    config=DataLoaderConfig(
        batch_size=16,
        shuffle=not is_eval,
        drop_last=not is_eval
    ),
    is_eval=is_eval
)
```

#### CollatorRegistry

**Location:** `utils.training.engine.data`

Registry of task-aware collation strategies.

```python
from utils.training.engine import CollatorRegistry

registry = CollatorRegistry()

# Get built-in collator
collator = registry.get('language_modeling')

# Register custom collator
def my_collator(batch):
    # Custom batching logic
    return {
        'input_ids': torch.tensor([...]),
        'labels': torch.tensor([...])
    }

registry.register('custom', my_collator)
```

**Built-in Collators:**

- `'language_modeling'` - For LM tasks (pads to seq_len, excludes padding from loss)
- `'classification'` - For classification (pads to max in batch, right-pads)
- `'vision'` - For vision tasks (normalizes images, handles variable sizes)
- `'sequence_to_sequence'` - For seq2seq (pads input/output independently)

---

### TrainingLoop & ValidationLoop

**Location:** `utils.training.engine.loop`

Epoch execution engines for training and validation with pluggable components.

#### Class: `TrainingLoop`

```python
from utils.training.engine import TrainingLoop

train_loop = TrainingLoop(
    gradient_accumulator=accumulator,
    gradient_monitor=monitor,
    loss_strategy=strategy,
    gradient_clip_norm=1.0
)

epoch_result = train_loop.run_epoch(
    epoch=1,
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    metrics_callback=None
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_accumulator` | `GradientAccumulator` | Required | Gradient accumulation handler |
| `gradient_monitor` | `GradientMonitor` | Required | Gradient health checker |
| `loss_strategy` | `LossStrategy` | Required | Task-specific loss |
| `gradient_clip_norm` | `Optional[float]` | `1.0` | Max gradient norm |

#### Key Methods

**`run_epoch(epoch, model, train_loader, optimizer, scheduler, device, metrics_callback)`**

Execute one training epoch.

```python
result = train_loop.run_epoch(
    epoch=5,
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=torch.device('cuda'),
    metrics_callback=lambda loss, step: print(f"Loss: {loss:.4f}")
)

print(f"Epoch loss: {result.loss}")
print(f"Time: {result.epoch_duration_seconds:.1f}s")
```

#### Class: `ValidationLoop`

```python
from utils.training.engine import ValidationLoop

val_loop = ValidationLoop(
    loss_strategy=strategy
)

val_result = val_loop.run_epoch(
    epoch=1,
    model=model,
    val_loader=val_loader,
    device=device,
    metrics_callback=None
)
```

#### Data Structures

**`EpochResult`**

```python
@dataclass
class EpochResult:
    epoch: int
    loss: float                    # Mean loss for epoch
    metrics: Dict[str, float]      # Additional metrics (accuracy, etc.)
    epoch_duration_seconds: float
    num_samples: int
    num_batches: int
    learning_rate: float           # LR at end of epoch
```

---

### MetricsEngine

**Location:** `utils.training.engine.metrics`

Comprehensive metrics tracking with drift detection and performance alerts.

#### Class: `MetricsEngine`

```python
from utils.training.engine import MetricsEngine, AlertConfig

engine = MetricsEngine(
    use_wandb=True,
    gradient_accumulation_steps=4,
    drift_threshold_warning=0.1,
    drift_threshold_critical=0.2,
    alert_config=AlertConfig(
        val_loss_spike_threshold=0.2,
        accuracy_drop_threshold=0.05
    )
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | `bool` | `False` | Log to Weights & Biases |
| `gradient_accumulation_steps` | `int` | `1` | For step-aware W&B logging |
| `drift_threshold_warning` | `float` | `0.1` | JS divergence warning threshold |
| `drift_threshold_critical` | `float` | `0.2` | JS divergence critical threshold |
| `alert_config` | `AlertConfig` | Required | Alert thresholds |

#### Key Methods

**`log_epoch(epoch, train_metrics, val_metrics, learning_rate, gradient_norm, epoch_duration, reference_profile, current_profile)`**

Log epoch-level metrics with drift detection.

```python
drift_metrics = engine.log_epoch(
    epoch=5,
    train_metrics={'loss': 0.42, 'accuracy': 0.85},
    val_metrics={'loss': 0.38, 'accuracy': 0.87},
    learning_rate=1e-4,
    gradient_norm=0.5,
    epoch_duration=120.5,
    reference_profile=ref_profile,
    current_profile=curr_profile
)

if drift_metrics.status == 'critical':
    print(f"🚨 Critical drift: {drift_metrics.affected_features}")
```

**`log_scalar(name, value, step)`**

Log per-batch scalar metrics to W&B.

```python
engine.log_scalar('train/batch_loss', 0.45, step=100)
engine.log_scalar('train/learning_rate', 1e-4, step=100)
```

**`log_confidence(logits, labels, step=None)`**

Log prediction confidence metrics.

```python
engine.log_confidence(
    logits=model_output,
    labels=true_labels,
    step=global_step
)
# Logs: top1_confidence, top5_confidence, entropy
```

**`has_alerts()` / `get_alerts()`**

Check for and retrieve performance alerts.

```python
if engine.has_alerts():
    for alert in engine.get_alerts():
        print(f"🚨 {alert['type']}: {alert['message']}")
```

#### Data Structures

**`DriftMetrics`**

```python
@dataclass
class DriftMetrics:
    js_divergence: float           # 0-1 (lower=less drift)
    status: Literal['healthy', 'warning', 'critical']
    affected_features: List[str]   # Features with drift
    timestamp: str                 # ISO 8601
    details: Dict[str, float]      # Per-feature scores
```

**`AlertConfig`**

```python
@dataclass
class AlertConfig:
    val_loss_spike_threshold: float      # Increase > X% triggers alert
    accuracy_drop_threshold: float       # Drop > X% triggers alert
    gradient_norm_threshold: float       # Norm > X triggers alert
    patience_epochs: int                 # Epochs before alert
```

---

### Trainer

**Location:** `utils.training.engine.trainer`

High-level orchestrator delegating to specialized components.

#### Class: `Trainer`

```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig

trainer = Trainer(
    model=model,
    config=model_config,
    training_config=config,
    task_spec=task_spec,
    hooks=None,
    device='cuda'
)

results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | Required | PyTorch model to train |
| `config` | `SimpleNamespace` | Required | Model architecture config |
| `training_config` | `TrainingConfig` | Required | Training hyperparameters |
| `task_spec` | `TaskSpec` | Required | Task specification |
| `hooks` | `Optional[TrainingHooks]` | `None` | Custom lifecycle hooks |
| `device` | `str` | `'cuda'` | Device for training |

#### Key Methods

**`train(train_data, val_data)`**

Execute full training loop with checkpointing and early stopping.

```python
results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset
)

print(f"Best epoch: {results['best_epoch']}")
print(f"Best val loss: {results['best_val_loss']:.4f}")
print(f"Total time: {results['total_time']:.1f}s")
```

#### Hook System: `TrainingHooks`

Extensible callback interface for custom behavior injection.

```python
class CustomHooks:
    def on_training_start(self) -> None:
        """Called once before training."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at epoch start."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """Called after each batch."""
        pass

    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        """Called after validation."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at epoch end with combined metrics."""
        pass

    def on_training_end(self) -> None:
        """Called once after training completes."""
        pass

trainer = Trainer(..., hooks=CustomHooks())
```

---

## Configuration System

### TrainingConfig

**Location:** `utils.training.training_config`

Complete training configuration for reproducible experiments.

#### Class: `TrainingConfig`

```python
from utils.training.training_config import TrainingConfig

config = TrainingConfig(
    # Reproducibility
    random_seed=42,
    deterministic=False,

    # Hyperparameters
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Features
    use_amp=True,
    gradient_accumulation_steps=1,
    early_stopping_patience=5,

    # Checkpointing
    checkpoint_dir='./checkpoints',
    save_every_n_epochs=1,

    # Experiment tracking
    wandb_project='transformer-training',
    run_name='baseline-v1',

    # Model
    vocab_size=50257,
    max_seq_len=128,
    d_model=768,
    num_layers=12
)
```

#### Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | `float` | `5e-5` | Optimizer learning rate |
| `batch_size` | `int` | `4` | Training batch size |
| `epochs` | `int` | `10` | Total epochs to train |
| `warmup_ratio` | `float` | `0.1` | Fraction of steps for LR warmup |
| `gradient_accumulation_steps` | `int` | `1` | Accumulate N batches before update |
| `max_grad_norm` | `float` | `1.0` | Gradient clipping threshold |
| `checkpoint_dir` | `str` | Required | Path for checkpoint saves |
| `random_seed` | `int` | `42` | Seed for reproducibility |
| `deterministic` | `bool` | `False` | Fully reproducible mode (slower) |

#### Key Methods

**`validate()`**

Validate configuration internally consistency.

```python
config.validate()
# Raises ValueError with detailed message if invalid
```

**`to_dict()`**

Convert to dictionary for serialization.

```python
config_dict = config.to_dict()
json.dump(config_dict, open('config.json', 'w'))
```

**`save(path=None)`**

Save configuration to JSON file.

```python
config_path = config.save()  # Auto-generates timestamped name
# or
config.save('experiments/my_config.json')
```

**`@classmethod load(path)`**

Load configuration from JSON file.

```python
config = TrainingConfig.load('experiments/my_config.json')
```

### TrainingConfigBuilder

**Location:** `utils.training.training_config`

Fluent builder API for configuration with presets.

#### Usage: Presets

```python
from utils.training.training_config import TrainingConfigBuilder

# 1. Quick Prototype (3 epochs, 12M params, for debugging)
config = TrainingConfigBuilder.quick_prototype().build()

# 2. Baseline (10 epochs, 125M params, balanced)
config = TrainingConfigBuilder.baseline().build()

# 3. Production (20 epochs, export enabled, checkpointing)
config = TrainingConfigBuilder.production().build()

# 4. Distributed (DDP/FSDP, 4 GPUs)
config = TrainingConfigBuilder.distributed().build()

# 5. Low Memory (Colab free tier, 2 batch, 8x accumulation)
config = TrainingConfigBuilder.low_memory().build()
```

#### Usage: Customization

```python
# Customize preset via method chaining
config = (TrainingConfigBuilder.baseline()
    .with_training(epochs=30, batch_size=8)
    .with_optimizer(gradient_accumulation_steps=4)
    .with_logging(run_name='custom-baseline')
    .build()
)
```

#### Builder Methods

**`.with_model(...)`** - Configure model architecture

```python
config = (TrainingConfigBuilder()
    .with_model(
        model_name='custom-gpt',
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=512
    )
    .build()
)
```

**`.with_training(...)`** - Configure training hyperparameters

```python
config = (builder
    .with_training(
        learning_rate=5e-5,
        batch_size=16,
        epochs=20,
        warmup_ratio=0.1,
        validation_split=0.1
    )
    .build()
)
```

**`.with_optimizer(...)`** - Configure optimizer settings

```python
config = (builder
    .with_optimizer(
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_accumulation_steps=4
    )
    .build()
)
```

**`.with_hardware(...)`** - Configure hardware settings

```python
config = (builder
    .with_hardware(
        use_amp=True,
        compile_mode='default',
        devices=1,
        precision='bf16-mixed'
    )
    .build()
)
```

**`.with_logging(...)`** - Configure experiment tracking

```python
config = (builder
    .with_logging(
        wandb_project='my-project',
        run_name='exp-1',
        notes='Testing new augmentation'
    )
    .build()
)
```

### TaskSpec

**Location:** `utils.training.task_spec`

Task specification for data loading, loss computation, and metrics.

#### Class: `TaskSpec`

```python
from utils.training.task_spec import TaskSpec

# Language modeling task
task = TaskSpec(
    name='wikitext-lm',
    modality='text',
    task_type='language_modeling',
    preprocessing_config={
        'truncate_length': 512,
        'stride': 256
    }
)

# Classification task
task = TaskSpec(
    name='imdb-sentiment',
    modality='text',
    task_type='classification',
    num_classes=2,
    class_names=['negative', 'positive']
)

# Vision task
task = TaskSpec(
    name='imagenet-classification',
    modality='vision',
    task_type='image_classification',
    num_classes=1000
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Task name |
| `modality` | `str` | Required | 'text', 'vision', 'audio', etc. |
| `task_type` | `str` | Required | 'language_modeling', 'classification', etc. |
| `num_classes` | `int` | `None` | For classification tasks |
| `preprocessing_config` | `Dict` | `{}` | Task-specific preprocessing |

#### Factory Methods

```python
# Language modeling
task = TaskSpec.language_modeling(name='wikitext')

# Classification
task = TaskSpec.classification(name='imdb', num_classes=2)

# Vision
task = TaskSpec.vision_tiny()  # Small preset for testing
```

---

## Production Features

### ModelRegistry

**Location:** `utils.training.model_registry`

SQLite-based registry for model versioning, metadata, and lineage tracking.

#### Class: `ModelRegistry`

```python
from utils.training.model_registry import ModelRegistry

registry = ModelRegistry('models.db')

# Register model
model_id = registry.register_model(
    name='transformer-v1',
    version='1.0.0',
    checkpoint_path='checkpoints/epoch_10.pt',
    task_type='language_modeling',
    metrics={'val_loss': 0.38, 'perplexity': 1.46},
    config_hash='abc123',
    training_run_id=42
)

# Tag for deployment
registry.promote_model(model_id, 'production')

# Retrieve
model = registry.get_model(tag='production')

# Compare
comparison = registry.compare_models([1, 2, 3])
print(comparison[['version', 'val_loss', 'created_at']])
```

#### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register_model(...)` | See constructor params below | Register new model version |
| `get_model(model_id=None, version=None, tag=None)` | `ModelRegistryEntry` | Retrieve model by ID, version, or tag |
| `promote_model(model_id, tag)` | `None` | Add tag to model |
| `list_models(limit=None)` | `pd.DataFrame` | List all models with metadata |
| `compare_models(model_ids)` | `pd.DataFrame` | Compare metrics across models |
| `delete_model(model_id)` | `None` | Mark model as retired |

#### Data Structure: `ModelRegistryEntry`

```python
@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: int
    name: str                      # e.g., 'transformer-v1'
    version: str                   # Semantic version (1.0.0)
    checkpoint_path: str           # Path to .pt file
    task_type: str                 # 'language_modeling', 'classification', etc.
    config_hash: str               # SHA-256 hash of config
    training_run_id: Optional[int] # Link to ExperimentDB
    parent_model_id: Optional[int] # For fine-tuned models
    created_at: str                # ISO 8601 timestamp
    metrics: Dict[str, float]      # val_loss, accuracy, etc.
    export_formats: List[str]      # ['onnx', 'torchscript', 'pytorch']
    model_size_mb: float           # Size in MB
    memory_req_gb: float           # Est. GPU memory
    status: str                    # 'active', 'retired', 'experimental'
    tags: List[str]                # ['production', 'staging', etc.]
```

---

### JobQueue & Scheduler

**Location:** `utils.training.job_queue`

SQLite-based job queue for automated training workflows.

#### Class: `JobManager`

```python
from utils.training.job_queue import JobManager, JobExecutor

# Submit jobs
manager = JobManager('jobs.db')

job_id = manager.submit_job(
    job_type='training',
    config={'training_config': config.to_dict()},
    priority=5,  # 1-10, higher = more urgent
    max_retries=3
)

# Monitor
job = manager.get_job(job_id)
print(f"Status: {job.status}")  # pending, running, completed, failed

# Execute (worker process)
executor = JobExecutor(manager, worker_id='worker-1')
executor.run_worker(max_jobs=10, timeout_minutes=60)
```

#### Scheduler: `TrainingScheduler`

```python
from utils.training.job_queue import TrainingScheduler

scheduler = TrainingScheduler('jobs.db')

# Schedule periodic retraining
schedule_id = scheduler.create_schedule(
    name='daily-retrain',
    job_type='training',
    config={'training_config': config.to_dict()},
    schedule_expr='0 2 * * *',  # Cron: daily at 2am
    priority=3,
    timezone='UTC'
)

# List schedules
schedules = scheduler.list_schedules()

# Trigger manually
scheduler.trigger_schedule(schedule_id)
```

#### Data Structure: `Job`

```python
@dataclass
class Job:
    job_id: int
    job_type: Literal['training', 'evaluation', 'export', 'retraining']
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled']
    priority: int                  # 1-10 (higher = more urgent)
    config: Dict[str, Any]         # Job-specific config
    created_at: str                # ISO 8601 timestamp
    started_at: Optional[str]      # When worker claimed job
    completed_at: Optional[str]    # When job finished
    error_message: Optional[str]   # Error if failed
    worker_id: Optional[str]       # Worker that executed
    retry_count: int               # Retries so far
    max_retries: int               # Max allowed retries
```

---

### ExportBundle

**Location:** `utils.training.export_utilities`

Production-ready export bundles with inference scripts, configs, and deployment artifacts.

#### Function: `create_export_bundle`

```python
from utils.training.export_utilities import create_export_bundle

export_dir = create_export_bundle(
    model=trained_model,
    config=model_config,
    task_spec=task_spec,
    training_config=training_config,
    output_dir='./exports',
    formats=['onnx', 'torchscript', 'pytorch']
)

# Generated structure:
# exports/model_<timestamp>/
# ├── artifacts/
# │   ├── model.onnx
# │   ├── model.torchscript.pt
# │   └── model.pytorch.pt
# ├── configs/
# │   ├── task_spec.json
# │   ├── training_config.json
# │   └── torchserve_config.json
# ├── inference.py
# ├── README.md
# └── requirements.txt
```

#### Generated Artifacts

| File | Purpose |
|------|---------|
| `model.onnx` | ONNX format for cross-platform inference |
| `model.torchscript.pt` | TorchScript for production deployment |
| `model.pytorch.pt` | PyTorch state dict |
| `inference.py` | Standalone inference script |
| `task_spec.json` | Task configuration (modality, preprocessing) |
| `training_config.json` | Training hyperparameters |
| `torchserve_config.json` | TorchServe deployment config |
| `README.md` | Quick-start guide |
| `Dockerfile` | Container deployment |
| `requirements.txt` | Runtime dependencies |

---

### RetrainingTriggers

**Location:** `utils.training.retraining_triggers`

Automatic retraining triggers based on performance metrics and data drift.

#### Class: `RetrainingTrigger`

```python
from utils.training.retraining_triggers import RetrainingTrigger

trigger = RetrainingTrigger(
    monitor_metric='val_loss',
    degradation_threshold=0.1,  # 10% increase triggers retrain
    min_days_between_retrains=7,
    drift_threshold=0.2
)

# Check if retraining needed
should_retrain = trigger.should_retrain(
    current_metrics={'val_loss': 0.42},
    reference_metrics={'val_loss': 0.38},
    drift_score=0.15
)

if should_retrain:
    # Submit retraining job
    manager.submit_job(job_type='retraining', ...)
```

---

## Data Loading

### UniversalDataModule

**Location:** `utils.training.dataset_utilities`

Universal data module supporting text, vision, and custom datasets.

#### Class: `UniversalDataModule`

```python
from utils.training.dataset_utilities import UniversalDataModule

module = UniversalDataModule(
    task_spec=task_spec,
    batch_size=32,
    num_workers=4,
    seed=42
)

# Load from HuggingFace
train_loader, val_loader = module.load_huggingface(
    dataset_name='wikitext',
    dataset_config='wikitext-103-v1',
    train_split='train',
    validation_split='validation',
    max_samples=None
)

# Load from local file
train_loader, val_loader = module.load_from_file(
    file_path='data.jsonl',
    validation_split=0.1
)
```

---

## Utilities

### MetricsTracker

**Location:** `utils.training.metrics_tracker`

Legacy metrics tracking with W&B integration (maintained for backward compatibility).

```python
from utils.training.metrics_tracker import MetricsTracker

tracker = MetricsTracker(use_wandb=True)

# Log epoch-level metrics
tracker.log_epoch(
    epoch=5,
    train_metrics={'loss': 0.42},
    val_metrics={'loss': 0.38},
    learning_rate=1e-4,
    gradient_norm=0.5
)

# Get summary
summary = tracker.get_summary()
print(summary[['epoch', 'train_loss', 'val_loss']])
```

### ExperimentDB

**Location:** `utils.training.experiment_db`

SQLite-based experiment tracking for local tracking (alternative to W&B).

```python
from utils.training.experiment_db import ExperimentDB

db = ExperimentDB('experiments.db')

# Log run
run_id = db.log_run(
    run_name='baseline-v1',
    config=config.to_dict(),
    notes='Initial baseline'
)

# Log metrics
db.log_metric(run_id, 'train/loss', 0.42, epoch=5)
db.log_metric(run_id, 'val/loss', 0.38, epoch=5)

# Find best
best = db.get_best_run('val/loss', mode='min')
print(f"Best: {best['run_name']} (loss={best['best_value']:.4f})")
```

### SeedManager

**Location:** `utils.training.seed_manager`

Reproducibility control with fast and deterministic modes.

```python
from utils.training.seed_manager import set_random_seed

# Fast mode (default): 20% faster, minor GPU non-determinism
set_random_seed(42, deterministic=False)

# Deterministic mode: Bit-exact reproducibility, 5-10% slower
set_random_seed(42, deterministic=True)

# Seeds: Python random, NumPy, PyTorch CPU, PyTorch GPU, DataLoader workers
```

---

## Common Workflows

### Workflow 1: Simple Training Loop

```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig, TrainingConfigBuilder
from utils.training.task_spec import TaskSpec

# 1. Create configuration
config = TrainingConfigBuilder.baseline().build()

# 2. Create task spec
task_spec = TaskSpec.language_modeling(name='wikitext')

# 3. Initialize trainer
trainer = Trainer(
    model=model,
    config=model_config,
    training_config=config,
    task_spec=task_spec
)

# 4. Train
results = trainer.train(train_data, val_data)

print(f"Best val loss: {results['best_val_loss']:.4f}")
```

### Workflow 2: Custom Loss Strategy

```python
from utils.training.engine import LossStrategyRegistry

# Register custom loss
registry = LossStrategyRegistry()

class CustomWeightedLoss:
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def compute_loss(self, logits, labels, inputs=None):
        return F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            reduction='mean'
        )

    def validate_inputs(self, logits, labels):
        assert logits.ndim == 2
        assert labels.ndim == 1

registry.register('weighted-ce', CustomWeightedLoss(weights))

# Use in training
strategy = registry.get('weighted-ce')
```

### Workflow 3: Model Registry & Versioning

```python
from utils.training.model_registry import ModelRegistry

registry = ModelRegistry('models.db')

# After training
model_id = registry.register_model(
    name='transformer-gpt',
    version='1.0.0',
    checkpoint_path=checkpoint_path,
    task_type='language_modeling',
    metrics=results['final_metrics'],
    config_hash=hashlib.sha256(
        json.dumps(asdict(model_config)).encode()
    ).hexdigest()
)

# Promote to production
registry.promote_model(model_id, 'production')

# Load for inference
prod_model = registry.get_model(tag='production')
```

### Workflow 4: Export Bundle

```python
from utils.training.export_utilities import create_export_bundle

export_dir = create_export_bundle(
    model=model,
    config=model_config,
    task_spec=task_spec,
    training_config=config,
    formats=['onnx', 'torchscript']
)

# Deploy with Docker
import subprocess
subprocess.run(['docker', 'build', '-t', 'model:v1', export_dir])
subprocess.run(['docker', 'run', '-p', '8080:8080', 'model:v1'])
```

### Workflow 5: Scheduled Retraining

```python
from utils.training.job_queue import JobManager, TrainingScheduler

manager = JobManager('jobs.db')
scheduler = TrainingScheduler('jobs.db')

# Schedule daily retraining
scheduler.create_schedule(
    name='daily-retrain',
    job_type='retraining',
    config={'training_config': config.to_dict()},
    schedule_expr='0 2 * * *'  # 2am daily
)

# Worker process
executor = JobExecutor(manager, worker_id='worker-1')
executor.run_worker(max_jobs=10)
```

---

## API Stability & Deprecation

### Stability Guarantees

**Green (Stable):** Public APIs in v4.0+ are stable.
- `Trainer`, `TrainingConfig`, `CheckpointManager`
- `LossStrategy` and implementations
- Engine components (`GradientMonitor`, `GradientAccumulator`)

**Yellow (Evolving):** May have minor changes in v4.1+
- `MetricsEngine` (expanding alert types)
- `JobQueue` (adding distributed worker support)

**Red (Experimental):** May change substantially
- Custom hook protocols
- Internal metrics formats

### Deprecation Policy

- **2 major versions** before removal
- Warnings logged when using deprecated APIs
- Migration guides provided
- Example: `test_fine_tuning()` deprecated in v4.0, removal in v6.0

### Version Numbering

`v{major}.{minor}.{patch}`

- **Major:** Breaking API changes
- **Minor:** New features, backward compatible
- **Patch:** Bug fixes

---

## Index

- [CheckpointManager](#checkpointmanager) - State persistence
- [LossStrategy](#lossstrategy) - Task-specific loss computation (5 implementations)
- [GradientMonitor](#gradientmonitor) - Gradient health checks
- [GradientAccumulator](#gradientaccumulator) - Gradient accumulation management
- [DataLoaderFactory](#dataloaderfactory) - Optimized DataLoader creation
- [TrainingLoop & ValidationLoop](#training-and-validation-loops) - Epoch execution
- [MetricsEngine](#metricsengine) - Comprehensive metrics + drift detection
- [Trainer](#trainer) - High-level orchestrator
- [TrainingConfig](#trainingconfig) - Complete training configuration
- [TrainingConfigBuilder](#trainingconfigbuilder) - Fluent configuration API
- [TaskSpec](#taskspec) - Task specification
- [ModelRegistry](#modelregistry) - Model versioning and deployment
- [JobQueue & Scheduler](#jobqueue) - Automated workflow orchestration
- [ExportBundle](#export-bundle) - Production deployment artifacts
- [RetrainingTriggers](#retraining-triggers) - Automatic retraining
- [UniversalDataModule](#universaldatamodule) - Multi-modal data loading
- [MetricsTracker](#metricstracker) - Legacy metrics tracking
- [ExperimentDB](#experimentdb) - Local experiment tracking
- [SeedManager](#seedmanager) - Reproducibility control

---

**Last Updated:** 2025-11-20
**Maintainer:** MLOps & Architecture Teams
**License:** Apache 2.0
