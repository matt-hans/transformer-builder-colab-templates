# API Reference - Training Engine v4.0

**Version:** 4.0.0
**Last Updated:** 2025-11-20
**Status:** Production

This document provides comprehensive API documentation for the modular training engine introduced in v4.0. The engine refactors monolithic utilities into composable components for improved maintainability, testability, and extensibility.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
   - [CheckpointManager](#checkpointmanager)
   - [LossStrategy](#lossstrategy)
   - [GradientMonitor](#gradientmonitor)
   - [GradientAccumulator](#gradientaccumulator)
   - [DataLoaderFactory](#dataloaderfactory)
   - [MetricsEngine](#metricsengine)
   - [TrainingLoop & ValidationLoop](#training-and-validation-loops)
   - [Trainer](#trainer)
3. [Production Features](#production-features)
   - [ModelRegistry](#modelregistry)
   - [JobQueue & Scheduler](#jobqueue-and-scheduler)
   - [Export Bundle](#export-bundle)
   - [Retraining Triggers](#retraining-triggers)
   - [Flash Attention](#flash-attention)
4. [Configuration System](#configuration-system)
   - [TrainingConfig](#trainingconfig)
   - [TrainingConfigBuilder](#trainingconfigbuilder)
5. [Data Loading](#data-loading)
   - [TaskSpec](#taskspec)
   - [CollatorRegistry](#collatorregistry)
   - [UniversalDataModule](#universaldatamodule)
6. [Utilities](#utilities)
   - [SeedManager](#seedmanager)
   - [ExperimentDB](#experimentdb)
   - [Dashboard](#dashboard)
7. [Migration from v3.x](#migration-from-v3x)

---

## Overview

The v4.0 training engine introduces a **modular architecture** based on design patterns:

- **Strategy Pattern**: Task-specific loss computation (`LossStrategy`)
- **Registry Pattern**: Extensible collator and loss strategy registration
- **Builder Pattern**: Fluent configuration API (`TrainingConfigBuilder`)
- **Protocol-Based Design**: Framework-agnostic interfaces (no hard dependencies)
- **Composition over Inheritance**: Components delegate to specialized modules

**Key Benefits:**
- **Testability**: Each component can be unit tested in isolation
- **Extensibility**: Register custom strategies without modifying core code
- **Type Safety**: Protocol-based interfaces with full type hints
- **Performance**: <5ms overhead per training step
- **Production-Ready**: Built-in health checks, drift detection, job scheduling

---

## Core Components

### CheckpointManager

**Module:** `utils.training.engine.checkpoint`

Comprehensive checkpoint management for training state persistence and recovery.

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
    drive_backup=False,
    drive_backup_path=None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | `str` | Required | Directory to save checkpoints |
| `keep_best_k` | `int` | `3` | Number of best checkpoints to keep (by monitor metric) |
| `keep_last_n` | `int` | `5` | Number of most recent checkpoints to keep |
| `monitor` | `str` | `'val_loss'` | Metric to monitor (e.g., 'val_loss', 'val_accuracy') |
| `mode` | `Literal['min', 'max']` | `'min'` | 'min' or 'max' for monitored metric |
| `save_interval_epochs` | `int` | `1` | Save checkpoint every N epochs |
| `drive_backup` | `bool` | `False` | Enable Google Drive backup (Colab only) |
| `drive_backup_path` | `Optional[str]` | `None` | Path in Drive for backups |

**Methods:**

#### `save()`

Save checkpoint with full training state.

```python
checkpoint_path = manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=5,
    metrics={'val_loss': 0.38, 'train_loss': 0.42},
    global_step=1000,
    custom_state={'strategy_config': {...}}
)
# Returns: Path to saved checkpoint file
```

**Parameters:**
- `model` (`nn.Module`): PyTorch model
- `optimizer` (`torch.optim.Optimizer`): Optimizer instance
- `scheduler` (`Any`): Learning rate scheduler
- `epoch` (`int`): Current epoch number
- `metrics` (`Dict[str, float]`): Dictionary of metrics (must include monitor metric)
- `global_step` (`Optional[int]`): Total batches processed (optional)
- `custom_state` (`Optional[Dict[str, Any]]`): Additional state to save

**Returns:** `Path` to saved checkpoint file

**Raises:** `ValueError` if monitor metric not in metrics dict

**Saved State:**
```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'metrics': dict,
    'monitor': str,
    'monitor_value': float,
    'rng_state': dict,  # Python, NumPy, PyTorch, CUDA
    'timestamp': str,
    'git_commit': Optional[str],
    'custom_state': dict
}
```

#### `load()`

Load checkpoint and return state dictionary.

```python
state = manager.load(checkpoint_path=None)  # None = load best
model.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])
scheduler.load_state_dict(state['scheduler_state_dict'])

# Resume training
start_epoch = state['epoch'] + 1
```

**Parameters:**
- `checkpoint_path` (`Optional[Path]`): Path to checkpoint (None for best checkpoint)

**Returns:** `Dict[str, Any]` with checkpoint state (see Saved State above)

**Raises:**
- `FileNotFoundError`: If no checkpoints found
- `RuntimeError`: If checkpoint file is corrupted

**Features:**
- Automatic RNG state restoration for reproducibility
- Corruption detection with recovery guidance
- Validation of checkpoint structure

#### `get_best()`

Get path to best checkpoint (by monitor metric).

```python
best_path = manager.get_best()
# Returns: Path | None
```

#### `list_checkpoints()`

List all checkpoints sorted by epoch (descending).

```python
checkpoints = manager.list_checkpoints()
# Returns: List[CheckpointMetadata]

for ckpt in checkpoints:
    print(f"Epoch {ckpt.epoch}: {ckpt.best_metric:.4f}")
```

**Data Class: `CheckpointMetadata`**

```python
@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    global_step: int
    best_metric: float
    timestamp: str
    git_commit: Optional[str] = None
    metrics: Optional[str] = None  # JSON string
    config: Optional[str] = None   # JSON string
```

**Example: Complete Workflow**

```python
from utils.training.engine import CheckpointManager

# Setup
manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    keep_best_k=3,
    keep_last_n=5,
    monitor='val_loss',
    mode='min'
)

# Training loop
for epoch in range(n_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # Save checkpoint
    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics={'val_loss': val_loss, 'train_loss': train_loss},
        global_step=epoch * len(train_loader)
    )

    print(f"✓ Checkpoint saved: {path.name}")

# Resume training
state = manager.load()  # Load best checkpoint
model.load_state_dict(state['model_state_dict'])
optimizer.load_state_dict(state['optimizer_state_dict'])
start_epoch = state['epoch'] + 1
```

**Retention Policy:**

Checkpoints are automatically cleaned up based on `keep_best_k` and `keep_last_n`:

- **Best K**: Keep top K checkpoints by monitor metric
- **Last N**: Keep N most recent checkpoints
- **Union**: A checkpoint is kept if it's in EITHER best K or last N

Example with `keep_best_k=3`, `keep_last_n=5`:
- Keeps best 3 checkpoints by val_loss
- Keeps 5 most recent checkpoints
- Total kept: 5-8 checkpoints (overlap possible)

**Google Drive Backup (Colab):**

```python
# Enable Drive backup for Colab persistence
from google.colab import drive
drive.mount('/content/drive')

manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    drive_backup=True,
    drive_backup_path='/content/drive/MyDrive/model_checkpoints'
)

# Checkpoints automatically backed up to Drive after each save
```

---

### LossStrategy

**Module:** `utils.training.engine.loss`

Task-specific loss computation using the Strategy Pattern for maximum flexibility.

#### Protocol: `LossStrategy`

```python
from typing import Protocol
from utils.training.engine.loss import LossInputs

class LossStrategy(Protocol):
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        """
        Compute task-specific loss.

        Args:
            inputs: Type-safe dictionary of loss computation inputs

        Returns:
            Scalar loss tensor (mean reduction)
        """
        ...
```

#### TypedDict: `LossInputs`

```python
from typing import TypedDict, Optional
import torch

class LossInputs(TypedDict, total=False):
    """
    Type-safe container for loss computation inputs.

    Attributes:
        logits: Model output logits [batch, seq_len, vocab_size] or [batch, num_classes]
        labels: Target labels [batch, seq_len] or [batch]
        attention_mask: Optional attention mask [batch, seq_len]
        pad_token_id: Token ID to exclude from loss (default: 0)
        pixel_values: For vision tasks [batch, channels, height, width]
        class_weights: Optional class weights [num_classes]
    """
    logits: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pad_token_id: Optional[int]
    pixel_values: Optional[torch.Tensor]
    class_weights: Optional[torch.Tensor]
```

#### Built-in Strategies

##### `LanguageModelingLoss`

Loss for causal language modeling (next-token prediction).

```python
from utils.training.engine import LanguageModelingLoss

strategy = LanguageModelingLoss()
loss = strategy.compute_loss({
    'logits': logits,  # [batch, seq_len, vocab_size]
    'labels': labels,  # [batch, seq_len]
    'pad_token_id': 0
})
```

**Features:**
- Token shifting for autoregressive modeling
- Padding exclusion via `ignore_index`
- Suitable for GPT-style decoder-only transformers

**Implementation Details:**
```python
# Shift tokens: predict next token
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

# Cross-entropy with padding exclusion
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=pad_token_id,
    reduction='mean'
)
```

##### `ClassificationLoss`

Loss for classification tasks (no token shifting).

```python
from utils.training.engine import ClassificationLoss

strategy = ClassificationLoss()
loss = strategy.compute_loss({
    'logits': logits,  # [batch, num_classes] or [batch, seq, classes]
    'labels': labels,  # [batch] or [batch, seq]
    'class_weights': weights  # [num_classes] (optional)
})
```

**Features:**
- Supports class weights for imbalanced datasets
- Handles sequence classification (takes last token)
- Multi-class classification support

##### `VisionLoss`

Loss for vision tasks (image classification, segmentation).

```python
from utils.training.engine import VisionLoss

strategy = VisionLoss()
loss = strategy.compute_loss({
    'logits': logits,  # [batch, classes] or [batch, classes, H, W]
    'labels': labels   # [batch] or [batch, H, W]
})
```

**Supported Tasks:**
- Image classification: `[batch, num_classes]` logits
- Semantic segmentation: `[batch, num_classes, H, W]` logits

##### `PEFTAwareLoss`

Wrapper for PEFT/LoRA models (Parameter-Efficient Fine-Tuning).

```python
from utils.training.engine import PEFTAwareLoss, LanguageModelingLoss

base_strategy = LanguageModelingLoss()
peft_strategy = PEFTAwareLoss(base_strategy, model)

loss = peft_strategy.compute_loss(inputs)
```

**Features:**
- Verifies PEFT setup (frozen base model + trainable adapters)
- Warns if all parameters are trainable
- Ensures gradients only flow to adapter parameters

##### `QuantizationSafeLoss`

Wrapper for quantized models (4-bit, 8-bit).

```python
from utils.training.engine import QuantizationSafeLoss, LanguageModelingLoss

base_strategy = LanguageModelingLoss()
quant_strategy = QuantizationSafeLoss(base_strategy)

loss = quant_strategy.compute_loss(inputs)
```

**Features:**
- Detects quantized dtypes (int8, uint8, qint8)
- Auto-converts FP16 logits to FP32 for stability
- Prevents numerical issues in quantized models

#### LossStrategyRegistry

Registry for loss strategy lookup with type safety.

```python
from utils.training.engine import LossStrategyRegistry, get_loss_strategy

# Get strategy by task type
strategy = get_loss_strategy("language_modeling")
loss = strategy.compute_loss(inputs)

# List available strategies
available = LossStrategyRegistry.list_available()
# Returns: ['language_modeling', 'causal_lm', 'classification',
#           'vision_classification', 'segmentation']
```

**Register Custom Strategy:**

```python
from utils.training.engine import LossStrategyRegistry, LossInputs
import torch.nn.functional as F

@LossStrategyRegistry.register("custom_task")
class CustomLoss:
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        logits = inputs['logits']
        labels = inputs['labels']

        # Custom loss logic
        loss = F.cross_entropy(logits, labels)
        return loss

# Use custom strategy
strategy = get_loss_strategy("custom_task")
```

**Typo Detection:**

```python
# Typo in task type
strategy = get_loss_strategy("languag_modeling")
# Raises: ValueError: Unknown task_type 'languag_modeling'.
#         Available: language_modeling, classification, ...
#         Did you mean: language_modeling?
```

**Example: Complete Workflow**

```python
from utils.training.engine import get_loss_strategy, ModelOutput

# Auto-select strategy from task type
task_type = "language_modeling"
loss_strategy = get_loss_strategy(task_type)

# Training loop
for batch in train_loader:
    input_ids, labels = batch

    # Forward pass
    logits = model(input_ids)

    # Compute loss
    loss = loss_strategy.compute_loss({
        'logits': logits,
        'labels': labels,
        'pad_token_id': tokenizer.pad_token_id
    })

    # Backward pass
    loss.backward()
    optimizer.step()
```

---

### GradientMonitor

**Module:** `utils.training.engine.gradient_monitor`

Comprehensive gradient health monitoring for training stability.

#### Class: `GradientMonitor`

```python
from utils.training.engine import GradientMonitor

monitor = GradientMonitor(
    vanishing_threshold=1e-7,
    explosion_threshold=10.0,
    max_consecutive_failures=3,
    log_histogram=False,
    max_histogram_samples=200000
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vanishing_threshold` | `float` | `1e-7` | Gradient norm threshold for vanishing detection |
| `explosion_threshold` | `float` | `10.0` | Gradient norm threshold for explosion detection |
| `max_consecutive_failures` | `int` | `3` | Halt training after N consecutive NaN/Inf gradients |
| `log_histogram` | `bool` | `False` | Enable gradient histogram logging to W&B |
| `max_histogram_samples` | `int` | `200000` | Max gradient values to sample for histogram |

**Methods:**

#### `check_gradients()`

Check gradient health and detect anomalies.

```python
# In training loop after loss.backward()
health = monitor.check_gradients(model)

if not health.is_healthy:
    print(f"⚠️ Gradient issues: {health}")
    print(f"Affected layers: {health.affected_layers}")
```

**Returns:** `GradientHealth` dataclass

**Data Class: `GradientHealth`**

```python
@dataclass
class GradientHealth:
    has_nan: bool                    # True if any gradients are NaN
    has_inf: bool                    # True if any gradients are Inf
    max_norm: float                  # Maximum gradient norm
    min_norm: float                  # Minimum gradient norm
    affected_layers: List[str]       # Layers with NaN/Inf
    is_healthy: bool                 # Overall health status
    vanishing_layers: List[str]      # Layers below vanishing threshold
    exploding_layers: List[str]      # Layers above explosion threshold
```

**Example Output:**

```python
# Healthy gradients
GradientHealth(
    has_nan=False,
    has_inf=False,
    max_norm=0.523,
    min_norm=0.001,
    affected_layers=[],
    is_healthy=True,
    vanishing_layers=[],
    exploding_layers=[]
)
# __str__: "✓ Healthy (norm: 1.00e-03 - 5.23e-01)"

# Unhealthy gradients
GradientHealth(
    has_nan=True,
    has_inf=False,
    max_norm=float('inf'),
    min_norm=0.0,
    affected_layers=['transformer.layers.11.attn.q_proj (NaN)'],
    is_healthy=False,
    vanishing_layers=['transformer.layers.0.mlp.fc1'],
    exploding_layers=[]
)
# __str__: "⚠️ Issues: NaN in 1 layers, Vanishing in 1 layers"
```

**Raises:** `RuntimeError` if consecutive failures exceed `max_consecutive_failures`

**Error Message Example:**

```
RuntimeError: Training unstable: 3 consecutive gradient failures. Issues: NaN=True, Inf=False
Affected layers: ['transformer.layers.11.attn.q_proj (NaN)', ...]

Remediation steps:
1. Lower learning rate (try 0.1x current LR)
2. Enable gradient clipping (max_norm=1.0)
3. Check loss computation for division by zero
4. Verify data preprocessing (normalize inputs)
5. Try mixed precision training (AMP)
```

#### `compute_gradient_norm()`

Compute L2 norm of gradients across all parameters.

```python
loss.backward()
grad_norm = monitor.compute_gradient_norm(model)
print(f"Gradient norm: {grad_norm:.4f}")
```

**Returns:** `float` L2 norm (0.0 if no gradients exist)

**Formula:** `sqrt(sum(||grad_i||_2^2))` for all parameters

#### `log_gradient_distribution()`

Log per-parameter gradient norms and histogram to tracker.

```python
from utils.training.engine import GradientMonitor, MetricsEngine

monitor = GradientMonitor(log_histogram=True)
tracker = MetricsEngine(use_wandb=True)

# In training loop after backward()
monitor.log_gradient_distribution(
    model=model,
    tracker=tracker,
    step=global_step
)
```

**Logged Metrics:**
- `gradients/{layer_name}/norm`: Per-layer gradient norm
- `gradients/distribution`: Histogram of sampled gradient values (if `log_histogram=True`)

#### `reset_failure_counts()`

Reset failure counters (for checkpoint resume).

```python
monitor.reset_failure_counts()
```

**Example: Complete Workflow**

```python
from utils.training.engine import GradientMonitor

monitor = GradientMonitor(
    vanishing_threshold=1e-7,
    explosion_threshold=10.0,
    max_consecutive_failures=3
)

# Training loop
for batch in train_loader:
    optimizer.zero_grad()

    # Forward pass
    logits = model(inputs)
    loss = loss_fn(logits, labels)

    # Backward pass
    loss.backward()

    # Check gradient health
    health = monitor.check_gradients(model)

    if not health.is_healthy:
        print(f"⚠️ Gradient issues detected: {health}")

        if health.has_nan:
            print("NaN gradients in layers:")
            for layer in health.affected_layers:
                print(f"  - {layer}")

            # Remediation: lower learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

            # Reset failure counter
            monitor.reset_failure_counts()
            continue

    # Compute gradient norm for logging
    grad_norm = monitor.compute_gradient_norm(model)
    tracker.log_scalar('train/gradient_norm', grad_norm, step=step)

    # Optimizer step
    optimizer.step()
```

**Convenience Function:**

```python
from utils.training.engine import check_gradient_health

# Quick health check without creating monitor instance
health = check_gradient_health(
    model=model,
    vanishing_threshold=1e-7,
    explosion_threshold=10.0
)

if not health.is_healthy:
    print(f"Issues: {health.affected_layers}")
```

---

### GradientAccumulator

**Module:** `utils.training.engine.gradient_accumulator`

Unified gradient accumulation manager compatible with both manual and PyTorch Lightning workflows.

#### Class: `GradientAccumulator`

```python
from utils.training.engine import GradientAccumulator

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    max_grad_norm=1.0,
    scaler=None,
    batch_size=8,
    trainer=None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | `torch.optim.Optimizer` | Required | PyTorch optimizer for parameter updates |
| `accumulation_steps` | `int` | `1` | Number of batches to accumulate (1 = no accumulation) |
| `max_grad_norm` | `Optional[float]` | `1.0` | Maximum gradient norm for clipping (None to disable) |
| `scaler` | `Optional[Any]` | `None` | GradScaler for AMP training |
| `batch_size` | `int` | `1` | Physical batch size for effective batch size calculation |
| `trainer` | `Optional[Any]` | `None` | PyTorch Lightning Trainer instance |

**Raises:** `ValueError` if both `accumulation_steps > 1` and `trainer.accumulate_grad_batches > 1`

**Properties:**

- `is_lightning_managed` (`bool`): True if Lightning trainer controls accumulation
- `effective_batch_size` (`int`): Physical batch size × accumulation steps
- `effective_step` (`int`): Number of optimizer.step() calls (for metrics logging)
- `stats` (`AccumulationStats`): Current accumulation statistics

**Data Class: `AccumulationStats`**

```python
@dataclass
class AccumulationStats:
    total_steps: int              # Total backward() calls
    optimizer_steps: int          # Total optimizer.step() calls
    current_accumulation: int     # Position in accumulation window (0 to steps-1)
    effective_batch_size: int     # Physical batch × accumulation steps
    is_accumulating: bool         # True if currently accumulating gradients
    last_grad_norm: float         # Last computed gradient norm (pre-clip)
```

**Methods:**

#### `accumulate()`

Accumulate gradients and optionally step optimizer.

```python
for batch_idx, batch in enumerate(dataloader):
    loss = model(batch)

    # Accumulate gradients
    should_step = accumulator.accumulate(
        loss=loss,
        model=model,
        is_final_batch=(batch_idx == len(dataloader) - 1)
    )

    if should_step:
        print(f"Optimizer stepped at batch {batch_idx}")
        grad_norm = accumulator.stats.last_grad_norm
        print(f"Gradient norm: {grad_norm:.4f}")
```

**Parameters:**
- `loss` (`torch.Tensor`): Loss tensor from forward pass (not scaled)
- `model` (`nn.Module`): Model to compute gradients for
- `is_final_batch` (`bool`): True if this is the final batch in epoch

**Returns:** `bool` - True if optimizer.step() was called, False if still accumulating

**Automatic Steps:**
1. Scale loss by `1/accumulation_steps`
2. Compute gradients via `loss.backward()` (AMP-aware)
3. Check if accumulation window is complete
4. If complete:
   - Unscale gradients (AMP)
   - Compute pre-clip gradient norm
   - Clip gradients to `max_grad_norm`
   - Step optimizer (with overflow check for AMP)
   - Zero gradients for next cycle
   - Update effective step counter

#### `reset_epoch()`

Reset accumulation state for new epoch.

```python
# At end of each epoch
accumulator.reset_epoch()
```

**Warning:** Logs warning if accumulated gradients are lost (non-zero accumulation counter)

#### `state_dict()` / `load_state_dict()`

Checkpoint accumulator state for resume.

```python
# Save
accumulator_state = accumulator.state_dict()
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'accumulator': accumulator_state
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
accumulator.load_state_dict(checkpoint['accumulator'])
```

**Example: Manual Accumulation**

```python
from utils.training.engine import GradientAccumulator

# Setup
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,  # Effective batch = 4 × physical batch
    max_grad_norm=1.0,
    batch_size=8
)

print(f"Effective batch size: {accumulator.effective_batch_size}")
# Output: Effective batch size: 32

# Training loop
for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        loss = model(batch)

        # Accumulate gradients (handles loss scaling, backward, clipping, step)
        should_step = accumulator.accumulate(
            loss=loss,
            model=model,
            is_final_batch=(batch_idx == len(dataloader) - 1)
        )

        # Log metrics at effective steps
        if should_step:
            tracker.log_scalar(
                'train/loss',
                loss.item(),
                step=accumulator.effective_step  # Aligns with optimizer updates
            )

            # Log gradient norm
            grad_norm = accumulator.stats.last_grad_norm
            tracker.log_scalar('train/gradient_norm', grad_norm,
                             step=accumulator.effective_step)

    # Reset for next epoch
    accumulator.reset_epoch()
```

**Example: Lightning Integration**

```python
import pytorch_lightning as pl
from utils.training.engine import GradientAccumulator

# Lightning trainer with accumulation
trainer = pl.Trainer(
    max_epochs=10,
    accumulate_grad_batches=4  # Lightning manages accumulation
)

# GradientAccumulator detects Lightning and delegates
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,  # Disabled (Lightning manages)
    trainer=trainer
)

print(accumulator.is_lightning_managed)
# Output: True

# accumulator.accumulate() becomes pass-through
# Lightning controls when optimizer steps
```

**Conflict Detection:**

```python
# ERROR: Double accumulation
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,  # Manual accumulation
    trainer=trainer  # trainer.accumulate_grad_batches=4
)
# Raises: ValueError: Gradient accumulation conflict detected!
#         Manual accumulation_steps: 4
#         Lightning accumulate_grad_batches: 4
#
#         Resolution options:
#         1. Set accumulation_steps=1 (let Lightning manage)
#         2. Set trainer.accumulate_grad_batches=1 (use manual)
#         3. Remove trainer parameter (disable Lightning integration)
```

**Performance:**

- **Overhead**: <2ms per batch (loss scaling + counter updates)
- **Memory**: Constant (no additional buffers)
- **AMP Support**: Automatic scaler integration
- **Final Batch**: Guaranteed step on last batch of epoch (no gradient loss)

---

### DataLoaderFactory

**Module:** `utils.training.engine.data`

Architecture-agnostic data loading with registry-based collators and reproducibility.

#### Class: `DataLoaderFactory`

```python
from utils.training.engine import DataLoaderFactory, DataLoaderConfig

factory = DataLoaderFactory()
train_loader = factory.create(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    task_spec=task_spec,
    random_seed=42
)
```

**Data Class: `DataLoaderConfig`**

```python
@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    collator_name: Optional[str] = None
    random_seed: int = 42
```

**Methods:**

#### `create()`

Create DataLoader with automatic collator selection.

```python
train_loader = factory.create(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    task_spec=task_spec,  # Auto-selects collator by modality
    collator_name=None,   # Override auto-selection
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    random_seed=42
)
```

**Parameters:**
- `dataset` (`Dataset | HFDataset`): PyTorch or HuggingFace dataset
- `batch_size` (`int`): Batch size
- `shuffle` (`bool`): Shuffle dataset
- `task_spec` (`Optional[TaskSpec]`): For auto-collator selection
- `collator_name` (`Optional[str]`): Explicit collator name
- `num_workers` (`int`): DataLoader worker processes
- `pin_memory` (`bool`): Pin memory for GPU transfer
- `drop_last` (`bool`): Drop incomplete last batch
- `random_seed` (`int`): Seed for reproducibility

**Returns:** `DataLoader` instance

**Features:**
- Auto-collator selection from `TaskSpec.modality`
- Worker seeding for reproducibility
- Automatic pin_memory for CUDA
- Generator-based shuffling (deterministic)

**Example:**

```python
from utils.training.engine import DataLoaderFactory
from utils.training.task_spec import TaskSpec

factory = DataLoaderFactory()

# Text task (auto-selects LanguageModelingDataCollator)
task_spec = TaskSpec(
    name='text-task',
    modality='text',
    max_seq_len=512
)
train_loader = factory.create(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    task_spec=task_spec
)

# Vision task (auto-selects VisionDataCollator)
task_spec = TaskSpec.vision_tiny()
val_loader = factory.create(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False,
    task_spec=task_spec
)
```

---

### CollatorRegistry

**Module:** `utils.training.engine.data`

Registry for task-specific data collators with decorator-based registration.

#### Class: `CollatorRegistry`

```python
from utils.training.engine import CollatorRegistry

registry = CollatorRegistry.get_instance()
```

**Methods:**

#### `register()`

Decorator to register a collator factory.

```python
@registry.register('custom_text', modality='text', description='Custom text collator')
def create_custom_collator(tokenizer, **kwargs):
    return CustomCollator(tokenizer)

# Use custom collator
collator = registry.get_collator(collator_name='custom_text', tokenizer=tokenizer)
```

#### `get_collator()`

Get collator instance by TaskSpec or name.

```python
# Auto-select from TaskSpec
collator = registry.get_collator(task_spec=task_spec)

# Explicit name
collator = registry.get_collator(collator_name='text', tokenizer=tokenizer)
```

**Parameters:**
- `task_spec` (`Optional[TaskSpec]`): For auto-selection by modality
- `collator_name` (`Optional[str]`): Explicit collator name
- `**kwargs`: Arguments passed to collator factory

**Returns:** Collator instance (callable)

**Built-in Collators:**

1. **Text Collator** (`'text'`, modality=`'text'`)
   - Uses `LanguageModelingDataCollator`
   - Dynamic padding
   - Configurable padding side

2. **Vision Collator** (`'vision'`, modality=`'vision'`)
   - Uses `VisionDataCollator`
   - ImageNet normalization (default)
   - Custom mean/std support

3. **Default Collator** (`'default'`, modality=`'any'`)
   - Falls back to `transformers.default_data_collator`

**Example:**

```python
from utils.training.engine import CollatorRegistry

registry = CollatorRegistry.get_instance()

# List available collators
collators = registry.list_collators()
for info in collators:
    print(f"{info.name} ({info.modality}): {info.description}")

# Output:
# text (text): Text collator with dynamic padding
# vision (vision): Vision collator with normalization
# default (any): HuggingFace default collator
```

---

### MetricsEngine

**Module:** `utils.training.engine.metrics`

Comprehensive metrics tracking with drift detection, confidence tracking, and performance alerts.

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

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | `bool` | `True` | Enable W&B logging |
| `gradient_accumulation_steps` | `int` | `1` | For effective step tracking |
| `drift_threshold_warning` | `float` | `0.1` | JS divergence threshold for warning status |
| `drift_threshold_critical` | `float` | `0.2` | JS divergence threshold for critical status |
| `alert_config` | `Optional[AlertConfig]` | `None` | Alert configuration |
| `alert_callbacks` | `Optional[List[AlertCallback]]` | `None` | Custom alert callbacks |
| `experiment_db` | `Optional[Any]` | `None` | ExperimentDB instance |
| `run_id` | `Optional[int]` | `None` | Run ID for ExperimentDB |

**Data Class: `AlertConfig`**

```python
@dataclass
class AlertConfig:
    val_loss_spike_threshold: float = 0.2  # 20% spike
    accuracy_drop_threshold: float = 0.05  # 5% drop
    gradient_explosion_threshold: float = 10.0
    enable_email_alerts: bool = False
    enable_slack_alerts: bool = False
```

**Methods:**

#### `log_epoch()`

Log epoch-level metrics with optional drift detection.

```python
drift_metrics = engine.log_epoch(
    epoch=5,
    train_metrics={'loss': 0.42, 'accuracy': 0.85},
    val_metrics={'loss': 0.38, 'accuracy': 0.87},
    learning_rate=1e-4,
    gradient_norm=0.5,
    epoch_duration=120.5,
    reference_profile=ref_profile,  # Optional
    current_profile=curr_profile
)

if drift_metrics and drift_metrics.status != 'healthy':
    print(f"⚠️ Drift detected: {drift_metrics.status}")
```

**Parameters:**
- `epoch` (`int`): Current epoch number
- `train_metrics` (`Dict[str, float]`): Must include 'loss' and 'accuracy'
- `val_metrics` (`Dict[str, float]`): Must include 'loss' and 'accuracy'
- `learning_rate` (`float`): Current LR from scheduler
- `gradient_norm` (`float`): Maximum gradient norm
- `epoch_duration` (`float`): Time taken (seconds)
- `reference_profile` (`Optional[Dict]`): Reference dataset profile
- `current_profile` (`Optional[Dict]`): Current dataset profile

**Returns:** `Optional[DriftMetrics]` if drift detection performed

**Data Class: `DriftMetrics`**

```python
@dataclass
class DriftMetrics:
    js_divergence: float                         # 0-1, lower is better
    status: Literal['healthy', 'warning', 'critical']
    affected_features: List[str]
    timestamp: str
    details: Dict[str, float] = field(default_factory=dict)
```

**Drift Status Thresholds:**
- **Healthy**: JS divergence < 0.1
- **Warning**: 0.1 ≤ JS divergence < 0.2
- **Critical**: JS divergence ≥ 0.2

#### `log_confidence()`

Log prediction confidence metrics during validation.

```python
engine.log_confidence(logits, labels, step=100)
```

**Data Class: `ConfidenceMetrics`**

```python
@dataclass
class ConfidenceMetrics:
    top1_confidence: float       # Mean confidence of top-1 predictions (0-1)
    top5_confidence: float       # Mean confidence within top-5 predictions
    entropy: float               # Mean prediction entropy (lower = more confident)
    num_samples: int
    calibration_error: Optional[float] = None
```

#### `has_alerts()` / `get_alerts()`

Check for performance alerts.

```python
if engine.has_alerts():
    alerts = engine.get_alerts()
    for alert in alerts:
        print(f"🚨 {alert['type']}: {alert['message']}")
```

**Alert Types:**
- **val_loss_spike**: Validation loss increased by > threshold
- **accuracy_drop**: Accuracy decreased by > threshold
- **gradient_explosion**: Gradient norm exceeded threshold

**Example: Complete Workflow**

```python
from utils.training.engine import MetricsEngine, AlertConfig
from utils.training.drift_metrics import profile_dataset

# Setup
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

# Profile reference dataset (once at start)
ref_profile = profile_dataset(train_dataset, task_spec)

# Training loop
for epoch in range(n_epochs):
    # Train
    train_metrics = train_epoch(model, train_loader)

    # Validate
    val_metrics = validate(model, val_loader)

    # Profile current dataset (optional)
    curr_profile = profile_dataset(val_dataset, task_spec)

    # Log metrics with drift detection
    drift = engine.log_epoch(
        epoch=epoch,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        learning_rate=scheduler.get_last_lr()[0],
        gradient_norm=grad_norm,
        epoch_duration=epoch_time,
        reference_profile=ref_profile,
        current_profile=curr_profile
    )

    # Check drift
    if drift and drift.status == 'critical':
        print(f"🚨 Critical drift detected!")
        print(f"  JS divergence: {drift.js_divergence:.4f}")
        print(f"  Affected features: {drift.affected_features}")

        # Remediation: update reference profile
        ref_profile = curr_profile

    # Check alerts
    if engine.has_alerts():
        alerts = engine.get_alerts()
        for alert in alerts:
            print(f"⚠️ {alert['type']}: {alert['message']}")

# Export metrics
summary_df = engine.get_summary()
summary_df.to_csv('training_metrics.csv', index=False)
```

---

### Training and Validation Loops

**Module:** `utils.training.engine.loop`

Extracted training and validation epoch execution logic.

#### Class: `TrainingLoop`

```python
from utils.training.engine import TrainingLoop

loop = TrainingLoop(
    model=model,
    optimizer=optimizer,
    loss_strategy=loss_strategy,
    gradient_accumulator=accumulator,
    gradient_monitor=monitor,
    metrics_tracker=tracker,
    device='cuda'
)
```

**Methods:**

#### `run_epoch()`

Execute one training epoch.

```python
result = loop.run_epoch(
    dataloader=train_loader,
    epoch=5,
    scheduler=scheduler
)

print(f"Epoch {result.epoch}: Loss={result.avg_loss:.4f}, Acc={result.avg_accuracy:.4f}")
```

**Returns:** `EpochResult` dataclass

**Data Class: `EpochResult`**

```python
@dataclass
class EpochResult:
    epoch: int
    avg_loss: float
    avg_accuracy: float
    batch_count: int
    sample_count: int
    duration: float
    learning_rate: float
    max_gradient_norm: float
```

#### Class: `ValidationLoop`

```python
from utils.training.engine import ValidationLoop

loop = ValidationLoop(
    model=model,
    loss_strategy=loss_strategy,
    metrics_tracker=tracker,
    device='cuda'
)
```

**Methods:**

#### `run_epoch()`

Execute one validation epoch.

```python
result = loop.run_epoch(
    dataloader=val_loader,
    epoch=5
)

print(f"Val Loss: {result.avg_loss:.4f}, Acc: {result.avg_accuracy:.4f}")
```

**Returns:** `EpochResult` dataclass (same as TrainingLoop)

---

### Trainer

**Module:** `utils.training.engine.trainer`

High-level training orchestrator that coordinates all components.

#### Class: `Trainer`

```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig

config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    checkpoint_dir='./checkpoints'
)

trainer = Trainer(
    model=model,
    config=model_config,
    training_config=config,
    task_spec=task_spec,
    hooks=None
)
```

**Parameters:**
- `model` (`nn.Module`): PyTorch model to train
- `config` (`Union[SimpleNamespace, Any]`): Model configuration
- `training_config` (`TrainingConfig`): Training hyperparameters
- `task_spec` (`Optional[TaskSpec]`): Task specification
- `hooks` (`Optional[TrainingHooks]`): Lifecycle hooks

**Methods:**

#### `train()`

Execute complete training workflow.

```python
results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset,
    resume_from='./checkpoints/checkpoint_epoch0005.pt'
)

print(f"Best model at epoch {results['best_epoch']}")
print(f"Final loss: {results['final_loss']:.4f}")
print(f"Training time: {results['training_time']:.1f}s")
```

**Parameters:**
- `train_data` (`Union[Dataset, HFDataset, DataLoader]`): Training dataset
- `val_data` (`Optional[Union[Dataset, HFDataset, DataLoader]]`): Validation dataset
- `resume_from` (`Optional[str]`): Checkpoint path to resume from

**Returns:** `Dict[str, Any]` with:
- `metrics_summary` (`pd.DataFrame`): Per-epoch metrics
- `best_epoch` (`int`): Epoch with best validation performance
- `final_loss` (`float`): Final training loss
- `checkpoint_path` (`Path`): Path to best checkpoint
- `training_time` (`float`): Total training time (seconds)

**Protocol: `TrainingHooks`**

```python
class TrainingHooks(Protocol):
    def on_training_start(self) -> None: ...
    def on_epoch_start(self, epoch: int) -> None: ...
    def on_batch_end(self, batch_idx: int, loss: float) -> None: ...
    def on_validation_end(self, metrics: Dict[str, float]) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None: ...
    def on_training_end(self) -> None: ...
```

**Example: Custom Hooks**

```python
class CustomHooks:
    def on_epoch_start(self, epoch: int) -> None:
        print(f"Starting epoch {epoch}")

    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        if metrics['val_loss'] < 0.3:
            print("🎉 Achieved target loss!")

trainer = Trainer(..., hooks=CustomHooks())
```

**Complete Example:**

```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec

# Configuration
training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    checkpoint_dir='./checkpoints',
    use_wandb=True,
    gradient_accumulation_steps=4,
    export_bundle=True
)

task_spec = TaskSpec.language_modeling_tiny()

# Create trainer
trainer = Trainer(
    model=model,
    config=model_config,
    training_config=training_config,
    task_spec=task_spec
)

# Train
results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset
)

# Analyze results
print(f"✓ Training complete!")
print(f"  Best epoch: {results['best_epoch']}")
print(f"  Final loss: {results['final_loss']:.4f}")
print(f"  Time: {results['training_time']:.1f}s")
print(f"  Checkpoint: {results['checkpoint_path']}")

# Export metrics
results['metrics_summary'].to_csv('metrics.csv', index=False)
```

---

## Production Features

### ModelRegistry

**Module:** `utils.training.model_registry`

SQLite-based model registry for versioning, tagging, and lineage tracking.

#### Class: `ModelRegistry`

```python
from utils.training.model_registry import ModelRegistry

registry = ModelRegistry('models.db')
```

**Methods:**

#### `register_model()`

Register a trained model with metadata.

```python
model_id = registry.register_model(
    name="transformer-v1",
    version="1.0.0",
    checkpoint_path="checkpoints/epoch_10.pt",
    task_type="language_modeling",
    metrics={"val_loss": 0.38, "perplexity": 1.46},
    config_hash="abc123",
    training_run_id=42,
    parent_model_id=None,
    export_formats=["onnx", "torchscript"],
    model_size_mb=450.2,
    metadata={"notes": "Baseline model"}
)
```

#### `promote_model()`

Promote model to production/staging.

```python
registry.promote_model(model_id, "production")
```

**Tags:** `production`, `staging`, `experimental`, `retired`

#### `get_model()`

Retrieve model by ID or tag.

```python
# By ID
model = registry.get_model(model_id=42)

# By tag
prod_model = registry.get_model(tag="production")
```

#### `compare_models()`

Compare multiple models side-by-side.

```python
comparison = registry.compare_models([1, 2, 3])
print(comparison[['version', 'val_loss', 'model_size_mb']])
```

**Returns:** `pd.DataFrame`

---

### JobQueue and Scheduler

**Module:** `utils.training.job_queue`

SQLite-based job queue for automated training workflows.

#### Class: `JobManager`

```python
from utils.training.job_queue import JobManager

manager = JobManager('jobs.db')
```

**Methods:**

#### `submit_job()`

Submit training/evaluation job.

```python
job_id = manager.submit_job(
    job_type='training',
    config={'training_config': config.to_dict()},
    priority=5
)
```

#### `claim_job()`

Atomic job claiming for worker processes.

```python
job = manager.claim_job(worker_id='worker-1')
if job:
    execute_job(job)
    manager.mark_completed(job.job_id)
```

#### Class: `TrainingScheduler`

```python
from utils.training.job_queue import TrainingScheduler

scheduler = TrainingScheduler('jobs.db')
```

**Methods:**

#### `create_schedule()`

Create recurring training schedule.

```python
schedule_id = scheduler.create_schedule(
    name='daily-retrain',
    job_type='training',
    config={'training_config': config.to_dict()},
    schedule_expr='0 2 * * *',  # Daily at 2am
    priority=3
)
```

**Cron Expression Examples:**
- `'0 2 * * *'`: Daily at 2am
- `'0 */6 * * *'`: Every 6 hours
- `'0 0 * * 0'`: Weekly on Sunday

---

### Export Bundle

**Module:** `utils.training.export_utilities`

Production-ready export bundles with inference scripts, Docker, and TorchServe configs.

#### Function: `create_export_bundle()`

```python
from utils.training.export_utilities import create_export_bundle

export_dir = create_export_bundle(
    model=trained_model,
    config=model_config,
    task_spec=task_spec,
    training_config=training_config
)
```

**Generated Structure:**

```
exports/model_<timestamp>/
├── artifacts/
│   ├── model.onnx
│   ├── model.torchscript.pt
│   └── model.pytorch.pt
├── configs/
│   ├── task_spec.json
│   ├── training_config.json
│   └── torchserve_config.json
├── inference.py
├── README.md
├── Dockerfile
└── requirements.txt
```

**Usage:**

```bash
# Run inference
cd exports/model_<timestamp>/
python inference.py --input test.txt --format onnx

# Deploy with Docker
docker build -t transformer-inference .
docker run -p 8080:8080 transformer-inference
```

---

### Retraining Triggers

**Module:** `utils.training.retraining_triggers`

Automatic retraining triggers based on drift, performance, time, or data volume.

#### Class: `RetrainingTriggerManager`

```python
from utils.training.retraining_triggers import RetrainingTriggerManager

manager = RetrainingTriggerManager(
    drift_threshold=0.2,
    performance_threshold=0.1,
    time_threshold_days=7,
    data_volume_threshold=10000
)
```

**Methods:**

#### `check_triggers()`

Check if retraining should be triggered.

```python
should_retrain, reasons = manager.check_triggers(
    current_metrics={'val_loss': 0.45},
    baseline_metrics={'val_loss': 0.38},
    drift_score=0.25,
    new_data_count=15000,
    last_train_timestamp='2025-11-13T00:00:00'
)

if should_retrain:
    print(f"Retraining triggered: {reasons}")
```

**Trigger Types:**
- **Drift**: JS divergence > threshold
- **Performance**: Metric degradation > threshold
- **Time**: Days since last training > threshold
- **Data Volume**: New samples > threshold

---

### Flash Attention

**Module:** `utils.training.flash_attention_wrapper`

Automatic Flash Attention (SDPA) integration for 2-4x speedup.

**Features:**
- Auto-detection of PyTorch 2.0+ and CUDA
- Automatic layer patching
- CPU fallback (graceful)
- Zero configuration required

**Validation:**

```python
from utils.training.flash_attention_wrapper import FlashAttentionWrapper

wrapper = FlashAttentionWrapper(model, config)

if wrapper.sdpa_available:
    print(f"✅ Flash Attention enabled on {len(wrapper.patched_layers)} layers")
else:
    print("ℹ️ Flash Attention not available")
```

**Expected Speedup:**
- T4 GPU: 2.5x
- A100 GPU: 4x
- Combined with `torch.compile`: 30-50% total speedup

---

## Configuration System

### TrainingConfig

**Module:** `utils.training.training_config`

Comprehensive training configuration with validation and versioning.

#### Class: `TrainingConfig`

```python
from utils.training.training_config import TrainingConfig

config = TrainingConfig(
    # Hyperparameters
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    weight_decay=0.01,

    # Model architecture
    vocab_size=50257,
    d_model=768,
    num_layers=12,

    # Reproducibility
    random_seed=42,
    deterministic=False,

    # Optimization
    gradient_accumulation_steps=4,
    gradient_clip_norm=1.0,
    compile_mode="default",

    # Experiment tracking
    wandb_project="transformer-training",
    run_name="baseline-exp",

    # Export
    export_bundle=True,
    export_formats=["onnx", "torchscript"]
)
```

**Methods:**

#### `validate()`

Validate configuration (auto-called by Trainer).

```python
config.validate()  # Raises ValueError if invalid
```

#### `save()` / `load()`

Checkpoint configuration for reproducibility.

```python
# Save (auto-generates timestamped filename)
config_path = config.save()  # config_20251120_143022.json

# Load
loaded_config = TrainingConfig.load(config_path)
```

#### `to_dict()` / `from_dict()`

Serialize/deserialize configuration.

```python
config_dict = config.to_dict()
new_config = TrainingConfig.from_dict(config_dict)
```

---

### TrainingConfigBuilder

**Module:** `utils.training.config_builder`

Fluent API for building configurations with presets.

#### Class: `TrainingConfigBuilder`

```python
from utils.training.config_builder import TrainingConfigBuilder

config = (
    TrainingConfigBuilder()
    .with_preset('small')
    .with_learning_rate(5e-5)
    .with_batch_size(4)
    .with_gradient_accumulation(4)
    .with_wandb('my-project', 'baseline-run')
    .with_export_bundle(['onnx', 'torchscript'])
    .build()
)
```

**Presets:**

```python
builder = TrainingConfigBuilder()

# Small model (development)
config = builder.with_preset('small').build()
# epochs=5, batch_size=4, learning_rate=5e-5

# Medium model (production)
config = builder.with_preset('medium').build()
# epochs=10, batch_size=8, learning_rate=1e-4

# Large model (research)
config = builder.with_preset('large').build()
# epochs=20, batch_size=16, learning_rate=1e-4
```

**Methods:**

- `.with_preset(name)`: Apply preset configuration
- `.with_learning_rate(lr)`: Set learning rate
- `.with_batch_size(size)`: Set batch size
- `.with_epochs(n)`: Set number of epochs
- `.with_gradient_accumulation(steps)`: Set accumulation steps
- `.with_wandb(project, run_name)`: Configure W&B
- `.with_export_bundle(formats)`: Enable export bundle
- `.build()`: Build TrainingConfig instance

**Example:**

```python
# Development configuration
dev_config = (
    TrainingConfigBuilder()
    .with_preset('small')
    .with_epochs(3)
    .with_wandb(None, None)  # Disable W&B
    .build()
)

# Production configuration
prod_config = (
    TrainingConfigBuilder()
    .with_preset('large')
    .with_gradient_accumulation(8)
    .with_export_bundle(['onnx', 'torchscript'])
    .with_wandb('production', 'v2-baseline')
    .build()
)
```

---

## Data Loading

### TaskSpec

**Module:** `utils.training.task_spec`

Task specification for data loading and loss computation.

#### Class: `TaskSpec`

```python
from utils.training.task_spec import TaskSpec

# Language modeling
task_spec = TaskSpec(
    name='gpt2-wiki',
    modality='text',
    task_type='language_modeling',
    max_seq_len=512,
    vocab_size=50257
)

# Vision classification
task_spec = TaskSpec.vision_tiny()
```

**Factory Methods:**

- `.language_modeling_tiny()`: Small text task
- `.vision_tiny()`: Small vision task
- `.classification_tiny()`: Small classification task

---

### UniversalDataModule

**Module:** `utils.training.engine.data`

Universal data module compatible with Trainer.

#### Class: `UniversalDataModule`

```python
from utils.training.engine import UniversalDataModule

data_module = UniversalDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    task_spec=task_spec,
    batch_size=32,
    num_workers=4,
    random_seed=42
)

# Use with Trainer
trainer = Trainer(...)
results = trainer.train(
    train_data=data_module.train_dataloader(),
    val_data=data_module.val_dataloader()
)
```

---

## Utilities

### SeedManager

**Module:** `utils.training.seed_manager`

Reproducibility management with deterministic and fast modes.

#### Function: `set_random_seed()`

```python
from utils.training.seed_manager import set_random_seed

# Fast mode (default)
set_random_seed(42, deterministic=False)
# Seeds: Python, NumPy, PyTorch, CUDA
# cuDNN benchmark enabled (20% faster)

# Deterministic mode
set_random_seed(42, deterministic=True)
# Seeds: Python, NumPy, PyTorch, CUDA
# cuDNN benchmark disabled (bit-exact reproducibility)
# ~5-10% slower
```

---

### ExperimentDB

**Module:** `utils.training.experiment_db`

Local SQLite experiment tracking.

#### Class: `ExperimentDB`

```python
from utils.training.experiment_db import ExperimentDB

db = ExperimentDB('experiments.db')

# Create run
run_id = db.log_run('baseline-v1', config.to_dict(), notes='Initial baseline')

# Log metrics
db.log_metric(run_id, 'train/loss', 0.42, epoch=5)
db.log_metric(run_id, 'val/loss', 0.38, epoch=5)

# Mark complete
db.update_run_status(run_id, 'completed')

# Compare runs
comparison = db.compare_runs([1, 2, 3])
```

---

### Dashboard

**Module:** `utils.training.dashboard`

Training visualization with drift analysis.

#### Class: `Dashboard`

```python
from utils.training.dashboard import Dashboard

dashboard = Dashboard()

# Standard 6-panel dashboard
dashboard.plot(metrics_df, config, title="Training Metrics")

# 10-panel dashboard with drift analysis
dashboard.plot_with_drift(
    metrics_df=metrics_df,
    drift_data=drift_data,
    config=config,
    title="Training Metrics + Drift Analysis"
)
```

---

## Migration from v3.x

See [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md) for detailed migration instructions with before/after examples.

**Quick Summary:**

| v3.x | v4.0 |
|------|------|
| `test_fine_tuning()` | `Trainer.train()` |
| Monolithic function | Modular components |
| 30+ parameters | `TrainingConfig` object |
| Hardcoded Causal LM loss | `LossStrategy` registry |
| Manual gradient accumulation | `GradientAccumulator` |
| Basic checkpointing | `CheckpointManager` with retention |

**Migration Example:**

```python
# v3.x
from utils.tier3_training_utilities import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    learning_rate=5e-5,
    batch_size=4,
    use_wandb=True,
    checkpoint_dir='./checkpoints'
)

# v4.0
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfig

training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    checkpoint_dir='./checkpoints',
    use_wandb=True
)

trainer = Trainer(
    model=model,
    config=config,
    training_config=training_config,
    task_spec=task_spec
)

results = trainer.train(
    train_data=train_dataset,
    val_data=val_dataset
)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.0.0 | 2025-11-20 | Modular engine refactoring (P0-P2) |
| 3.7.0 | 2025-11-16 | Metrics engine with drift detection |
| 3.6.0 | 2025-11-15 | Flash Attention, distributed guardrails |
| 3.5.0 | 2025-11-14 | torch.compile, export bundle |
| 3.4.0 | 2025-11-10 | MetricsTracker, per-batch logging |

---

**End of API Reference**
