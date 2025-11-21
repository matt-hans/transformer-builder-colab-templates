# Gradient Accumulation Guide

## Overview

The `GradientAccumulator` class provides unified gradient accumulation management that works seamlessly with both manual accumulation and PyTorch Lightning's `accumulate_grad_batches`. It prevents double accumulation conflicts and ensures correct step counting for metrics logging.

## Features

- **Manual Accumulation**: Efficient gradient accumulation without PyTorch Lightning
- **Lightning Integration**: Automatic detection and delegation to Lightning trainer
- **Conflict Detection**: Prevents double accumulation with clear error messages
- **Metrics Integration**: Provides `effective_step` for accurate W&B logging
- **Zero Overhead**: No performance penalty when `accumulation_steps=1`
- **Type-Safe**: Full type hints and mypy compliance
- **Checkpointing**: State persistence for training resume

## Basic Usage

### Manual Gradient Accumulation

```python
from utils.training.engine import GradientAccumulator

# Initialize accumulator
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,      # Accumulate over 4 batches
    max_grad_norm=1.0,         # Gradient clipping
    batch_size=8               # Physical batch size
)

# Training loop
for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        loss = model(batch)

        # Accumulate gradients and check if optimizer should step
        should_step = accumulator.accumulate(
            loss=loss,
            model=model,
            is_final_batch=(batch_idx == len(dataloader) - 1)
        )

        if should_step:
            # Optimizer stepped - log metrics
            print(f"Optimizer step at batch {batch_idx}")
            print(f"Gradient norm: {accumulator.stats.last_grad_norm:.2e}")

        # Log metrics at effective steps (for W&B)
        metrics_tracker.log_scalar(
            'train/loss',
            loss.item(),
            step=accumulator.effective_step
        )

    # Reset for next epoch
    accumulator.reset_epoch()
```

**Key Points:**
- Loss scaling is automatic: `scaled_loss = loss / accumulation_steps`
- Optimizer steps every `accumulation_steps` batches
- `is_final_batch=True` forces step on last batch (incomplete accumulation)
- `effective_step` tracks optimizer updates, not micro-batches

### With Automatic Mixed Precision (AMP)

```python
from torch.amp import GradScaler

# Initialize with scaler
scaler = GradScaler('cuda')
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    scaler=scaler,  # AMP support
    batch_size=8
)

# Training loop (accumulator handles scaler automatically)
for batch in dataloader:
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = model(batch)

    # GradientAccumulator handles:
    # - scaler.scale(loss).backward()
    # - scaler.unscale_(optimizer)
    # - scaler.step(optimizer)
    # - scaler.update()
    should_step = accumulator.accumulate(loss, model)
```

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl

# Lightning trainer with accumulation
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Lightning manages accumulation
    max_epochs=10
)

# GradientAccumulator detects and delegates
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,       # Disabled (Lightning controls)
    trainer=trainer             # Automatic detection
)

# Check detection
assert accumulator.is_lightning_managed
print(f"Effective batch: {accumulator.effective_batch_size}")  # 32 (8 * 4)

# In training loop, always returns True (Lightning controls)
for batch in dataloader:
    loss = model(batch)
    should_step = accumulator.accumulate(loss, model)  # Always True

    # effective_step matches Lightning's global_step
    print(f"Effective step: {accumulator.effective_step}")
```

**Conflict Detection:**

```python
# This will raise ValueError
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,  # Manual accumulation
    trainer=trainer        # Lightning has accumulate_grad_batches=4
)
# Error: Gradient accumulation conflict detected!
#   Manual accumulation_steps: 4
#   Lightning accumulate_grad_batches: 4
#
# Resolution options:
#   1. Set accumulation_steps=1 (let Lightning manage)
#   2. Set trainer.accumulate_grad_batches=1 (use manual)
#   3. Remove trainer parameter (disable Lightning)
```

## Integration with MetricsTracker

The `GradientAccumulator` provides the `effective_step` property for accurate metrics logging:

```python
from utils.training.metrics_tracker import MetricsTracker

# Initialize tracker with accumulation awareness
tracker = MetricsTracker(
    use_wandb=True,
    gradient_accumulation_steps=4  # For commit control
)

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    batch_size=8
)

# Training loop
for batch_idx, batch in enumerate(dataloader):
    loss = model(batch)
    should_step = accumulator.accumulate(loss, model)

    # Log at effective steps (75% reduction with steps=4)
    tracker.log_scalar(
        'train/batch_loss',
        loss.item(),
        step=accumulator.effective_step
    )
    # W&B commits only when effective_step changes
```

**Benefits:**
- W&B log volume reduced by 75% with `accumulation_steps=4`
- Step counts align with optimizer updates, not micro-batches
- Metrics remain interpretable across different accumulation settings

## Effective Batch Size Calculation

```python
# Physical batch size: 8
# Accumulation steps: 4
# Effective batch size = 8 * 4 = 32

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    batch_size=8
)

print(f"Effective batch: {accumulator.effective_batch_size}")  # 32

# With Lightning
trainer = pl.Trainer(accumulate_grad_batches=8)
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,
    batch_size=4,
    trainer=trainer
)

print(f"Effective batch: {accumulator.effective_batch_size}")  # 32 (4 * 8)
```

## Gradient Clipping

```python
# Enable gradient clipping
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    max_grad_norm=1.0  # Clip to max norm 1.0
)

# Disable gradient clipping
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    max_grad_norm=None  # No clipping
)

# Access gradient norm
for batch in dataloader:
    loss = model(batch)
    should_step = accumulator.accumulate(loss, model)

    if should_step:
        grad_norm = accumulator.stats.last_grad_norm
        print(f"Pre-clip norm: {grad_norm:.2e}")
```

## Checkpointing and State Persistence

```python
# Save accumulator state with checkpoint
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'accumulator': accumulator.state_dict(),
    'epoch': epoch
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint and resume training
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    batch_size=8
)
accumulator.load_state_dict(checkpoint['accumulator'])

# Resume training with correct step counts
print(f"Resuming from step {accumulator.effective_step}")
```

## Accumulation Statistics

```python
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    batch_size=8
)

for batch_idx, batch in enumerate(dataloader):
    loss = model(batch)
    accumulator.accumulate(loss, model)

    stats = accumulator.stats

    print(f"Total steps: {stats.total_steps}")           # Micro-batch count
    print(f"Optimizer steps: {stats.optimizer_steps}")   # Effective updates
    print(f"Current accumulation: {stats.current_accumulation}")  # 0-3
    print(f"Effective batch: {stats.effective_batch_size}")  # 32
    print(f"Is accumulating: {stats.is_accumulating}")   # True between steps
    print(f"Last grad norm: {stats.last_grad_norm:.2e}") # Pre-clip norm
```

## Advanced: Mathematical Equivalence

Gradient accumulation is mathematically equivalent to using a larger batch size:

```python
# Configuration 1: Small batch + accumulation
config1 = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=8,
    batch_size=4
)
# Effective batch: 32

# Configuration 2: Large batch + no accumulation
config2 = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,
    batch_size=32
)
# Effective batch: 32

# Both produce identical parameter updates (within FP32 tolerance)
# Config1 uses less GPU memory (4 vs 32 samples at a time)
```

**When to use gradient accumulation:**
- GPU memory constraints (can't fit large batch)
- Distributed training (effective batch across GPUs)
- Stability (large effective batch without OOM)

## Performance Considerations

### Zero Overhead Mode

```python
# When accumulation_steps=1, no overhead
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,  # Standard training
    batch_size=32
)
# Optimizer steps every batch, no extra logic
```

### Typical Overhead

- **accumulation_steps=1**: 0% overhead (standard training)
- **accumulation_steps=4**: <1% overhead (gradient norm computation)
- **accumulation_steps=8**: <2% overhead (more norm computations)

### Memory Usage

Gradient accumulation **does not** reduce memory usage for gradients (gradients are accumulated in-place). It reduces memory for:
- Activations (smaller forward pass batch)
- Optimizer states (fewer updates)
- Model copies (distributed training)

## Troubleshooting

### Conflict Error

**Error:** `ValueError: Gradient accumulation conflict detected!`

**Cause:** Both manual `accumulation_steps > 1` and Lightning `accumulate_grad_batches > 1`

**Fix:**
```python
# Option 1: Use Lightning accumulation (recommended)
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,  # Disable manual
    trainer=trainer        # Lightning handles it
)

# Option 2: Use manual accumulation
trainer = pl.Trainer(accumulate_grad_batches=1)  # Disable Lightning
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,  # Manual control
    trainer=None
)
```

### Step Count Mismatch

**Problem:** W&B metrics show wrong step counts

**Cause:** Logging at micro-batch steps instead of effective steps

**Fix:**
```python
# Bad: logs at micro-batch steps
for batch_idx in range(100):
    loss = model(batch)
    accumulator.accumulate(loss, model)
    wandb.log({'loss': loss.item()}, step=batch_idx)  # Wrong!

# Good: logs at effective steps
for batch_idx in range(100):
    loss = model(batch)
    accumulator.accumulate(loss, model)
    wandb.log({'loss': loss.item()}, step=accumulator.effective_step)  # Correct!
```

### Incomplete Final Batch

**Problem:** Last few gradients not applied

**Cause:** Missing `is_final_batch=True`

**Fix:**
```python
for batch_idx, batch in enumerate(dataloader):
    loss = model(batch)
    accumulator.accumulate(
        loss=loss,
        model=model,
        is_final_batch=(batch_idx == len(dataloader) - 1)  # Force step
    )
```

## API Reference

### `GradientAccumulator`

```python
GradientAccumulator(
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = 1.0,
    scaler: Optional[GradScaler] = None,
    batch_size: int = 1,
    trainer: Optional[pl.Trainer] = None
)
```

**Methods:**

- `accumulate(loss, model, is_final_batch=False) -> bool`: Accumulate gradients, returns True if optimizer stepped
- `reset_epoch()`: Reset accumulation state between epochs
- `state_dict() -> dict`: Get state for checkpointing
- `load_state_dict(state_dict)`: Load state from checkpoint

**Properties:**

- `is_lightning_managed: bool`: True if Lightning controls accumulation
- `effective_batch_size: int`: Physical batch size * accumulation steps
- `effective_step: int`: Number of optimizer.step() calls
- `stats: AccumulationStats`: Current accumulation statistics

### `AccumulationStats`

```python
@dataclass
class AccumulationStats:
    total_steps: int                  # Total backward() calls
    optimizer_steps: int              # Number of optimizer.step() calls
    current_accumulation: int         # Position in accumulation window (0 to steps-1)
    effective_batch_size: int         # Physical batch * accumulation steps
    is_accumulating: bool             # True if between optimizer steps
    last_grad_norm: float             # Last computed gradient norm (pre-clip)
```

## Examples

### Example 1: Large Effective Batch on Limited GPU

```python
# Goal: Effective batch of 128 on GPU with 8GB memory
# Solution: Small physical batch + accumulation

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=16,  # 16 accumulation steps
    batch_size=8,           # Physical batch: 8
    max_grad_norm=1.0
)
# Effective batch: 128 (8 * 16)

for batch in small_dataloader:  # batch_size=8
    loss = model(batch)
    should_step = accumulator.accumulate(loss, model, is_final_batch=...)
```

### Example 2: Distributed Training with Lightning

```python
# Goal: Effective batch of 256 across 4 GPUs
# Solution: Lightning DDP + accumulation

trainer = pl.Trainer(
    accelerator='gpu',
    devices=4,                      # 4 GPUs
    strategy='ddp',
    accumulate_grad_batches=8       # Accumulate 8 batches
)

# Physical batch per GPU: 8
# Accumulation: 8
# GPUs: 4
# Effective batch: 8 * 8 * 4 = 256

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=1,           # Lightning manages
    batch_size=8,
    trainer=trainer
)

assert accumulator.effective_batch_size == 64  # 8 * 8 (per GPU)
# Total across GPUs: 256
```

### Example 3: Metrics Logging with Accumulation

```python
# Goal: Log metrics at correct frequency with accumulation

tracker = MetricsTracker(
    use_wandb=True,
    gradient_accumulation_steps=4
)

accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=4,
    batch_size=8
)

for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        loss = model(batch)
        should_step = accumulator.accumulate(loss, model)

        # Log per-batch metrics (logged but not committed to W&B)
        tracker.log_scalar('train/batch_loss', loss.item(),
                          step=accumulator.effective_step)

        if should_step:
            # Log per-step metrics (committed to W&B)
            tracker.log_scalar('train/grad_norm',
                              accumulator.stats.last_grad_norm,
                              step=accumulator.effective_step)
```

## Migration from Manual Accumulation

### Before (Manual Implementation)

```python
optimizer.zero_grad()
accumulation_counter = 0

for batch_idx, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # Manual scaling
    loss.backward()

    accumulation_counter += 1

    if accumulation_counter == accumulation_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        accumulation_counter = 0
```

### After (GradientAccumulator)

```python
accumulator = GradientAccumulator(
    optimizer=optimizer,
    accumulation_steps=accumulation_steps,
    max_grad_norm=1.0,
    batch_size=batch_size
)

for batch_idx, batch in enumerate(dataloader):
    loss = model(batch)
    accumulator.accumulate(loss, model, is_final_batch=...)
    # All logic handled internally!
```

**Benefits:**
- Less boilerplate code
- Automatic AMP support
- Lightning integration
- Correct step counting
- Checkpointing support
- Type-safe implementation

## References

- [PyTorch Gradient Accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)
- [PyTorch Lightning Accumulation](https://lightning.ai/docs/pytorch/stable/common/optimization.html#gradient-accumulation)
- [Training Pipeline v4 Refactoring Plan](../REFACTORING_PLAN_V4.md)
