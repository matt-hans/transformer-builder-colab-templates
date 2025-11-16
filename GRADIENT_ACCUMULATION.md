# Gradient Accumulation Feature

## Overview

Gradient accumulation allows you to simulate larger batch sizes than your GPU memory can physically hold. Instead of updating weights after every batch, gradients are accumulated over N batches before performing an optimizer step.

**Key Benefit**: Train with effective batch_size=32 on a 4GB GPU that can only fit batch_size=4 by setting `gradient_accumulation_steps=8`.

## Usage

### Basic Example

```python
from utils.tier3_training_utilities import test_fine_tuning

# Simulate batch_size=32 with limited GPU memory
result = test_fine_tuning(
    model=model,
    config=config,
    train_data=train_data,
    val_data=val_data,
    n_epochs=10,
    batch_size=4,                      # Physical batch size (fits in GPU)
    gradient_accumulation_steps=8,     # Accumulate over 8 batches
    learning_rate=5e-5,
    use_wandb=True
)

# Effective batch size = 4 * 8 = 32
```

### Parameters

- **`batch_size`**: Physical batch size loaded into GPU memory
- **`gradient_accumulation_steps`**: Number of batches to accumulate gradients over before updating weights
  - Default: `1` (no accumulation, update every batch)
  - Effective batch size = `batch_size * gradient_accumulation_steps`

### When to Use

**Use gradient accumulation when**:
- GPU memory limits your batch size to <8
- You want training stability from larger batches (e.g., batch_size=32+)
- You're comparing models with different hardware constraints

**Don't use gradient accumulation when**:
- You can already fit your desired batch size in memory
- Training with very small datasets (accumulation overhead not worth it)

## How It Works

### Mathematical Equivalence

Gradient accumulation produces mathematically equivalent gradients to using a larger physical batch:

```
# Standard training (batch_size=32)
loss = compute_loss(batch_32)
loss.backward()  # ∇L w.r.t. 32 samples
optimizer.step()

# Gradient accumulation (batch_size=4, accum_steps=8)
optimizer.zero_grad()
for i in range(8):
    loss = compute_loss(batch_4) / 8  # Scale loss!
    loss.backward()  # Accumulate ∇L
optimizer.step()  # Same total gradient as batch_32
```

**Key**: Loss must be scaled by `1/accumulation_steps` to maintain correct gradient magnitude.

### Implementation Details

The training loop follows this pattern:

```python
optimizer.zero_grad()
accumulation_counter = 0

for batch_idx, batch in enumerate(train_loader):
    # Forward + backward (accumulate gradients)
    loss = compute_loss(batch) / gradient_accumulation_steps
    loss.backward()

    accumulation_counter += 1

    # Update weights every N batches
    if accumulation_counter == gradient_accumulation_steps:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # LR scheduling per optimizer step
        optimizer.zero_grad()
        accumulation_counter = 0

# Handle incomplete final batch
if accumulation_counter > 0:
    optimizer.step()
```

**Incomplete Batches**: If total_batches % accumulation_steps != 0, the final incomplete batch is still applied (prevents gradient waste).

## Monitoring

### Console Output

Training prints effective batch size:

```
============================================================
FINE-TUNING TEST
============================================================
Training samples: 1000
Batch size: 4
Gradient accumulation steps: 8
Effective batch size: 32
============================================================
```

### W&B Logging

Metrics logged every epoch:

- `config/effective_batch_size`: Total effective batch size
- `config/gradient_accumulation_steps`: Number of accumulation steps
- `config/physical_batch_size`: GPU batch size

### Verifying Correctness

Check that optimizer steps match expected frequency:

```python
# With 100 batches and gradient_accumulation_steps=4:
# Expected optimizer steps = ceil(100/4) = 25

result = test_fine_tuning(..., gradient_accumulation_steps=4)

# Verify via gradient norm history length
assert len(result['grad_norm_history']) == 25
```

## Performance Considerations

### Memory

**Benefit**: Accumulation uses ~same memory as batch_size=4
- No need to store larger batches
- Gradients accumulate in-place

**Cost**: Slightly more memory for optimizer state (negligible)

### Speed

**Slower than large batch** (more forward passes):
- batch_size=32: 1 forward/backward per step
- batch_size=4, accum=8: 8 forward/backward per step

**Faster than sequential** (batching still helps):
- Better than batch_size=1 with 32 sequential steps

### Best Practices

1. **Choose accumulation_steps as power of 2**: 2, 4, 8, 16
   - Aligns with GPU architecture
   - Cleaner division of batches

2. **Match total effective batch to baseline**:
   ```python
   # Baseline: batch_size=32 on A100
   # Limited GPU: batch_size=4 on T4
   gradient_accumulation_steps = 32 // 4  # = 8
   ```

3. **Adjust learning rate if needed**:
   - Some optimizers (e.g., LAMB) scale with batch size
   - For AdamW: usually no adjustment needed

## Testing

### Unit Tests

See `tests/test_gradient_accumulation.py`:

- **Optimizer step frequency**: Verifies steps called every N batches
- **Backward compatibility**: accum_steps=1 behaves like original
- **Gradient equivalence**: Accumulated gradients = large batch gradients

### Integration Tests

See `tests/test_gradient_accumulation_simple.py`:

- **Smoke test**: Training completes without errors
- **Effective batch logging**: Console output correct
- **Loss convergence**: Loss decreases as expected

## Troubleshooting

### Loss not decreasing

**Symptom**: Training loss stays flat or increases

**Causes**:
1. Learning rate too high for effective batch size
   - **Fix**: Reduce LR by factor of accumulation_steps
2. Gradient overflow/underflow
   - **Fix**: Enable AMP (`use_amp=True`)

### Out of memory (OOM)

**Symptom**: CUDA out of memory error

**Causes**:
1. `batch_size` still too large
   - **Fix**: Reduce `batch_size` further, increase `gradient_accumulation_steps`
2. Model gradients not cleared
   - **Fix**: Verify `optimizer.zero_grad()` called

### Incorrect optimizer step count

**Symptom**: Wrong number of optimizer steps

**Debugging**:
```python
# Add logging to track steps
total_batches = len(train_loader) * n_epochs
expected_steps = math.ceil(total_batches / gradient_accumulation_steps)

print(f"Expected optimizer steps: {expected_steps}")
print(f"Actual gradient norms logged: {len(result['grad_norm_history'])}")
```

## References

- Original paper: "Training ImageNet in 1 Hour" (Goyal et al., 2017)
- PyTorch docs: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
- Effective batch size scaling: https://arxiv.org/abs/1706.02677

## Examples

### Example 1: Limited GPU Memory

```python
# Can only fit batch_size=2 on GPU
# Want effective batch_size=16 for stability

result = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=20,
    batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=1e-4,
    use_amp=True  # Enable FP16 for even more memory savings
)
```

### Example 2: Reproducing Baseline Results

```python
# Baseline trained with batch_size=64 on A100
# Reproduce on GTX 1080 Ti (can only fit batch_size=8)

baseline_batch_size = 64
your_batch_size = 8
accum_steps = baseline_batch_size // your_batch_size  # = 8

result = test_fine_tuning(
    model=model,
    config=config,
    batch_size=your_batch_size,
    gradient_accumulation_steps=accum_steps,
    learning_rate=5e-5,  # Same LR as baseline
    n_epochs=baseline_epochs
)
```

### Example 3: Hyperparameter Search

```python
import optuna

def objective(trial):
    batch_size = 4  # Fixed by GPU memory
    accum_steps = trial.suggest_categorical('accum_steps', [1, 2, 4, 8])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)

    result = test_fine_tuning(
        model=model_factory(),
        config=config,
        batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        learning_rate=lr,
        n_epochs=5
    )

    return result['final_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best effective batch: {4 * study.best_params['accum_steps']}")
```
