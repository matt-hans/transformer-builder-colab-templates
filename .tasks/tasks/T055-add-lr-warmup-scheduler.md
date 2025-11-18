---
id: T055
title: Add Learning Rate Warmup Scheduler
status: pending
priority: 2
agent: backend
dependencies: [T051, T052, T053]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [training, optimization, phase2, refactor, enhancement]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - utils/tier3_training_utilities.py
  - CLAUDE.md

est_tokens: 9000
actual_tokens: null
---

## Description

Implement learning rate warmup scheduler with linear warmup followed by cosine decay, improving training stability and final model quality. Warmup prevents early training instability from large gradients when model weights are randomly initialized.

Current state: Training uses constant learning rate throughout (e.g., 5e-5). No warmup period, leading to training instability in first 5-10% of steps. No decay, causing suboptimal final convergence.

Target state: `get_cosine_schedule_with_warmup()` utility creates PyTorch LR scheduler:
- Linear warmup from 0 → target LR over first 10% of steps
- Cosine decay from target LR → 0 over remaining 90% of steps
- Integrated with `test_fine_tuning()` via `use_lr_schedule` parameter
- Logged to W&B via MetricsTracker (requires T051's `log_scalar()`)

**Integration Points:**
- Uses T051's `log_scalar()` to log LR each step
- Works with T052's fixed loss calculation (warmup + correct loss = better training)
- Compatible with T053's reproducibility (scheduler state deterministic with seed)
- Logged to W&B and MetricsTracker for visualization

## Business Context

**User Story:** As an ML practitioner, I want learning rate warmup and decay, so that my models train stably and converge to better final performance.

**Why This Matters:**
Without warmup, large learning rates on random weights cause gradient explosions in first few batches. Without decay, constant LR prevents fine-grained convergence at the end of training. Research shows warmup+decay improves final accuracy by 1-5% absolute.

**What It Unblocks:**
- Stable training on large models (GPT-2 355M, 1.5B)
- Better final model quality (lower perplexity by 5-10%)
- Industry-standard training practice (matches HuggingFace, OpenAI recipes)

**Priority Justification:**
P2 (Important) - Training enhancement that significantly improves results. Not critical for basic training (P1), but essential for production-quality models. Depends on T051-T053's bug fixes being complete first.

## Acceptance Criteria

- [ ] `get_cosine_schedule_with_warmup()` function created in `utils/tier3_training_utilities.py`
- [ ] Function signature: `get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps) -> torch.optim.lr_scheduler.LambdaLR`
- [ ] Linear warmup: LR increases from 0 to `optimizer.param_groups[0]['lr']` over first `num_warmup_steps`
- [ ] Cosine decay: LR decreases from max LR to 0 following cosine curve over remaining steps
- [ ] `test_fine_tuning()` adds `use_lr_schedule` parameter (default: True)
- [ ] When enabled, scheduler created and stepped after each batch: `scheduler.step()`
- [ ] Current LR logged to MetricsTracker each epoch: `tracker.log_scalar('train/learning_rate', current_lr, step=epoch)`
- [ ] Validation: Train with warmup, verify LR starts at ~0, peaks at target, decays to ~0
- [ ] Validation: W&B dashboard shows smooth LR curve (linear up, cosine down)
- [ ] Unit test: Verify LR schedule values at steps 0, 100, 500, 1000 match expected cosine formula

## Test Scenarios

**Test Case 1: LR Warmup Phase (0% → 10% of training)**
- Given: 1000 total steps, 100 warmup steps, target LR=5e-5
- When: Check LR at steps 0, 50, 100
- Then: LR = [0, 2.5e-5, 5e-5] (linear increase)

**Test Case 2: LR Decay Phase (10% → 100% of training)**
- Given: 1000 total steps, 100 warmup steps, target LR=5e-5
- When: Check LR at steps 100, 550, 1000
- Then: LR = [5e-5, ~2.5e-5 (cosine), ~0] (cosine decay)

**Test Case 3: Integration with test_fine_tuning()**
- Given: `test_fine_tuning(model, config, n_epochs=10, use_lr_schedule=True)`
- When: Train completes
- Then: MetricsTracker logs show LR curve, final LR near 0, training stable

**Test Case 4: Disabled LR Schedule (Backward Compatibility)**
- Given: `test_fine_tuning(model, config, use_lr_schedule=False)`
- When: Train completes
- Then: LR constant at 5e-5 throughout, no scheduler created

**Test Case 5: W&B Visualization**
- Given: Training with `use_wandb=True, use_lr_schedule=True`
- When: Check W&B dashboard "train/learning_rate" plot
- Then: Shows linear warmup ramp, smooth cosine decay, matches expected curve

**Test Case 6: Scheduler State Determinism**
- Given: Train with seed=42, deterministic=True, warmup enabled
- When: Run twice with identical setup
- Then: LR schedules identical across runs (bit-identical LR values at each step)

**Test Case 7: Performance Impact (Small Dataset)**
- Given: WikiText-2, 1 epoch, with vs. without warmup
- When: Compare final validation perplexity
- Then: Warmup version ≤ no-warmup PPL (equal or better, typically 2-5% better)

## Technical Implementation

**Required Components:**

1. **Create `get_cosine_schedule_with_warmup()` in `utils/tier3_training_utilities.py`:**
```python
from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine decay.

    LR schedule:
      - Steps 0 to num_warmup_steps: Linear increase from 0 to initial LR
      - Steps num_warmup_steps to num_training_steps: Cosine decay to 0

    This is the standard schedule used in BERT, GPT-2, GPT-3 training.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of steps for linear warmup (typically 10% of total)
        num_training_steps: Total training steps (epochs * batches_per_epoch)
        num_cycles: Number of cosine cycles (default: 0.5 = half cosine wave to 0)
        last_epoch: Last epoch for resuming training (default: -1 = start from scratch)

    Returns:
        LambdaLR scheduler to be stepped after each optimizer step

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        >>> total_steps = 10 * len(train_loader)  # 10 epochs
        >>> warmup_steps = int(0.1 * total_steps)  # 10% warmup
        >>> scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        >>> for epoch in range(10):
        ...     for batch in train_loader:
        ...         loss = train_step(batch)
        ...         optimizer.step()
        ...         scheduler.step()  # Update LR after each batch
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
```

2. **Update `test_fine_tuning()` to use LR scheduler:**
```python
def test_fine_tuning(
    model: nn.Module,
    config: SimpleNamespace,
    n_epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    use_wandb: bool = False,
    random_seed: int = 42,
    deterministic: bool = False,
    use_lr_schedule: bool = True  # NEW: Enable/disable LR warmup+decay
) -> dict:
    """Fine-tune model with optional LR warmup and cosine decay."""

    # ... existing setup code ...

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create LR scheduler
    scheduler = None
    if use_lr_schedule:
        total_steps = n_epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
        print(f"📈 LR Schedule: {warmup_steps} warmup steps, {total_steps} total steps")

    tracker = MetricsTracker(use_wandb=use_wandb)
    global_step = 0

    for epoch in range(n_epochs):
        # Training epoch
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # ... training step (forward, backward, optimizer.step()) ...

            # Step scheduler after each batch
            if scheduler is not None:
                scheduler.step()
                global_step += 1

                # Log LR every epoch (not every batch to reduce logging volume)
                if batch_idx == 0:  # First batch of epoch
                    current_lr = optimizer.param_groups[0]['lr']
                    tracker.log_scalar('train/learning_rate', current_lr, step=epoch)

        # ... validation epoch ...

        # Log epoch metrics (existing code)
        tracker.log_epoch(
            epoch=epoch,
            train_metrics={'loss': avg_train_loss},
            val_metrics=val_results,
            learning_rate=optimizer.param_groups[0]['lr']  # Current LR
        )

    return {
        'final_train_loss': avg_train_loss,
        'final_val_loss': val_results['loss'],
        'final_perplexity': val_results['perplexity'],
        'metrics_summary': tracker.get_summary(),
        'best_epoch': tracker.get_summary()['val/loss'].idxmin(),
        'scheduler_used': scheduler is not None  # Track if schedule was used
    }
```

3. **Add unit tests in `tests/test_lr_scheduler.py`:**
```python
import pytest
import torch
from utils.tier3_training_utilities import get_cosine_schedule_with_warmup

def test_lr_warmup_phase():
    """Verify linear warmup from 0 to target LR"""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    # Check LR at various warmup steps
    lrs = []
    for step in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # LR should start near 0 and end at 1e-4
    assert lrs[0] < 1e-5, "LR should start near 0"
    assert abs(lrs[99] - 1e-4) < 1e-6, "LR should reach target at end of warmup"
    assert lrs[50] < lrs[99], "LR should increase during warmup"


def test_lr_cosine_decay():
    """Verify cosine decay from max LR to 0"""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

    # Skip warmup
    for _ in range(100):
        optimizer.step()
        scheduler.step()

    # Collect LR during decay phase
    lrs = []
    for step in range(900):  # Remaining steps
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # LR should decrease monotonically (cosine decay)
    assert lrs[0] > lrs[-1], "LR should decrease during decay"
    assert lrs[-1] < 1e-5, "LR should approach 0 at end"


def test_schedule_determinism():
    """Verify scheduler produces same LR sequence with same seed"""
    def get_lr_sequence(seed):
        torch.manual_seed(seed)
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 100, 1000)

        lrs = []
        for _ in range(1000):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
        return lrs

    lrs1 = get_lr_sequence(42)
    lrs2 = get_lr_sequence(42)

    assert lrs1 == lrs2, "LR schedule should be deterministic with same seed"
```

4. **Update CLAUDE.md with LR schedule usage:**
```python
# Training with LR Warmup and Decay
from utils.tier3_training_utilities import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    learning_rate=5e-5,
    use_lr_schedule=True,  # Enable warmup + cosine decay
    use_wandb=True
)

# Check W&B dashboard for "train/learning_rate" plot
# Should show linear ramp up, then smooth cosine decay
```

**Validation Commands:**

```bash
# Unit tests
pytest tests/test_lr_scheduler.py -v

# Integration test (manual in Colab)
# 1. Train with use_lr_schedule=True, log to W&B
# 2. Check W&B dashboard "train/learning_rate" plot
# 3. Verify: linear warmup (0 → 5e-5), cosine decay (5e-5 → ~0)

# Performance comparison
# 1. Train model A: use_lr_schedule=False (baseline)
# 2. Train model B: use_lr_schedule=True
# 3. Compare final val PPL: B should be ≤ A (typically 2-5% better)
```

**Code Patterns:**
- Scheduler stepped **after** `optimizer.step()` (PyTorch convention)
- Log LR once per epoch (not per batch) to reduce logging volume
- Default `use_lr_schedule=True` (best practice, can disable for debugging)
- Uses `LambdaLR` for flexibility (can customize schedule formula)

## Dependencies

**Hard Dependencies** (must be complete first):
- [T051] Add log_scalar() method - Required for logging LR to MetricsTracker
- [T052] Fix padding token handling - Warmup more effective with correct loss
- [T053] Fix DataLoader reproducibility - Scheduler state must be deterministic

**Soft Dependencies** (nice to have):
- [T054] Extract duplicated code - Makes adding scheduler to training loop cleaner

**External Dependencies:**
- PyTorch `torch.optim.lr_scheduler.LambdaLR` (available since PyTorch 1.0)
- Math library (standard Python)

**Blocks Future Tasks:**
- None (independent enhancement)

## Design Decisions

**Decision 1: Cosine Schedule vs. Linear Decay**
- **Rationale:** Cosine decay used in BERT, GPT-2, GPT-3. Empirically performs better than linear or exponential decay.
- **Alternatives:**
  - Linear decay - simpler but worse final convergence
  - Exponential decay - too aggressive, performance drops
  - Step decay - discontinuous, causes training instability
- **Trade-offs:**
  - Pro: State-of-the-art schedule, proven in major models
  - Con: Slightly more complex formula (negligible, built-in to PyTorch patterns)

**Decision 2: 10% Warmup by Default**
- **Rationale:** Standard in literature (BERT, GPT papers). Balances stability (longer warmup) vs. training efficiency (shorter warmup).
- **Alternatives:**
  - 5% warmup - may be too short for large models
  - 20% warmup - wastes training time on suboptimal LR
- **Trade-offs:**
  - Pro: Works well across model sizes (124M to 1.5B parameters)
  - Con: Not optimal for every dataset (users can override by passing custom `num_warmup_steps`)

**Decision 3: Default `use_lr_schedule=True`**
- **Rationale:** Best practice should be default. Users training production models benefit automatically.
- **Alternatives:**
  - Default False - users forget to enable, miss out on 2-5% PPL improvement
  - No parameter, always on - removes flexibility for debugging
- **Trade-offs:**
  - Pro: Production-quality defaults out-of-box
  - Con: Debugging harder (can disable with `use_lr_schedule=False`)

**Decision 4: Log LR Per Epoch (Not Per Batch)**
- **Rationale:** LR changes smoothly, per-batch logging excessive (1000s of log calls). Per-epoch sufficient for visualization.
- **Alternatives:**
  - Log every batch - bloats W&B logs, slow dashboard
  - Log every 10 batches - still verbose, marginal benefit
- **Trade-offs:**
  - Pro: Efficient logging, clear W&B plots
  - Con: Less granular than per-batch (acceptable, LR curve is smooth)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Warmup too short for large models → early instability | Medium | Low | Document: "For models >1B parameters, increase warmup to 1000 steps: `warmup_steps = max(1000, int(0.1 * total_steps))`". Add parameter to TrainingConfig. |
| Scheduler state not saved in checkpoints → incorrect resume | High | Medium | Future task: Save scheduler state in checkpoints (`torch.save({'scheduler': scheduler.state_dict(), ...})`). For now, document limitation. |
| LR decay to exactly 0 → optimizer numerical issues | Low | Low | Schedule caps at 0, not negative. PyTorch AdamW handles LR=0 gracefully (no updates). Test with final LR verification. |
| Users forget scheduler.step() → constant LR despite schedule | Medium | Low | Encapsulated in `test_fine_tuning()`, not exposed to users. Unit tests verify scheduler stepped correctly. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** User approved comprehensive refactor plan - Phase 2, Task 1 of 6. Training enhancement that improves final model quality by 2-5% PPL. Depends on T051-T053 bug fixes being complete.
**Dependencies:** [T051] log_scalar(), [T052] padding fix, [T053] reproducibility
**Estimated Complexity:** Complex (3-hour implementation requiring scheduler integration, logging, testing, and validation)

## Completion Checklist

**Code Quality:**
- [ ] `get_cosine_schedule_with_warmup()` function with full docstring
- [ ] Type hints on all parameters
- [ ] Clean integration into `test_fine_tuning()` via `use_lr_schedule` param
- [ ] Scheduler stepped after `optimizer.step()` (correct order)

**Testing:**
- [ ] Unit tests pass: `test_lr_warmup_phase()`, `test_lr_cosine_decay()`, `test_schedule_determinism()`
- [ ] Integration test: Train with warmup, verify LR curve in W&B
- [ ] Performance test: Warmup model achieves ≤ baseline PPL (typically 2-5% better)
- [ ] Backward compat test: `use_lr_schedule=False` → constant LR

**Documentation:**
- [ ] CLAUDE.md updated with LR schedule usage example
- [ ] Docstring includes realistic training loop example
- [ ] Comments explain 10% warmup rationale

**Integration:**
- [ ] Works with T051's `log_scalar()` for LR logging
- [ ] Compatible with T053's deterministic mode (scheduler state deterministic)
- [ ] Default `use_lr_schedule=True` for best-practice defaults

**Definition of Done:**
Task is complete when LR scheduler implemented, integrated into training loop, logs to W&B, unit tests pass, and validation shows 2-5% PPL improvement with warmup.
