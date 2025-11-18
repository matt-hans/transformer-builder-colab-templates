---
id: T051
title: Add Missing log_scalar() Method to MetricsTracker
status: pending
priority: 1
agent: backend
dependencies: []
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [bug-fix, metrics, phase1, refactor, critical, blocks-gpu-metrics]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - utils/training/metrics_tracker.py
  - CLAUDE.md

est_tokens: 3000
actual_tokens: null
---

## Description

Add missing `log_scalar()` method to `MetricsTracker` class to enable real-time scalar metric logging during training (learning rate, gradient norm, batch loss). This method is **currently missing** but required by T064-T065's GPU metrics tracking.

Current state: `MetricsTracker` in `utils/training/metrics_tracker.py` only has `log_epoch()` for end-of-epoch metrics. No method exists for logging per-batch/per-step metrics like learning rate decay, gradient norms, or GPU utilization.

Target state: `MetricsTracker.log_scalar(metric_name, value, step)` method added that:
- Logs to W&B via `wandb.log()` if W&B enabled
- Stores in internal `self._step_metrics` DataFrame for later analysis
- Supports optional step parameter (auto-increments if not provided)
- Thread-safe for multi-worker DataLoader scenarios

**Integration Points:**
- Enables T064's GPU metrics: `tracker.log_scalar('gpu/memory_used', gpu_mem, step=global_step)`
- Enables T065's gradient logging: `tracker.log_scalar('gradients/l2_norm', grad_norm, step=batch_idx)`
- Used in training loop for LR: `tracker.log_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step=epoch)`

## Business Context

**User Story:** As an ML engineer, I want to track per-batch metrics (learning rate, gradient norms, GPU usage) in W&B, so that I can debug training issues and optimize hyperparameters.

**Why This Matters:**
Epoch-level metrics are too coarse-grained for debugging. When training diverges at batch 347 of epoch 2, engineers need per-batch visibility into LR, gradients, and GPU state to diagnose the issue. Missing `log_scalar()` blocks this critical observability.

**What It Unblocks:**
- [T064] GPU metrics tracking (memory, utilization, temperature)
- [T065] Gradient distribution logging (per-layer norms, histograms)
- Per-batch learning rate logging for LR schedules
- Real-time training diagnostics in W&B dashboard

**Priority Justification:**
P1 (Critical) - This is a **blocking bug** for T064-T065. Without `log_scalar()`, GPU metrics and gradient tracking cannot be implemented. Must fix before proceeding with Phase 5 tasks.

## Acceptance Criteria

- [ ] `MetricsTracker.log_scalar(metric_name: str, value: float, step: Optional[int] = None)` method added
- [ ] Method logs to W&B if `self.use_wandb=True`: `wandb.log({metric_name: value}, step=step)`
- [ ] Method stores in `self._step_metrics` DataFrame: `{'step': step, 'metric': metric_name, 'value': value, 'timestamp': now}`
- [ ] Auto-increment step counter if `step=None`: `step = self._global_step; self._global_step += 1`
- [ ] Thread-safe: Uses lock around `self._step_metrics` append operations
- [ ] Returns None, raises no exceptions on valid inputs
- [ ] Validation: Can call `tracker.log_scalar('test/loss', 0.5, step=10)` - logs to W&B and internal storage
- [ ] Validation: `tracker.get_step_metrics()` returns DataFrame with all logged scalars
- [ ] Unit test added: `test_log_scalar_with_wandb()`, `test_log_scalar_without_wandb()`, `test_log_scalar_auto_increment()`
- [ ] Docstring with Args/Returns/Raises sections and example usage

## Test Scenarios

**Test Case 1: Log Scalar with W&B Enabled**
- Given: `tracker = MetricsTracker(use_wandb=True)`, W&B initialized
- When: `tracker.log_scalar('train/batch_loss', 0.42, step=100)`
- Then: W&B logs `{'train/batch_loss': 0.42}` at step 100, internal DataFrame stores entry

**Test Case 2: Log Scalar with W&B Disabled**
- Given: `tracker = MetricsTracker(use_wandb=False)`
- When: `tracker.log_scalar('val/accuracy', 0.87, step=50)`
- Then: No W&B call (wandb not used), internal DataFrame stores entry, no errors

**Test Case 3: Auto-Increment Step Counter**
- Given: `tracker = MetricsTracker()`, no step provided
- When: `tracker.log_scalar('lr', 5e-5)` called 3 times
- Then: Steps auto-assigned as 0, 1, 2; DataFrame shows sequential steps

**Test Case 4: Multiple Metrics at Same Step**
- Given: Training batch 42
- When: `tracker.log_scalar('train/loss', 0.5, step=42)` then `tracker.log_scalar('train/lr', 1e-4, step=42)`
- Then: W&B logs both at step 42 (merged in single wandb.log call if batched), DataFrame has 2 rows

**Test Case 5: Retrieve Step Metrics**
- Given: Logged 100 scalars over training
- When: `df = tracker.get_step_metrics()`
- Then: Returns DataFrame with 100 rows, columns=['step', 'metric', 'value', 'timestamp'], sorted by step

**Test Case 6: Thread Safety (Multi-Worker DataLoader)**
- Given: `num_workers=4` DataLoader, multiple threads logging
- When: Concurrent `log_scalar()` calls from 4 threads
- Then: All metrics recorded without corruption, no race conditions, DataFrame integrity maintained

**Test Case 7: Invalid Input Handling**
- Given: `tracker = MetricsTracker()`
- When: `tracker.log_scalar('', 0.5)` (empty metric name) or `tracker.log_scalar('loss', 'invalid')` (non-numeric value)
- Then: Raises ValueError with clear message: "metric_name cannot be empty" / "value must be numeric"

## Technical Implementation

**Required Components:**

1. **Add `log_scalar()` method to `utils/training/metrics_tracker.py`:**
```python
import threading
from typing import Optional
import pandas as pd
from datetime import datetime

class MetricsTracker:
    def __init__(self, use_wandb: bool = False):
        self.use_wandb = use_wandb
        self._epoch_metrics = []
        self._step_metrics = []  # NEW: Store per-step scalars
        self._global_step = 0    # NEW: Auto-increment counter
        self._lock = threading.Lock()  # NEW: Thread safety

    def log_scalar(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a scalar metric at a specific training step.

        Used for per-batch/per-step metrics like learning rate, gradient norms,
        or GPU utilization. Complements `log_epoch()` for finer-grained tracking.

        Args:
            metric_name: Metric identifier (e.g., 'train/learning_rate', 'gpu/memory_mb')
            value: Numeric value to log
            step: Training step/batch index. If None, auto-increments internal counter.

        Raises:
            ValueError: If metric_name is empty or value is non-numeric

        Example:
            >>> tracker = MetricsTracker(use_wandb=True)
            >>> for batch_idx, batch in enumerate(dataloader):
            ...     loss = train_batch(batch)
            ...     tracker.log_scalar('train/batch_loss', loss.item(), step=batch_idx)
            ...     tracker.log_scalar('train/lr', optimizer.param_groups[0]['lr'], step=batch_idx)
        """
        # Validation
        if not metric_name or not isinstance(metric_name, str):
            raise ValueError("metric_name must be a non-empty string")
        if not isinstance(value, (int, float)):
            raise ValueError(f"value must be numeric, got {type(value)}")

        # Auto-increment step if not provided
        if step is None:
            with self._lock:
                step = self._global_step
                self._global_step += 1

        # Log to W&B
        if self.use_wandb:
            try:
                import wandb
                wandb.log({metric_name: value}, step=step)
            except ImportError:
                pass  # W&B not available, skip silently

        # Store internally for later retrieval
        with self._lock:
            self._step_metrics.append({
                'step': step,
                'metric': metric_name,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })

    def get_step_metrics(self) -> pd.DataFrame:
        """
        Retrieve all logged step metrics as a DataFrame.

        Returns:
            DataFrame with columns ['step', 'metric', 'value', 'timestamp'],
            sorted by step ascending.

        Example:
            >>> df = tracker.get_step_metrics()
            >>> df[df['metric'] == 'train/batch_loss'].plot(x='step', y='value')
        """
        with self._lock:
            df = pd.DataFrame(self._step_metrics)
        return df.sort_values('step') if not df.empty else df

    def get_summary(self) -> pd.DataFrame:
        """Existing method - returns epoch-level metrics (unchanged)"""
        return pd.DataFrame(self._epoch_metrics)
```

2. **Add unit tests in `tests/test_metrics_tracker.py` (create if doesn't exist):**
```python
import pytest
from utils.training.metrics_tracker import MetricsTracker

def test_log_scalar_with_wandb(monkeypatch):
    """Test scalar logging when W&B is enabled"""
    # Mock wandb.log
    logged_data = []
    def mock_wandb_log(data, step):
        logged_data.append({'data': data, 'step': step})

    monkeypatch.setattr('wandb.log', mock_wandb_log)

    tracker = MetricsTracker(use_wandb=True)
    tracker.log_scalar('train/loss', 0.42, step=10)

    assert len(logged_data) == 1
    assert logged_data[0] == {'data': {'train/loss': 0.42}, 'step': 10}

def test_log_scalar_without_wandb():
    """Test scalar logging when W&B is disabled"""
    tracker = MetricsTracker(use_wandb=False)
    tracker.log_scalar('val/acc', 0.87, step=5)

    df = tracker.get_step_metrics()
    assert len(df) == 1
    assert df.iloc[0]['metric'] == 'val/acc'
    assert df.iloc[0]['value'] == 0.87
    assert df.iloc[0]['step'] == 5

def test_log_scalar_auto_increment():
    """Test auto-incrementing step counter"""
    tracker = MetricsTracker()
    tracker.log_scalar('lr', 5e-5)
    tracker.log_scalar('lr', 4e-5)
    tracker.log_scalar('lr', 3e-5)

    df = tracker.get_step_metrics()
    assert df['step'].tolist() == [0, 1, 2]

def test_log_scalar_invalid_inputs():
    """Test error handling for invalid inputs"""
    tracker = MetricsTracker()

    with pytest.raises(ValueError, match="metric_name must be"):
        tracker.log_scalar('', 0.5)

    with pytest.raises(ValueError, match="value must be numeric"):
        tracker.log_scalar('loss', 'invalid')
```

3. **Update CLAUDE.md usage example:**
```python
# Using MetricsTracker for Training with W&B
from utils.training.metrics_tracker import MetricsTracker

tracker = MetricsTracker(use_wandb=True)

for epoch in range(n_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = train_batch(model, batch, optimizer)

        # Log per-batch metrics
        tracker.log_scalar('train/batch_loss', loss.item(), step=epoch * len(dataloader) + batch_idx)
        tracker.log_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], step=epoch * len(dataloader) + batch_idx)

    # Log per-epoch metrics (existing method)
    tracker.log_epoch(epoch=epoch, train_metrics={'loss': avg_train_loss}, ...)
```

**Validation Commands:**

```bash
# Manual testing (local environment)
python -c "
from utils.training.metrics_tracker import MetricsTracker
tracker = MetricsTracker(use_wandb=False)
tracker.log_scalar('test/metric', 0.5, step=1)
df = tracker.get_step_metrics()
print(df)
# Expected: DataFrame with 1 row
"

# Unit tests
pytest tests/test_metrics_tracker.py::test_log_scalar_with_wandb -v
pytest tests/test_metrics_tracker.py::test_log_scalar_auto_increment -v
```

**Code Patterns:**
- Thread-safe with `threading.Lock()` for multi-worker DataLoader compatibility
- Graceful W&B handling (try/except ImportError for offline mode)
- Validation at method entry (fail-fast on invalid inputs)
- Consistent return type (None) and exceptions (ValueError)

## Dependencies

**Hard Dependencies** (must be complete first):
- None (standalone bug fix)

**Soft Dependencies** (nice to have):
- None

**External Dependencies:**
- pandas (already in requirements.txt)
- wandb (optional, gracefully handled if missing)

**Blocks Future Tasks:**
- [T064] GPU Metrics Tracking - requires `log_scalar()` to log GPU memory/utilization
- [T065] Gradient Distribution Logging - requires `log_scalar()` for per-layer gradient norms

## Design Decisions

**Decision 1: Auto-Increment Step vs. Require Explicit Step**
- **Rationale:** Optional step parameter with auto-increment balances convenience (simple usage) and control (explicit step for complex scenarios).
- **Alternatives:**
  - Always require step - verbose for simple cases
  - Always auto-increment - loses control in advanced scenarios (e.g., resuming from checkpoint)
- **Trade-offs:**
  - Pro: Flexible API, works for both batch-level and custom step schemes
  - Con: Users must understand step semantics (documented in docstring)

**Decision 2: Thread Safety via Lock**
- **Rationale:** DataLoader with `num_workers > 0` may call `log_scalar()` from multiple threads. Lock prevents race conditions in `_step_metrics` list.
- **Alternatives:**
  - No lock - risk of data corruption
  - Queue-based logging - more complex, overkill for this use case
- **Trade-offs:**
  - Pro: Correct behavior in multi-threaded scenarios
  - Con: Slight performance overhead (negligible for per-batch logging)

**Decision 3: Separate `_step_metrics` vs. Merge with `_epoch_metrics`**
- **Rationale:** Step-level and epoch-level metrics have different granularity and access patterns. Separate storage simplifies querying.
- **Alternatives:**
  - Single `_all_metrics` list with type field - harder to query
  - Nested dict structure - complex, harder to convert to DataFrame
- **Trade-offs:**
  - Pro: Clean separation of concerns, easy `get_step_metrics()` implementation
  - Con: Two internal data structures (acceptable, clear purpose)

**Decision 4: W&B Logging via `wandb.log()` Not Batched**
- **Rationale:** Each `log_scalar()` call immediately logs to W&B for real-time dashboard updates.
- **Alternatives:**
  - Batch W&B logs (e.g., every 10 calls) - reduces API calls but delays visibility
  - Async logging queue - complex, may lose data on crash
- **Trade-offs:**
  - Pro: Real-time W&B dashboard updates, simple implementation
  - Con: More W&B API calls (acceptable, W&B handles this efficiently)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Lock contention degrades performance in high-frequency logging | Medium | Low | Lock only held during list append (microseconds). If performance issue arises, switch to lock-free queue (deque with append). Profile before optimizing. |
| DataFrame memory grows unbounded with millions of logged scalars | Medium | Medium | Document in docstring: "For very long training runs (>1M steps), consider periodic `get_step_metrics()` export and `_step_metrics.clear()`." Add future task for automatic rollover. |
| W&B API rate limiting on excessive log_scalar calls | Low | Low | W&B batches logs internally. If users hit limits, document best practice: log every N batches, not every batch. Add optional `log_every_n` parameter in future. |
| Thread-safety assumptions break with multiprocessing DataLoader | Medium | Low | Current design handles threading (num_workers with shared memory). If multiprocessing used, would need process-safe queue. Document limitation, test with standard DataLoader. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** User approved comprehensive refactor plan - Phase 1, Task 4 of 18. Critical bug fix that blocks T064-T065 GPU/gradient metrics tracking. Expert analysis identified this as missing method preventing advanced observability.
**Dependencies:** None (standalone fix)
**Estimated Complexity:** Standard (1-hour implementation, straightforward method addition with thread safety)

## Completion Checklist

**Code Quality:**
- [ ] Method follows PEP 8, type hints on all parameters
- [ ] Docstring with Args/Returns/Raises and example usage
- [ ] Thread-safe implementation with lock around shared state
- [ ] Input validation at method entry (fail-fast)

**Testing:**
- [ ] Unit tests pass: `test_log_scalar_with_wandb()`, `test_log_scalar_auto_increment()`
- [ ] Manual test confirms W&B logging works (check W&B dashboard)
- [ ] Thread safety tested with `num_workers=4` DataLoader
- [ ] Edge cases tested: empty metric name, non-numeric value

**Documentation:**
- [ ] CLAUDE.md updated with `log_scalar()` usage example
- [ ] Method docstring includes realistic training loop example
- [ ] Comments explain thread safety design

**Integration:**
- [ ] No breaking changes to existing `log_epoch()` method
- [ ] `get_step_metrics()` returns DataFrame with expected schema
- [ ] Ready for T064/T065 to use for GPU and gradient logging

**Definition of Done:**
Task is complete when `log_scalar()` method exists, passes unit tests, logs to W&B correctly, handles thread safety, and is documented in CLAUDE.md.
