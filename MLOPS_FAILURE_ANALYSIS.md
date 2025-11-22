# MLOps Pipeline Failure Analysis: KeyError 'loss_history'

**Date**: 2025-01-22
**Severity**: HIGH (Production training completed but metrics reporting failed)
**Status**: Root cause identified, fix in progress

---

## Executive Summary

A production training run completed successfully (10 epochs, 166 minutes) but failed during post-training metrics reporting with `KeyError: 'loss_history'`. Investigation reveals a **systemic API version mismatch** between the modern v4.0 Trainer engine and legacy notebook expectations.

**Root Cause**: Notebook expects deprecated `loss_history` list field; Trainer v4.0 returns `metrics_summary` DataFrame.

**Impact**: Training succeeded, metrics were tracked internally, but final reporting crashed. No data loss, but poor user experience.

---

## 1. Architecture Analysis

### 1.1 Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Notebook (Cell 31)                  │
│  - Calls trainer.train()                                         │
│  - Expects: results['loss_history'] (legacy API)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│            Trainer.train() (utils/training/engine/trainer.py)    │
│  1. Executes training epochs                                     │
│  2. Delegates metrics to MetricsTracker                          │
│  3. Returns _format_results()                                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Trainer._format_results() (Line 1061-1084)          │
│  Returns:                                                        │
│    ✅ 'metrics_summary': DataFrame (train/loss, val/loss, etc)  │
│    ✅ 'best_epoch': int                                          │
│    ✅ 'final_loss': float                                        │
│    ✅ 'checkpoint_path': str                                     │
│    ✅ 'training_time': float                                     │
│    ❌ 'loss_history': MISSING (deprecated field)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│          MetricsTracker (utils/training/metrics_tracker.py)      │
│  - Stores metrics_history: List[Dict]                           │
│  - Converts to DataFrame via get_summary()                       │
│  - Supports W&B logging + offline storage                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Metrics Tracking Architecture

**MetricsTracker Design** (metrics_tracker.py):
- **Internal storage**: `self.metrics_history` - List of dicts per epoch
- **Public API**: `get_summary()` → pandas DataFrame
- **Columns**: `epoch`, `train/loss`, `train/perplexity`, `train/accuracy`, `val/loss`, `val/perplexity`, `val/accuracy`, `learning_rate`, `gradient_norm`, `epoch_duration`, `system/gpu_memory_mb`, `system/gpu_utilization`

**Trainer Integration** (trainer.py L1061-1084):
```python
def _format_results(self, training_time: float) -> Dict[str, Any]:
    metrics_df = self.metrics_tracker.get_summary()

    results = {
        'metrics_summary': metrics_df,              # ✅ New API (v4.0)
        'best_epoch': ...,
        'final_loss': metrics_df['train/loss'].iloc[-1],
        'checkpoint_path': ...,
        'training_time': training_time
    }
    return results
```

**Notebook Expectation** (training.ipynb Cell 31):
```python
# ❌ Legacy API (v3.x)
print(f"   Train Loss: {results['loss_history'][-1]:.4f}")
```

---

## 2. Root Cause Classification

### 2.1 Failure Mode: API Version Mismatch

**Type**: Backward-incompatible API change
**Location**: Interface contract between `Trainer.train()` and notebook consumer

**Timeline**:
1. **v3.x (Legacy)**: Trainer returned `loss_history: List[float]`
2. **v4.0 (Modern)**: Trainer returns `metrics_summary: DataFrame` (more comprehensive)
3. **Notebook**: Still expects v3.x API → KeyError at runtime

### 2.2 Why This Wasn't Caught Earlier

**Missing Integration Tests**:
- ❌ No test validates `Trainer.train()` return value schema
- ❌ No test ensures notebook compatibility with Trainer API
- ❌ No version compatibility matrix documented

**Gradual Migration**:
- Modern `Trainer` implemented (Phase 0 refactor)
- Notebook partially updated (uses `Trainer`)
- **BUT**: Metrics reporting code not updated to match new API

---

## 3. Evidence Analysis

### 3.1 Training Success Evidence

**From error output**:
```
✅ Training finished in 9963.5s (166.1 min)
```

- Training loop completed all 10 epochs
- Checkpoints saved (epochs 4, 9)
- Metrics printed during training (train_loss, val_loss, val_ppl, val_acc)
- Total time: 166 minutes (reasonable for production run)

**Inference**: Training pipeline is **fully functional**. Only reporting layer failed.

### 3.2 W&B Integration Failure

**Error message**:
```
You must call wandb.init() before wandb.log()
```

**Analysis**:
- MetricsTracker L316-321: Catches W&B exceptions gracefully
- W&B initialization issue is **separate** from loss_history bug
- Likely cause: `use_wandb=True` but `wandb.init()` never called in notebook

**Location of W&B init** (expected):
- Modern API: Should be in notebook before creating Trainer
- Legacy API: TrainingCoordinator handled W&B setup automatically

---

## 4. Detailed API Contract Analysis

### 4.1 Current Trainer Return Schema (v4.0)

**File**: `utils/training/engine/trainer.py` L1061-1084

```python
{
    'metrics_summary': pd.DataFrame,  # Columns: epoch, train/loss, val/loss, etc.
    'best_epoch': int,                 # Epoch with lowest val/loss
    'final_loss': float,               # Last epoch train/loss
    'checkpoint_path': str | None,     # Path to best checkpoint
    'training_time': float             # Total seconds
}
```

### 4.2 Notebook Expectation (Legacy v3.x)

**File**: `training.ipynb` Cell 31 L2307

```python
results['loss_history']  # ❌ MISSING - expects List[float]
results['best_epoch']     # ✅ PRESENT
results['metrics_summary'] # ✅ PRESENT (but not used correctly)
```

### 4.3 Data Availability Analysis

**Question**: Can we recover `loss_history` from existing data?

**Answer**: YES - Three recovery paths available:

**Path 1: Extract from metrics_summary** (recommended)
```python
loss_history = results['metrics_summary']['train/loss'].tolist()
```

**Path 2: Access MetricsTracker directly**
```python
loss_history = [m['train/loss'] for m in trainer.metrics_tracker.metrics_history]
```

**Path 3: Read from checkpoints**
```python
checkpoint = torch.load(results['checkpoint_path'])
loss_history = checkpoint['custom_state']['metrics_history']
```

---

## 5. Production Impact Assessment

### 5.1 Severity Analysis

**Data Loss**: None
- Metrics stored in `metrics_tracker.metrics_history`
- Checkpoints saved successfully
- All training artifacts intact

**User Experience**: Poor
- Training succeeded but appears to fail
- No final metrics displayed
- Confusing error message

**Business Impact**: Medium
- Training resources not wasted (10 epochs completed)
- Metrics can be manually extracted from checkpoints
- BUT: User workflow interrupted, manual intervention required

### 5.2 Affected Use Cases

**Scenarios that fail**:
1. ✅ Training completes → ❌ Final metrics display fails
2. ✅ Checkpoints saved → ❌ Notebook crashes before saving final model
3. ✅ W&B metrics logged during training → ❌ Post-training summary fails

**Scenarios that work**:
1. ✅ Training progress (per-epoch metrics printed)
2. ✅ Checkpoint saving
3. ✅ W&B logging (if `wandb.init()` called correctly)

---

## 6. Failure Recovery Strategy

### 6.1 Immediate Recovery (Manual)

If you're currently stuck with this error, recover metrics via:

```python
# Option A: Extract from metrics_summary
if 'metrics_summary' in results:
    df = results['metrics_summary']
    loss_history = df['train/loss'].tolist()
    print(f"Final Train Loss: {loss_history[-1]:.4f}")
    print(f"Final Val Loss: {df['val/loss'].iloc[-1]:.4f}")

# Option B: Load from checkpoint
checkpoint = torch.load(f"{training_config.checkpoint_dir}/checkpoint_epoch0009.pt")
metrics_history = checkpoint['custom_state']['metrics_history']
for m in metrics_history[-3:]:  # Last 3 epochs
    print(f"Epoch {m['epoch']}: train_loss={m['train/loss']:.4f}")
```

### 6.2 Automated Fix (Code Changes Required)

See Section 7 for production-ready fix implementation.

---

## 7. Production-Ready Fix

### 7.1 Fix Strategy: Backward-Compatible Adapter

**Goal**: Support both legacy and modern APIs without breaking existing code.

**Approach**:
1. Add `loss_history` to Trainer return value (derived from `metrics_summary`)
2. Add deprecation warning for legacy field
3. Update notebook to use modern API
4. Provide migration guide

### 7.2 Implementation Plan

**File**: `utils/training/engine/trainer.py`

```python
def _format_results(self, training_time: float) -> Dict[str, Any]:
    """Format comprehensive training results with backward compatibility."""
    metrics_df = self.metrics_tracker.get_summary()

    # Modern API (v4.0+)
    results = {
        'metrics_summary': metrics_df,
        'best_epoch': self.metrics_tracker.get_best_epoch('val/loss', 'min') if 'val/loss' in metrics_df.columns else 0,
        'final_loss': metrics_df['train/loss'].iloc[-1] if not metrics_df.empty else 0.0,
        'checkpoint_path': str(self.checkpoint_manager.get_best()) if self.checkpoint_manager.get_best() else None,
        'training_time': training_time
    }

    # Legacy compatibility (v3.x) - DEPRECATED
    # Derive loss_history from metrics_summary for backward compatibility
    if not metrics_df.empty:
        results['loss_history'] = metrics_df['train/loss'].tolist()
        results['val_loss_history'] = metrics_df['val/loss'].tolist() if 'val/loss' in metrics_df.columns else []

        # Add deprecation warning (only log once, not print to avoid spam)
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "DEPRECATION WARNING: 'loss_history' and 'val_loss_history' fields are deprecated. "
            "Use 'metrics_summary' DataFrame instead. "
            "Legacy fields will be removed in v5.0. "
            "See MIGRATION_GUIDE.md for details."
        )
    else:
        results['loss_history'] = []
        results['val_loss_history'] = []

    logger.info(f"Training complete in {training_time:.1f}s")
    logger.info(f"Best model at epoch {results['best_epoch']}")

    return results
```

**File**: `training.ipynb` Cell 31 (Updated)

```python
# Modern API (v4.0+) - RECOMMENDED
if 'metrics_summary' in results and not results['metrics_summary'].empty:
    final_metrics = results['metrics_summary'].iloc[-1]
    print(f"   Train Loss: {final_metrics['train/loss']:.4f}")
    print(f"   Val Loss: {final_metrics['val/loss']:.4f}")
    print(f"   Perplexity: {final_metrics['val/perplexity']:.2f}")
    print(f"   Accuracy: {final_metrics['val/accuracy']:.4f}")

# Legacy fallback (v3.x) - DEPRECATED
elif 'loss_history' in results and results['loss_history']:
    print(f"   Train Loss: {results['loss_history'][-1]:.4f}")
    print("   ⚠️  Using deprecated 'loss_history' field. Update to 'metrics_summary' for full metrics.")
```

### 7.3 Rollback Plan

If the fix introduces regressions:

**Rollback Steps**:
1. Revert `trainer.py` changes (remove `loss_history` additions)
2. Keep notebook using `metrics_summary` (safer, modern API)
3. Document manual workaround for v3.x users

**Git Commands**:
```bash
# If fix is in last commit
git revert HEAD

# If fix is in multiple commits
git revert <commit-sha>

# Emergency: reset to known good state
git reset --hard <commit-before-fix>
```

---

## 8. W&B Integration Issue (Secondary)

### 8.1 Error Analysis

**Error**: `You must call wandb.init() before wandb.log()`

**Root Cause**:
- MetricsTracker L316-321 calls `wandb.log()` if `use_wandb=True`
- Notebook doesn't call `wandb.init()` before creating Trainer
- Trainer doesn't initialize W&B automatically (by design - separation of concerns)

### 8.2 Fix Implementation

**File**: `training.ipynb` - Add W&B initialization cell before training

```python
# Cell 29.5 (NEW) - W&B Initialization
if training_config.wandb_project:
    try:
        import wandb

        # Initialize W&B run
        wandb.init(
            project=training_config.wandb_project,
            entity=training_config.wandb_entity,
            name=training_config.run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=training_config.to_dict(),
            notes=training_config.notes,
            tags=["training", "transformer-builder", "v4.0"]
        )

        print(f"✅ W&B initialized: {wandb.run.name}")
        print(f"   View run: {wandb.run.url}")
    except Exception as e:
        print(f"⚠️  W&B initialization failed: {e}")
        print("   Training will continue without W&B logging.")
        training_config.wandb_project = None  # Disable W&B
else:
    print("ℹ️  W&B logging disabled (wandb_project not set)")
```

**Alternative**: Auto-initialize in Trainer (more opinionated)

```python
# In Trainer._setup_metrics()
def _setup_metrics(self) -> MetricsTracker:
    use_wandb = self.training_config.wandb_project is not None

    # Auto-initialize W&B if configured
    if use_wandb:
        try:
            import wandb
            if wandb.run is None:  # Not already initialized
                wandb.init(
                    project=self.training_config.wandb_project,
                    entity=self.training_config.wandb_entity,
                    name=self.training_config.run_name,
                    config=self.training_config.to_dict()
                )
        except Exception as e:
            logger.warning(f"W&B auto-initialization failed: {e}. Disabling W&B logging.")
            use_wandb = False

    return MetricsTracker(
        use_wandb=use_wandb,
        gradient_accumulation_steps=self.training_config.gradient_accumulation_steps
    )
```

---

## 9. Integration Testing Recommendations

### 9.1 Missing Test Coverage

**Current State**: No integration tests for `Trainer.train()` return contract

**Required Tests**:

```python
# tests/integration/test_trainer_api_contract.py

import pytest
from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec

def test_trainer_return_value_schema():
    """Ensure Trainer.train() returns expected schema."""
    # Setup minimal training scenario
    config = TrainingConfig(epochs=1, batch_size=2, max_train_samples=10, max_val_samples=5)
    task_spec = TaskSpec.text_tiny()

    trainer = Trainer(model, config, config, task_spec, tokenizer=tokenizer)
    results = trainer.train(train_data, val_data)

    # Modern API (v4.0+)
    assert 'metrics_summary' in results
    assert isinstance(results['metrics_summary'], pd.DataFrame)
    assert 'train/loss' in results['metrics_summary'].columns
    assert 'val/loss' in results['metrics_summary'].columns

    assert 'best_epoch' in results
    assert isinstance(results['best_epoch'], int)

    assert 'final_loss' in results
    assert isinstance(results['final_loss'], float)

    assert 'checkpoint_path' in results

    assert 'training_time' in results
    assert isinstance(results['training_time'], float)

    # Legacy compatibility (v3.x) - DEPRECATED
    assert 'loss_history' in results
    assert isinstance(results['loss_history'], list)
    assert len(results['loss_history']) == config.epochs

def test_notebook_metrics_reporting_flow():
    """Simulate notebook Cell 31 metrics reporting."""
    # Train model
    results = trainer.train(train_data, val_data)

    # Simulate notebook code (should NOT raise KeyError)
    try:
        # Legacy path (v3.x)
        final_train_loss = results['loss_history'][-1]

        # Modern path (v4.0+)
        if 'metrics_summary' in results:
            final_metrics = results['metrics_summary'].iloc[-1]
            val_loss = final_metrics['val/loss']

        # Both should work
        assert final_train_loss is not None
        assert val_loss is not None

    except KeyError as e:
        pytest.fail(f"Notebook metrics reporting failed: {e}")

def test_wandb_integration():
    """Ensure W&B logging doesn't crash training."""
    # Test with W&B enabled but not initialized
    config = TrainingConfig(wandb_project="test-project")
    trainer = Trainer(model, config, config, task_spec, tokenizer=tokenizer)

    # Should complete without crashing (W&B failures are caught)
    results = trainer.train(train_data, val_data)
    assert results is not None
```

### 9.2 CI/CD Integration

**Add to GitHub Actions** (.github/workflows/test.yml):

```yaml
- name: Run integration tests
  run: |
    pytest tests/integration/test_trainer_api_contract.py -v

- name: Validate notebook compatibility
  run: |
    jupyter nbconvert --to notebook --execute training.ipynb --ExecutePreprocessor.timeout=600
```

---

## 10. Migration Guide for Users

### 10.1 For v3.x Users Upgrading to v4.0

**Breaking Change**: `Trainer.train()` return value schema changed.

**Old Code (v3.x)**:
```python
results = trainer.train(train_data, val_data)
final_loss = results['loss_history'][-1]
```

**New Code (v4.0+)**:
```python
results = trainer.train(train_data, val_data)
final_loss = results['metrics_summary']['train/loss'].iloc[-1]

# Or use convenience field (backward compatible)
final_loss = results['final_loss']
```

**Full Metrics Access**:
```python
# Get DataFrame with all metrics
df = results['metrics_summary']

# Available columns:
# - epoch
# - train/loss, train/perplexity, train/accuracy
# - val/loss, val/perplexity, val/accuracy
# - learning_rate, gradient_norm, epoch_duration
# - system/gpu_memory_mb, system/gpu_utilization

# Plot training curve
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['train/loss'], label='Train')
plt.plot(df['epoch'], df['val/loss'], label='Val')
plt.legend()
plt.show()
```

### 10.2 Deprecation Timeline

- **v4.0** (current): `loss_history` added for backward compatibility (with warning)
- **v4.5** (next minor): Warning escalated to error in strict mode
- **v5.0** (next major): `loss_history` removed entirely

---

## 11. Recommendations

### 11.1 Immediate Actions (This Week)

1. ✅ **Implement backward-compatible fix** (Section 7.2)
   - Add `loss_history` to Trainer return value
   - Update notebook to use modern API

2. ✅ **Add W&B initialization** (Section 8.2)
   - Add W&B init cell to notebook
   - OR auto-initialize in Trainer

3. ✅ **Add integration tests** (Section 9.1)
   - Test Trainer return value schema
   - Test notebook compatibility

### 11.2 Short-Term Improvements (This Month)

4. **Document API versioning**
   - Create CHANGELOG.md with breaking changes
   - Add API version to docstrings

5. **Add runtime version checks**
   - Trainer logs API version used
   - Notebook checks compatibility

6. **Improve error messages**
   - Catch KeyError and suggest migration
   - Link to migration guide

### 11.3 Long-Term Architecture (Next Quarter)

7. **Formalize API contract**
   - Use TypedDict or dataclass for return values
   - Enforce schema with runtime validation

8. **Version compatibility matrix**
   - Document which Trainer versions work with which notebooks
   - Automated compatibility testing in CI

9. **Semantic versioning for training API**
   - Major version bump for breaking changes
   - Deprecation warnings for 1+ minor versions before removal

---

## 12. Conclusion

**Root Cause**: API version mismatch between Trainer v4.0 (returns `metrics_summary`) and notebook (expects `loss_history`).

**Fix**: Add backward-compatible `loss_history` field to Trainer while migrating notebook to modern API.

**Impact**: Low risk fix with high user experience improvement.

**Validation**: Integration tests will prevent future API mismatches.

**Timeline**: Fix can be deployed immediately (no breaking changes).

---

## Appendix A: Code References

**Key Files**:
- `utils/training/engine/trainer.py` L1061-1084: `_format_results()`
- `utils/training/metrics_tracker.py` L364-380: `get_summary()`
- `training.ipynb` Cell 31 L2307: Metrics reporting
- `CLAUDE.md` L271-291: API documentation (needs update)

**Related Issues**:
- v4.0 migration tracking: #TBD
- W&B integration improvements: #TBD

---

**Analyst**: Claude Code (MLOps Engineer)
**Reviewed**: Pending
**Approved**: Pending
