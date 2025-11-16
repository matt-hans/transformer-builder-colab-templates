# Code Quality Analysis - T035 (AMP Training) v3

**Agent:** verify-quality (Stage 4)  
**Task:** T035 - Mixed Precision Training with AMP  
**Date:** 2025-11-16  
**Decision:** BLOCK  
**Score:** 62/100

---

## Executive Summary

**CRITICAL ISSUES: 2**
- Function complexity >15 (BLOCKS)
- Code duplication ~70% in training loop branches (BLOCKS)

**File Analysis:**
- `tier3_training_utilities.py`: 831 lines (PASS - was 1008)
- `amp_benchmark.py`: 197 lines (PASS)
- `test_amp_utils.py`: 353 lines (PASS)

**Previous v2 Issues:**
- File size violation: FIXED (1008 → 831 lines)
- Code duplication: STILL PRESENT (93% → 70% in training loops)

---

## CRITICAL: BLOCKS ❌

### 1. Function Complexity Violation
**File:** `utils/tier3_training_utilities.py:99`  
**Function:** `test_fine_tuning()`  
**Cyclomatic Complexity:** 18 (threshold: 15)

**Problem:**
The function has excessive branching due to:
- AMP vs FP32 conditional logic (lines 222-284)
- Multiple nested loops (epochs → batches)
- Optional matplotlib/wandb integrations
- Validation phase
- Visualization code

**Impact:**
- Hard to test all execution paths
- High cognitive load for maintainers
- Risk of bugs in edge cases
- Violates **ZERO TOLERANCE** policy for complexity >15

**Fix Required:**
Extract helper functions to reduce complexity:

```python
# BEFORE (complexity 18):
def test_fine_tuning(...):
    # Setup code
    for epoch in range(n_epochs):
        for i in range(0, len(train_data), batch_size):
            if use_amp:
                # 30+ lines of AMP training logic
            else:
                # 30+ lines of FP32 training logic
        # Validation phase
    # Visualization code

# AFTER (complexity ~8 per function):
def test_fine_tuning(...):
    metrics_tracker = MetricsTracker(use_wandb=use_wandb)
    trainer = _TrainingLoop(model, config, use_amp, optimizer, scheduler)
    
    for epoch in range(n_epochs):
        train_metrics = trainer.train_epoch(train_data, batch_size)
        val_metrics = trainer.validate(val_data)
        metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)
    
    return _build_results(metrics_tracker, trainer)

def _TrainingLoop:
    def train_epoch(self, data, batch_size):
        for batch in self._batch_iterator(data, batch_size):
            loss, accuracy = self._training_step(batch)
            # Track metrics
        return metrics
    
    def _training_step(self, batch):
        # Single unified training step
        # AMP vs FP32 handled internally
        if self.use_amp:
            return self._amp_training_step(batch)
        else:
            return self._fp32_training_step(batch)
```

**Effort:** 3-4 hours  
**Blocking:** YES

---

### 2. Code Duplication in Training Loops
**File:** `utils/tier3_training_utilities.py:222-284`  
**Duplication:** ~70% between AMP and FP32 branches

**Problem:**
The if/else branches (lines 222-284) contain nearly identical logic with only 4 differences:
1. `autocast()` context manager (line 223)
2. `scaler.scale()` instead of direct `backward()` (line 244 vs 277)
3. `scaler.unscale_()` before clipping (line 247)
4. `scaler.step()` and `scaler.update()` (lines 252-253 vs 283)

**Duplicated Code (38 lines in each branch):**
- `_safe_get_model_output()` call
- Shift logits/labels computation
- Loss calculation
- Accuracy computation
- Gradient clipping
- Metric tracking

**Impact:**
- Bug fixes must be applied in two places
- Inconsistencies can creep in
- Violates DRY principle
- Exceeds 10% duplication threshold

**Fix Required:**
Unify the training loop with conditional AMP context:

```python
# UNIFIED APPROACH:
def _training_step(self, batch, optimizer, scaler=None):
    """Unified training step supporting both FP32 and AMP."""
    optimizer.zero_grad()
    
    # Conditional autocast context
    amp_context = autocast() if (scaler is not None) else contextlib.nullcontext()
    
    with amp_context:
        logits = _safe_get_model_output(self.model, batch)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )
    
    # Accuracy computed outside autocast
    with torch.no_grad():
        accuracy = self.compute_accuracy(shift_logits, shift_labels)
    
    # Backward with optional scaling
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    
    # Gradient clipping (works for both)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    
    # Optimizer step with optional scaler
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    return loss.item(), accuracy, grad_norm.item()
```

**Effort:** 2-3 hours  
**Blocking:** YES (duplication >10%)

---

## HIGH: WARNING ⚠️

### 3. Function Complexity Near Threshold
**File:** `utils/tier3_training_utilities.py:626`  
**Function:** `test_benchmark_comparison()`  
**Cyclomatic Complexity:** 14 (threshold: 15)

**Problem:**
Complex function with multiple phases:
- Baseline model loading
- Parameter comparison
- Inference speed benchmarking
- Perplexity comparison
- Visualization

**Impact:**
- Close to blocking threshold
- Could become blocker with minor additions

**Recommendation:**
Extract benchmarking phases into helper functions:
- `_compare_parameters()`
- `_benchmark_inference_speed()`
- `_compare_perplexity()`

**Effort:** 2 hours  
**Blocking:** NO (but recommended)

---

### 4. Function Complexity at Threshold
**File:** `utils/tier3_training_utilities.py:444`  
**Function:** `test_hyperparameter_search()`  
**Cyclomatic Complexity:** 11 (threshold: 10 for warning)

**Problem:**
Nested objective function with conditional search space logic.

**Recommendation:**
Extract objective function to module level:
```python
class OptunaObjective:
    def __init__(self, model_factory, config, train_data, vocab_size, search_space=None):
        self.model_factory = model_factory
        # ... store config
    
    def __call__(self, trial):
        # Objective logic here
```

**Effort:** 1 hour  
**Blocking:** NO

---

## MEDIUM: WARNING ⚠️

### 5. Missing Type Hints
**Files:** All modified files  
**Issue:** Inconsistent type hint coverage (~60%)

**Examples:**
```python
# Missing return type for objective function
def objective(trial):  # Should be: def objective(trial: optuna.Trial) -> float:

# Missing parameter types in test functions
def test_end_to_end_training_with_amp():  # Should mark as pytest fixture
```

**Recommendation:**
Add comprehensive type hints per PEP 484:
```python
from typing import Any, Dict, List, Optional, Callable
import optuna

def test_hyperparameter_search(
    model_factory: Callable[[], nn.Module],
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_trials: int = 10,
    search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
```

**Effort:** 1 hour  
**Blocking:** NO

---

### 6. Long Function (test_fine_tuning)
**File:** `utils/tier3_training_utilities.py:99-441`  
**Lines:** 343 lines (threshold: 50 for warning, 100 for concern)

**Problem:**
Function contains:
- Setup (40 lines)
- Training loop (100 lines)
- Validation loop (30 lines)
- Metrics tracking (40 lines)
- Visualization (80 lines)

**Impact:**
- Hard to navigate
- Mixes concerns (training + visualization)
- Testing requires executing entire function

**Recommendation:**
Extract visualization to separate function:
```python
def test_fine_tuning(...) -> Dict[str, Any]:
    # Training logic only (200 lines)
    return results

def visualize_training_results(results: Dict[str, Any]) -> None:
    """Plot training metrics from test_fine_tuning() results."""
    # Visualization logic (80 lines)
```

**Effort:** 1 hour  
**Blocking:** NO (not over 1000 lines)

---

## METRICS

### Complexity Analysis
| File | Functions | Avg Complexity | Max Complexity | >10 | >15 |
|------|-----------|----------------|----------------|-----|-----|
| tier3_training_utilities.py | 7 | 9.7 | 18 | 3 | 1 ❌ |
| amp_benchmark.py | 1 | 8.0 | 8 | 0 | 0 ✅ |
| test_amp_utils.py | 27 | 2.1 | 4 | 0 | 0 ✅ |

**Status:** FAIL (1 function >15)

### File Size
| File | Lines | Status | Change |
|------|-------|--------|--------|
| tier3_training_utilities.py | 831 | ✅ PASS | -177 from v2 |
| amp_benchmark.py | 197 | ✅ PASS | New module |
| test_amp_utils.py | 353 | ✅ PASS | Comprehensive |

**Status:** PASS (all <1000 lines)

### Code Duplication
| Location | Type | Duplication | Lines | Status |
|----------|------|-------------|-------|--------|
| tier3:222-284 | Training loop if/else | ~70% | 38x2 | ❌ FAIL |
| amp_benchmark | None detected | 0% | - | ✅ PASS |

**Status:** FAIL (>10% duplication in critical path)

### SOLID Principles
| Principle | Status | Notes |
|-----------|--------|-------|
| Single Responsibility | ⚠️ WARNING | test_fine_tuning() does training + visualization |
| Open/Closed | ✅ PASS | Good use of parameters for extension |
| Liskov Substitution | ✅ PASS | No inheritance violations |
| Interface Segregation | ✅ PASS | No fat interfaces |
| Dependency Inversion | ✅ PASS | Uses dependency injection (model, config) |

**Status:** WARNING (SRP violation in test_fine_tuning)

### Code Smells
| Smell | Location | Severity |
|-------|----------|----------|
| Long Method | test_fine_tuning (343 lines) | HIGH |
| Duplicate Code | Lines 222-284 | CRITICAL |
| Feature Envy | None detected | - |
| Shotgun Surgery Risk | AMP changes touch multiple branches | MEDIUM |

---

## REFACTORING ROADMAP

### Priority 1 (CRITICAL - Required for PASS)
1. **Extract Training Loop** (4 hours)
   - Create `TrainingLoop` class
   - Reduce `test_fine_tuning()` complexity to <10
   - Unify AMP/FP32 branches

2. **Eliminate Duplication** (2 hours)
   - Implement unified training step
   - Remove if/else branches in training loop
   - Test parity between FP32 and FP16 paths

### Priority 2 (HIGH - Recommended)
3. **Refactor test_benchmark_comparison** (2 hours)
   - Extract benchmarking helpers
   - Reduce complexity to <10

4. **Add Type Hints** (1 hour)
   - Complete type coverage
   - Add mypy to CI

### Priority 3 (MEDIUM - Nice to Have)
5. **Extract Visualization** (1 hour)
   - Separate plotting from training logic
   - Return data, visualize separately

6. **Improve Test Coverage** (2 hours)
   - Add unit tests for helper functions
   - Test edge cases (no CUDA, OOM scenarios)

**Total Effort:** 12 hours

---

## POSITIVE OBSERVATIONS ✅

1. **File size reduced:** Successfully brought tier3 from 1008 → 831 lines
2. **Clean module extraction:** amp_benchmark.py is well-scoped (197 lines)
3. **Comprehensive tests:** test_amp_utils.py covers edge cases thoroughly
4. **Good documentation:** Docstrings are clear and complete
5. **Proper error handling:** Graceful fallback when CUDA unavailable
6. **No dead code:** All functions are actively used
7. **Naming conventions:** Consistent snake_case, clear function names
8. **Architecture-agnostic:** Helper functions (_detect_vocab_size, etc.) work across model types

---

## FINAL RECOMMENDATION

**DECISION:** BLOCK ❌

**Rationale:**
Despite improvements from v2 (file size reduction), the code contains:
1. **CRITICAL:** Function complexity violation (18 > 15 threshold)
2. **CRITICAL:** Code duplication >10% in core training logic

Both issues violate **ZERO TOLERANCE** blocking criteria defined in the agent mandate.

**Required Actions:**
1. Refactor `test_fine_tuning()` to reduce complexity below 10
2. Eliminate training loop duplication by unifying AMP/FP32 paths
3. Re-run quality analysis to verify compliance

**Estimated Fix Time:** 6 hours (Priority 1 tasks)

**After Fixes:**
Expected score: 85/100 (PASS with minor warnings)

---

## Technical Debt Score: 7/10

**Breakdown:**
- Complexity debt: 3/10 (one function >15, two near threshold)
- Duplication debt: 4/10 (70% duplication in critical path)
- Architecture debt: 0/10 (good separation of concerns)
- Testing debt: 0/10 (comprehensive test coverage)

**Trend:** IMPROVING (v2 had file size violation, now resolved)

---

**Reviewer:** verify-quality (Stage 4 Agent)  
**Next Step:** Assign to developer for Priority 1 refactoring
