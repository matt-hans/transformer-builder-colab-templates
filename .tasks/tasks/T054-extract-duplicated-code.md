---
id: T054
title: Extract Duplicated Training Loop Code into Shared Functions
status: pending
priority: 1
agent: backend
dependencies: []
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [refactor, code-quality, phase1, technical-debt, blocks-decomposition]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - utils/tier3_training_utilities.py
  - CLAUDE.md

est_tokens: 12000
actual_tokens: null
---

## Description

Extract duplicated training loop code from `test_fine_tuning()` and `test_hyperparameter_search()` into reusable shared functions, eliminating 180 lines of copy-pasted code. This refactor improves maintainability and **must complete before T062** (decomposing `test_fine_tuning()` into smaller functions).

Current state: `utils/tier3_training_utilities.py` has near-identical training/validation loop code in 3 places:
1. `test_fine_tuning()` - 80 lines (train loop + val loop + metrics)
2. `test_hyperparameter_search()` - 80 lines (same logic, different context)
3. `test_glue_benchmark()` - 20 lines (simplified variant)

Total duplication: ~180 lines. Changes require updating 3 locations, high risk of divergence.

Target state: Extracted shared functions:
- `_run_training_epoch(model, dataloader, optimizer, device, pad_token_id) -> float` (avg loss)
- `_run_validation_epoch(model, dataloader, device, pad_token_id) -> dict` (loss + perplexity)
- `_calculate_perplexity(loss: float) -> float`

**Integration Points:**
- Used by `test_fine_tuning()`, `test_hyperparameter_search()`, `test_glue_benchmark()`
- Blocks T062's decomposition (must extract common code first, then decompose what remains)
- Simplifies future changes (gradient clipping, mixed precision, etc. update once)

## Business Context

**User Story:** As a developer, I want training logic defined once, so that bug fixes and improvements apply consistently across all training functions.

**Why This Matters:**
Code duplication creates three problems:
1. **Bug multiplication**: T052's padding fix needs 3 identical changes. Miss one location → partial fix.
2. **Feature divergence**: Adding gradient clipping to `test_fine_tuning()` but forgetting `test_hyperparameter_search()` creates inconsistent behavior.
3. **Maintenance burden**: 180 lines of duplicate code = 3x testing, 3x documentation, 3x cognitive load.

**What It Unblocks:**
- [T062] Decompose `test_fine_tuning()` (can't split until common code extracted)
- Future training improvements apply everywhere (gradient clipping, AMP, LR schedules)
- Easier code reviews (changes to shared functions vs. scattered duplicates)

**Priority Justification:**
P1 (Critical) - Refactoring **prerequisite** for T062. Expert analysis identified this as blocking task—must extract shared code before decomposition, or we'll duplicate even more code during split.

## Acceptance Criteria

- [ ] `_run_training_epoch()` function extracts train loop logic (forward, backward, optimizer step)
- [ ] `_run_validation_epoch()` function extracts validation loop logic (forward, metrics)
- [ ] `_calculate_perplexity()` function extracts PPL calculation: `torch.exp(torch.tensor(loss))`
- [ ] All 3 functions have docstrings with Args/Returns/Raises sections
- [ ] `test_fine_tuning()` refactored to call shared functions, <50 lines of loop code remaining
- [ ] `test_hyperparameter_search()` refactored to call shared functions
- [ ] `test_glue_benchmark()` refactored to call shared functions (or uses simplified variant)
- [ ] Validation: Run test suite, all tests pass (behavior unchanged)
- [ ] Validation: Golden value test - Train with seed 42 before/after refactor, losses identical
- [ ] Code reduction: 180 → 60 lines (120 lines eliminated via extraction)

## Test Scenarios

**Test Case 1: Golden Value Test (Reproducibility)**
- Given: `test_fine_tuning()` with seed 42, train 3 epochs on synthetic data (BEFORE refactor)
- When: Record epoch losses: `[2.453, 2.102, 1.895]`
- Then: Run AFTER refactor with same seed, losses **bit-identical** (confirms behavior unchanged)

**Test Case 2: Shared Function in test_fine_tuning()**
- Given: Refactored `test_fine_tuning()` calling `_run_training_epoch()`
- When: Train for 1 epoch
- Then: Returns average loss, model weights updated, optimizer state advanced

**Test Case 3: Shared Function in test_hyperparameter_search()**
- Given: Refactored `test_hyperparameter_search()` calling same `_run_training_epoch()`
- When: Optuna trial runs training
- Then: Same training logic executes, trial loss matches expected value

**Test Case 4: Validation Epoch with Perplexity**
- Given: Call `_run_validation_epoch()` with validation DataLoader
- When: Epoch completes
- Then: Returns `{'loss': 1.85, 'perplexity': 6.36}`, model in eval mode, no gradients computed

**Test Case 5: Future Change Propagation**
- Given: Add gradient clipping to `_run_training_epoch()`: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- When: Run `test_fine_tuning()` and `test_hyperparameter_search()`
- Then: Both use gradient clipping automatically (change in one place applies everywhere)

**Test Case 6: Edge Case - Empty DataLoader**
- Given: Empty validation DataLoader (no batches)
- When: Call `_run_validation_epoch()`
- Then: Returns `{'loss': float('inf'), 'perplexity': float('inf')}` or raises clear error

**Test Case 7: Code Reduction Metrics**
- Given: Count lines in `test_fine_tuning()` before refactor
- When: Extract to shared functions
- Then: `test_fine_tuning()` loop code reduced from 80 lines to <30 lines (60% reduction)

## Technical Implementation

**Required Components:**

1. **Extract `_run_training_epoch()` in `utils/tier3_training_utilities.py`:**
```python
def _run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_token_id: int = 0,
    log_every: int = 0  # Optional: log batch losses to tracker
) -> float:
    """
    Run one training epoch over the provided DataLoader.

    Handles forward pass, loss calculation (with padding exclusion),
    backward pass, and optimizer step for each batch.

    Args:
        model: PyTorch model to train (will be set to train mode)
        dataloader: Training data batches
        optimizer: Optimizer for parameter updates
        device: Device to run training on (cuda/cpu)
        pad_token_id: Token ID to exclude from loss (default: 0)
        log_every: If >0, print loss every N batches (default: 0 = silent)

    Returns:
        Average loss across all batches (float)

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        >>> avg_loss = _run_training_epoch(model, train_loader, optimizer, device)
        >>> print(f"Training loss: {avg_loss:.4f}")
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        targets = batch['labels'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids)
        logits = _extract_output_tensor(outputs)

        # Calculate loss (exclude padding tokens - T052 fix)
        batch_size, seq_len, vocab_size = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=pad_token_id
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Optional logging
        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else float('inf')


def _run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int = 0
) -> dict:
    """
    Run one validation epoch over the provided DataLoader.

    Evaluates model in eval mode (no gradients), calculates loss and perplexity.

    Args:
        model: PyTorch model to evaluate
        dataloader: Validation data batches
        device: Device to run evaluation on
        pad_token_id: Token ID to exclude from loss

    Returns:
        Dictionary with 'loss' (float) and 'perplexity' (float)

    Example:
        >>> results = _run_validation_epoch(model, val_loader, device)
        >>> print(f"Val loss: {results['loss']:.4f}, PPL: {results['perplexity']:.2f}")
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids)
            logits = _extract_output_tensor(outputs)

            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1),
                ignore_index=pad_token_id
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = _calculate_perplexity(avg_loss)

    return {'loss': avg_loss, 'perplexity': perplexity}


def _calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.

    Perplexity = exp(loss), bounded at 1e6 to prevent overflow.

    Args:
        loss: Cross-entropy loss (float)

    Returns:
        Perplexity (float), capped at 1e6 for numerical stability

    Example:
        >>> ppl = _calculate_perplexity(2.5)
        >>> print(f"Perplexity: {ppl:.2f}")  # ~12.18
    """
    if loss == float('inf'):
        return float('inf')

    ppl = torch.exp(torch.tensor(loss)).item()
    return min(ppl, 1e6)  # Cap at 1M to prevent overflow
```

2. **Refactor `test_fine_tuning()` to use shared functions:**
```python
def test_fine_tuning(
    model: nn.Module,
    config: SimpleNamespace,
    n_epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    use_wandb: bool = False,
    random_seed: int = 42,
    deterministic: bool = False
) -> dict:
    """Fine-tune model using shared training/validation functions."""

    # Setup (seed, device, optimizer, data loaders)
    set_random_seed(random_seed, deterministic=deterministic)
    device = _get_device(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pad_token_id = getattr(config, 'pad_token_id', 0)

    # ... DataLoader creation (with worker_init_fn from T053) ...

    tracker = MetricsTracker(use_wandb=use_wandb)

    for epoch in range(n_epochs):
        # Training epoch (SIMPLIFIED - was 40 lines, now 1 call)
        avg_train_loss = _run_training_epoch(
            model, train_loader, optimizer, device, pad_token_id
        )

        # Validation epoch (SIMPLIFIED - was 30 lines, now 1 call)
        val_results = _run_validation_epoch(
            model, val_loader, device, pad_token_id
        )

        # Log metrics
        tracker.log_epoch(
            epoch=epoch,
            train_metrics={'loss': avg_train_loss},
            val_metrics=val_results,
            learning_rate=optimizer.param_groups[0]['lr']
        )

        print(f"Epoch {epoch + 1}/{n_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_results['loss']:.4f}, "
              f"PPL: {val_results['perplexity']:.2f}")

    return {
        'final_train_loss': avg_train_loss,
        'final_val_loss': val_results['loss'],
        'final_perplexity': val_results['perplexity'],
        'metrics_summary': tracker.get_summary(),
        'best_epoch': tracker.get_summary()['val/loss'].idxmin()
    }
```

3. **Refactor `test_hyperparameter_search()` similarly:**
```python
def test_hyperparameter_search(...) -> dict:
    """Optuna hyperparameter search using shared training functions."""

    def objective(trial):
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            # Use shared functions (eliminates duplication)
            avg_train_loss = _run_training_epoch(model, train_loader, optimizer, device, pad_token_id)
            val_results = _run_validation_epoch(model, val_loader, device, pad_token_id)

            trial.report(val_results['loss'], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_results['loss']

    # ... rest of Optuna setup ...
```

4. **Add golden value test in `tests/test_training_utilities.py`:**
```python
def test_extraction_golden_values():
    """Verify refactor doesn't change training behavior (golden values)"""
    from utils.tier3_training_utilities import test_fine_tuning
    from types import SimpleNamespace

    # Small dummy model for fast testing
    config = SimpleNamespace(vocab_size=100, max_seq_len=32, d_model=64)
    model = DummyTransformer(config)  # Minimal test model

    # Train with fixed seed
    results = test_fine_tuning(
        model, config,
        n_epochs=3,
        learning_rate=1e-3,
        batch_size=2,
        random_seed=42,
        deterministic=True
    )

    # Golden values (recorded before refactor)
    expected_final_loss = 2.1023  # Tolerance: ±0.01
    assert abs(results['final_val_loss'] - expected_final_loss) < 0.01, \
        f"Golden value mismatch: {results['final_val_loss']} != {expected_final_loss}"
```

**Validation Commands:**

```bash
# Test 1: Golden value test (before refactor, record values)
python -c "
from utils.tier3_training_utilities import test_fine_tuning
# ... run training with seed 42, print final loss ...
# Record: final_val_loss = 2.1023
"

# Test 2: Golden value test (after refactor, compare)
pytest tests/test_training_utilities.py::test_extraction_golden_values -v

# Test 3: Code metrics (count lines reduced)
wc -l utils/tier3_training_utilities.py  # Before: ~450 lines
# After refactor: ~330 lines (120 lines eliminated)

# Test 4: Run full test suite
pytest tests/ -v
```

**Code Patterns:**
- Private functions (prefix `_`) for internal utilities not meant for external use
- Return simple types (float, dict) not complex objects (easier to test)
- Docstrings include realistic usage examples
- Error handling: return `float('inf')` for empty dataloaders (graceful degradation)

## Dependencies

**Hard Dependencies** (must be complete first):
- None (standalone refactor)

**Soft Dependencies** (nice to have):
- [T052] Padding token handling fix (extracted functions should include this)
- [T053] DataLoader reproducibility (extracted functions compatible with seeding)

**External Dependencies:**
- None (uses existing PyTorch APIs)

**Blocks Future Tasks:**
- [T061] Replace print statements with logging (easier after extraction - log in one place)
- **[T062] Decompose test_fine_tuning()** - CRITICAL: Must extract shared code first, then decompose what remains. Cannot split before extraction or we'll duplicate even more code.

## Design Decisions

**Decision 1: Extract to Module-Level Functions vs. TrainingEngine Class**
- **Rationale:** Functions simpler for current codebase. Class-based design (TrainingEngine) is overkill until we have >5 training variants.
- **Alternatives:**
  - `TrainingEngine` class with `run_epoch()` method - heavyweight, premature abstraction
  - Keep duplicated code - technical debt, maintenance burden
- **Trade-offs:**
  - Pro: Minimal refactor, easy to understand, testable
  - Con: Less extensible than class (acceptable, can refactor to class later if needed)

**Decision 2: Private Functions (`_` Prefix) vs. Public API**
- **Rationale:** These are internal utilities for tier3 functions, not meant for external use. Private prefix signals intent.
- **Alternatives:**
  - Public functions - pollutes module API, confusing for users
  - Separate module (e.g., `training_loops.py`) - adds complexity, premature
- **Trade-offs:**
  - Pro: Clear separation of public API vs. internals
  - Con: Users can't easily reuse (acceptable, can make public later if demand exists)

**Decision 3: Return Simple Types (float, dict) Not Objects**
- **Rationale:** Easier to test, serialize, and log. Dict with `{'loss': ..., 'perplexity': ...}` clearer than custom `ValidationResult` object.
- **Alternatives:**
  - Return `ValidationResult` dataclass - overkill for 2 fields
  - Return tuple `(loss, ppl)` - unclear field names, error-prone
- **Trade-offs:**
  - Pro: Simple, Pythonic, easy to extend (add more keys to dict)
  - Con: No type safety on dict keys (acceptable, docstring specifies)

**Decision 4: Golden Value Test for Behavior Verification**
- **Rationale:** Unit tests alone can't catch subtle logic changes. Golden value test ensures refactor is behavior-preserving.
- **Alternatives:**
  - Only unit tests - may miss emergent behavior changes
  - No tests - risky, refactor could introduce bugs
- **Trade-offs:**
  - Pro: High confidence refactor didn't change behavior
  - Con: Golden values brittle to upstream changes (e.g., PyTorch version) - acceptable, update as needed

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Refactor introduces subtle behavior change (e.g., optimizer state handling) | High | Low | Golden value test with fixed seed verifies bit-identical results before/after. Run full test suite. Code review focuses on equivalence. |
| Extracted functions don't cover all edge cases (e.g., empty DataLoader) | Medium | Medium | Add explicit edge case tests: empty dataloader → return `float('inf')`. Document in docstring. |
| Future changes to one usage (e.g., `test_fine_tuning()`) need to be abstracted | Medium | Medium | Acceptable trade-off. If `test_fine_tuning()` diverges significantly, can add parameters to shared functions or create specialized variant. Monitor during T062 decomposition. |
| Code reduction target (180 → 60 lines) not achieved | Low | Low | Current analysis shows 80 lines per function * 3 functions = 240 lines before, ~60 shared + 3*20 wrappers = 120 lines after. 120 lines saved. Re-measure after extraction, adjust if needed. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** User approved comprehensive refactor plan - Phase 1, Task 7 of 18. Critical refactor that blocks T062's decomposition. Expert analysis identified 180 lines of duplicated training loop code across 3 functions, creating maintenance burden and bug multiplication risk.
**Dependencies:** None (standalone refactor, but incorporates T052/T053 fixes in extracted code)
**Estimated Complexity:** Complex (4-hour refactor requiring careful extraction, golden value testing, and verification across multiple usage sites)

**Expert Note:** Saves 180 lines of duplicated code. Must complete before T062 (decompose test_fine_tuning) to avoid duplicating even more code during split.

## Completion Checklist

**Code Quality:**
- [ ] `_run_training_epoch()`, `_run_validation_epoch()`, `_calculate_perplexity()` functions created
- [ ] All functions have docstrings with Args/Returns/Examples
- [ ] Functions use `_` prefix to signal internal API
- [ ] Type hints on all parameters

**Testing:**
- [ ] Golden value test passes (before/after losses identical)
- [ ] Unit tests for each extracted function
- [ ] Edge case tests: empty DataLoader, very high loss (PPL overflow)
- [ ] Integration tests: `test_fine_tuning()`, `test_hyperparameter_search()` still work

**Documentation:**
- [ ] Docstrings include realistic usage examples
- [ ] Comments explain design decisions (e.g., why `float('inf')` for empty loader)
- [ ] CLAUDE.md updated if training loop usage changes

**Integration:**
- [ ] `test_fine_tuning()` refactored to use shared functions (<50 lines of loop code)
- [ ] `test_hyperparameter_search()` refactored similarly
- [ ] `test_glue_benchmark()` refactored (or uses simplified variant)
- [ ] Code reduction: 120+ lines eliminated (measured via `wc -l`)

**Definition of Done:**
Task is complete when training loop code extracted to shared functions, all 3 usage sites refactored, golden value test passes (behavior unchanged), and 120+ lines of duplication eliminated.
