---
id: T062
title: Decompose test_fine_tuning() into Smaller Functions
status: pending
priority: 3
agent: backend
dependencies: [T054, T061]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [refactor, code-quality, phase4, decomposition]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - utils/tier3_training_utilities.py

est_tokens: 18000
actual_tokens: null
---

## Description

Decompose monolithic `test_fine_tuning()` (currently 200+ lines after T054's extraction) into smaller focused functions for setup, training, validation, and metrics. Improves readability and testability.

**CRITICAL:** Must run **after** T054 extracts shared training/validation loop code. Decomposing before extraction would create even more duplicated code.

Current state: `test_fine_tuning()` handles everything - setup, data loading, training loop coordination, metrics aggregation, result formatting. 200+ lines, hard to test components in isolation.

Target state: Decomposed into:
- `_setup_training(model, config, ...)` → (optimizer, scheduler, data_loaders)
- `_train_model(model, optimizer, scheduler, loaders, n_epochs, ...)` → metrics_df
- `_format_results(metrics_df)` → final results dict
- Main `test_fine_tuning()` orchestrates these functions (~50 lines)

## Business Context

**Why This Matters:** 200-line functions are hard to understand, test, and modify. Decomposition into 4 focused functions (50 lines each) improves maintainability.

**Priority:** P3 - Code quality improvement. Not critical, but makes future changes easier.

## Acceptance Criteria

- [ ] `_setup_training()` function extracts setup logic (optimizer, scheduler, dataloaders)
- [ ] `_train_model()` function contains main training loop orchestration
- [ ] `_format_results()` function formats final metrics dictionary
- [ ] `test_fine_tuning()` orchestrates these functions in ~50 lines
- [ ] Golden value test: Train before/after decomposition, losses identical
- [ ] Unit tests for individual functions (can mock components)
- [ ] Code reduction: 200 lines → 4 functions of ~50 lines each

## Test Scenarios

**Test Case 1:** Golden value preservation
- Given: Train with seed 42 before decomposition, record losses
- When: Decompose and train with same seed
- Then: Losses bit-identical (confirms behavior unchanged)

**Test Case 2:** Individual function testing
- Given: `_setup_training()` called with mock model/config
- When: Inspect returned optimizer, scheduler, dataloaders
- Then: Correctly configured (can test setup in isolation)

**Test Case 3:** Main function orchestration
- Given: Refactored `test_fine_tuning()` calls subfunctions
- When: Run full training
- Then: Same results as before, cleaner code

## Technical Implementation

```python
def _setup_training(model, config, learning_rate, weight_decay, batch_size, ...):
    """Setup optimizer, scheduler, and data loaders for training."""
    optimizer = torch.optim.AdamW(
        _get_optimizer_grouped_parameters(model, weight_decay),
        lr=learning_rate
    )

    total_steps = n_epochs * len(train_dataset) // batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps * 0.1, total_steps)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, ...)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, ...)

    return optimizer, scheduler, train_loader, val_loader


def _train_model(model, optimizer, scheduler, train_loader, val_loader, n_epochs, ...):
    """Execute training loop, return metrics DataFrame."""
    tracker = MetricsTracker(use_wandb=use_wandb)

    for epoch in range(n_epochs):
        train_loss = _run_training_epoch(model, train_loader, optimizer, ...)
        val_results = _run_validation_epoch(model, val_loader, ...)

        if scheduler:
            scheduler.step()

        tracker.log_epoch(epoch, train_metrics={'loss': train_loss}, val_metrics=val_results, ...)

    return tracker.get_summary()


def _format_results(metrics_df):
    """Format final results dictionary from metrics DataFrame."""
    return {
        'final_train_loss': metrics_df['train/loss'].iloc[-1],
        'final_val_loss': metrics_df['val/loss'].iloc[-1],
        'final_perplexity': metrics_df['val/perplexity'].iloc[-1],
        'metrics_summary': metrics_df,
        'best_epoch': metrics_df['val/loss'].idxmin()
    }


def test_fine_tuning(model, config, n_epochs=10, learning_rate=5e-5, ...):
    """Fine-tune transformer model with training loop decomposition."""
    set_random_seed(random_seed, deterministic)

    optimizer, scheduler, train_loader, val_loader = _setup_training(
        model, config, learning_rate, weight_decay, batch_size, ...
    )

    metrics_df = _train_model(
        model, optimizer, scheduler, train_loader, val_loader, n_epochs, ...
    )

    return _format_results(metrics_df)
```

## Dependencies

**Hard Dependencies:**
- [T054] Extract duplicated code - **MUST complete first**. Decomposing before extraction creates more duplication.
- [T061] Replace prints with logging - Makes decomposed functions cleaner

**Blocks:** None

## Design Decisions

**Decision 1:** Decompose after T054 extraction, not before
- **Rationale:** T054 extracts shared training/validation loops. Decomposing first would duplicate those 80 lines across multiple functions.
- **Trade-offs:** Pro: Minimal duplication. Con: Must wait for T054 (acceptable, correct order)

**Decision 2:** Three subfunctions (setup, train, format)
- **Rationale:** Clean separation of concerns. Each function has single responsibility.
- **Trade-offs:** Pro: Testable, readable. Con: More functions to maintain (acceptable, better than monolith)

**Decision 3:** Golden value test required
- **Rationale:** Large refactor requires verification that behavior unchanged.
- **Trade-offs:** Pro: High confidence. Con: Brittle to upstream changes (acceptable, update as needed)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Decomposing before T054 creates duplication | High | Low (blocked by dependency) | Enforce dependency in manifest. Document in task description: "MUST run after T054." |
| Golden value test fails due to non-determinism | Medium | Low | Use T053's deterministic mode (seed + cudnn.deterministic=True) to ensure bit-identical results. |
| Over-decomposition (too many small functions) | Low | Low | Stick to 3 subfunctions (setup, train, format). Don't decompose further unless clear benefit. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 4, Task 1 of 2. Code quality refactor decomposing 200-line function into 4 focused functions. **MUST run after T054** to avoid duplicating extracted code.
**Dependencies:** [T054] Extract shared code first, [T061] Logging refactor (cleaner decomposed code)
**Estimated Complexity:** Complex (6 hours - careful decomposition + golden value testing)

**Expert Note:** Golden value test required - run with fixed seed before/after refactor, verify bit-identical losses.

## Completion Checklist

- [ ] Three subfunctions created: _setup_training, _train_model, _format_results
- [ ] Main test_fine_tuning() orchestrates in ~50 lines
- [ ] Golden value test passes (before/after losses identical)
- [ ] Unit tests for subfunctions
- [ ] Code metrics: 200 lines → 4×50 lines (cleaner structure)

**Definition of Done:** test_fine_tuning() decomposed, golden value test passes, unit tests for subfunctions pass.
