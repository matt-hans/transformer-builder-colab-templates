---
id: T065
title: Add Gradient Distribution Logging (Per-Layer Norms, Histograms)
status: pending
priority: 3
agent: backend
dependencies: [T051, T057]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [metrics, gradients, phase5, enhancement, debugging]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - utils/tier3_training_utilities.py
  - utils/training/metrics_tracker.py

est_tokens: 9000
actual_tokens: null
---

## Description

Add gradient distribution logging to track per-layer gradient norms and histograms, enabling diagnosis of vanishing/exploding gradients at specific layers. Logs to W&B for visual analysis.

**Requires T051's `log_scalar()` and T057's `_compute_gradient_norm()`** for per-layer norm calculation.

Current state: Only global gradient norm tracked (T057). Can't diagnose which layer has vanishing gradients (e.g., "Layer 11 has grad norm 0.0001, layer 1 has norm 5.0").

Target state: `_log_gradient_distribution(model, tracker, step)` function:
- Per-layer gradient norms logged: `gradients/layer.0.attn.norm`, `gradients/layer.11.mlp.norm`
- Gradient histogram (optional): Distribution of gradient values across all parameters
- Logged every N epochs (configurable, default: every 5 epochs to reduce overhead)

## Business Context

**Why This Matters:** Debugging training failures requires knowing which layers have gradient issues. Global norm shows "gradients exploding" but not where. Per-layer norms pinpoint problematic layers (e.g., "embedding layer has norm 100x others").

**Priority:** P3 - Advanced debugging tool. Nice-to-have for difficult training scenarios.

## Acceptance Criteria

- [ ] `_log_gradient_distribution(model, tracker, step)` function created
- [ ] Computes per-layer gradient norms: `layer_norm = layer.grad.norm(2).item()` for each module
- [ ] Logs to MetricsTracker: `tracker.log_scalar(f'gradients/{layer_name}/norm', layer_norm, step)`
- [ ] Optional histogram: `wandb.log({'gradients/histogram': wandb.Histogram(all_grads)}, step=step)`
- [ ] Integrated into training loop: called every 5 epochs (configurable via `log_grad_dist_every`)
- [ ] Validation: W&B dashboard shows per-layer gradient norm plots
- [ ] Validation: Histogram shows distribution (most grads in -0.1 to 0.1 range for healthy training)
- [ ] Handles modules without gradients (frozen layers) gracefully

## Test Scenarios

**Test Case 1:** Per-layer norm logging
- Given: 12-layer transformer, train for 10 epochs
- When: Check W&B "gradients/layer.0.attn/norm" plot
- Then: Shows gradient norm for layer 0 attention across epochs

**Test Case 2:** Vanishing gradient detection
- Given: Layer 11 has vanishing gradients (norm < 0.001)
- When: Inspect per-layer plots
- Then: Clear visualization shows layer 11 norm 100x smaller than layer 0

**Test Case 3:** Histogram visualization
- Given: Healthy training (grads in -0.1 to 0.1)
- When: Check W&B "gradients/histogram"
- Then: Bell curve centered at 0, most values in [-0.1, 0.1] range

**Test Case 4:** Logging frequency control
- Given: `log_grad_dist_every=5`, train for 10 epochs
- When: Count logged gradient distributions
- Then: Logged at epochs 0, 5, 10 (3 times, not 11)

## Technical Implementation

```python
import torch
import torch.nn as nn
from typing import Optional

def _log_gradient_distribution(
    model: nn.Module,
    tracker: MetricsTracker,
    step: int,
    log_histogram: bool = False
) -> None:
    """Log per-layer gradient norms and optional histogram (requires T051, T057)."""
    all_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Per-layer norm
            layer_grad_norm = param.grad.norm(2).item()
            tracker.log_scalar(f'gradients/{name}/norm', layer_grad_norm, step=step)

            # Collect for histogram
            if log_histogram:
                all_grads.extend(param.grad.detach().cpu().flatten().tolist())

    # Optional histogram (requires wandb)
    if log_histogram and tracker.use_wandb and all_grads:
        try:
            import wandb
            wandb.log({'gradients/histogram': wandb.Histogram(all_grads)}, step=step)
        except ImportError:
            pass


# Integrate into training loop
def test_fine_tuning(
    model, config, n_epochs=10, ...,
    log_grad_dist_every: int = 5  # NEW parameter
):
    tracker = MetricsTracker(use_wandb=use_wandb)

    for epoch in range(n_epochs):
        train_loss = _run_training_epoch(model, train_loader, optimizer, ...)

        # Log gradient distribution every N epochs
        if epoch % log_grad_dist_every == 0:
            _log_gradient_distribution(model, tracker, step=epoch, log_histogram=True)

        val_results = _run_validation_epoch(model, val_loader, ...)
        tracker.log_epoch(epoch, ...)

    return ...
```

## Dependencies

**Hard Dependencies:**
- [T051] Add log_scalar() method - **Required** for per-layer logging
- [T057] Gradient norm utility - Used as reference implementation

**Soft Dependencies:**
- wandb (for histogram visualization) - gracefully skipped if not available

**Blocks:** None

## Design Decisions

**Decision 1:** Log every 5 epochs (not every epoch)
- **Rationale:** Per-layer logging overhead (~0.1s per epoch for 12-layer model). Every 5 epochs balances detail vs. performance.
- **Trade-offs:** Pro: Low overhead. Con: Less granular (acceptable, gradients change slowly)

**Decision 2:** Optional histogram (not default)
- **Rationale:** Histogram computation expensive (all grads → CPU → list). Useful for debugging, not routine monitoring.
- **Trade-offs:** Pro: Available when needed. Con: Requires explicit enable (acceptable, power-user feature)

**Decision 3:** Use parameter names (not module names)
- **Rationale:** `transformer.h.0.attn.c_attn.weight` more specific than `transformer.h.0.attn`.
- **Trade-offs:** Pro: Precise diagnosis. Con: Verbose names (acceptable, W&B groups hierarchically)

**Decision 4:** Log to W&B (not just MetricsTracker)
- **Rationale:** Per-layer norms create 100+ metrics (12 layers × 8 params/layer). W&B handles this scale well, MetricsTracker DataFrame would be unwieldy.
- **Trade-offs:** Pro: Scalable, good visualization. Con: Requires W&B (acceptable for advanced feature)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Logging overhead slows training | Medium | Low | Log every 5 epochs (not every epoch). Skip histogram unless debugging. Profile and adjust frequency if needed. |
| Too many metrics overwhelm W&B dashboard | Medium | Medium | W&B groups metrics hierarchically (gradients/layer.0/...). Use dashboard filters. Document in CLAUDE.md: "Filter by 'gradients/layer.0' to view single layer." |
| Histogram computation OOMs on large models | Low | Low | Skip histogram for models >1B params. Document limitation. Add check: `if num_params < 500M: log_histogram=True` |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 5, Task 2 of 2. Advanced gradient monitoring for debugging training issues. Requires T051's log_scalar() and T057's gradient norm utility.
**Dependencies:** [T051] log_scalar(), [T057] gradient norm utility
**Estimated Complexity:** Standard (3 hours - per-layer iteration + W&B integration)

## Completion Checklist

- [ ] _log_gradient_distribution() function created
- [ ] Per-layer gradient norms logged
- [ ] Optional histogram logged to W&B
- [ ] Integrated into training loop (every 5 epochs)
- [ ] W&B dashboard shows per-layer norm plots
- [ ] Histogram visualization works
- [ ] No performance degradation (<1% overhead)

**Definition of Done:** Per-layer gradient norms logged, W&B shows plots for debugging vanishing/exploding gradients, histogram optional.
