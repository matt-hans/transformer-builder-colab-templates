---
id: T058
title: Add Gradient Clipping to Training Loop
status: pending
priority: 2
agent: backend
dependencies: [T051, T052, T053, T057]
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

est_tokens: 6000
actual_tokens: null
---

## Description

Add gradient clipping to training loop to prevent gradient explosions, improving training stability for large models. Clips gradients to maximum norm of 1.0 before optimizer step, standard practice in transformer training.

Current state: No gradient clipping. Large models (GPT-2 355M+) prone to gradient explosions causing NaN losses.

Target state: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` called after `loss.backward()`, before `optimizer.step()`. Logs pre/post-clip norms using T057's `_compute_gradient_norm()` and T051's `log_scalar()`.

## Business Context

**Why This Matters:** Gradient explosions cause training failures. Clipping prevents NaN losses, enables stable training of large models (355M-1.5B parameters).

**What It Unblocks:** Safe training of large models, reduced training failures, professional ML practice.

**Priority:** P2 - Important stability improvement. Depends on T051/T057.

## Acceptance Criteria

- [ ] Gradient clipping added to `_run_training_epoch()` (extracted function from T054)
- [ ] Clips to max_norm=1.0: `clip_grad_norm_(model.parameters(), 1.0)`
- [ ] Logs pre-clip norm using T057's `_compute_gradient_norm()`
- [ ] Logs post-clip norm (return value from `clip_grad_norm_()`)
- [ ] `test_fine_tuning()` adds `max_grad_norm` parameter (default: 1.0, None to disable)
- [ ] Validation: Train large model, verify no NaN losses
- [ ] Unit test: Mock large gradients, verify clipped to max_norm

## Test Scenarios

**Test Case 1:** Large gradients clipped
- Given: Model with gradient norm 15.0
- When: Clip to max_norm=1.0
- Then: Post-clip norm = 1.0

**Test Case 2:** Small gradients unchanged
- Given: Gradient norm 0.5
- When: Clip to max_norm=1.0
- Then: Post-clip norm = 0.5 (no clipping needed)

**Test Case 3:** Logging to MetricsTracker
- Given: Training with clipping enabled
- When: Epoch completes
- Then: W&B shows `gradients/pre_clip_norm` and `gradients/post_clip_norm` plots

**Test Case 4:** Disable clipping
- Given: `test_fine_tuning(max_grad_norm=None)`
- When: Train
- Then: No clipping applied, gradients unconstrained

## Technical Implementation

```python
def _run_training_epoch(
    model, dataloader, optimizer, device, pad_token_id=0,
    max_grad_norm=1.0, tracker=None
):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        logits = _extract_output_tensor(outputs)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=pad_token_id
        )

        loss.backward()

        # Gradient clipping
        if max_grad_norm is not None:
            pre_clip_norm = _compute_gradient_norm(model)
            post_clip_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            if tracker and batch_idx == 0:  # Log first batch of epoch
                tracker.log_scalar('gradients/pre_clip_norm', pre_clip_norm)
                tracker.log_scalar('gradients/post_clip_norm', post_clip_norm.item())

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
```

## Dependencies

**Hard Dependencies:**
- [T051] log_scalar() - For logging clip metrics
- [T052] Padding fix - Independent
- [T053] Reproducibility - Independent
- [T057] Gradient norm utility - Required for pre-clip logging

**Blocks:** None

## Design Decisions

**Decision 1:** max_norm=1.0 default
- **Rationale:** Standard in GPT-2/GPT-3 papers. Works across model sizes.
- **Trade-offs:** Pro: Proven default. Con: May be too aggressive for small models (users can disable)

**Decision 2:** Log first batch only
- **Rationale:** Reduce logging volume. Gradient norms change slowly across batches.
- **Trade-offs:** Pro: Efficient. Con: Less granular (acceptable)

**Decision 3:** Optional clipping (max_grad_norm=None to disable)
- **Rationale:** Flexibility for debugging. Some models don't need clipping.
- **Trade-offs:** Pro: User control. Con: More parameters (acceptable)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| max_norm=1.0 too aggressive for small models | Low | Low | Expose parameter, users can increase to 5.0 or disable. Document in CLAUDE.md. |
| Logging overhead slows training | Low | Very Low | Log only first batch per epoch (negligible overhead). |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 2, Task 4 of 6. Training stability improvement preventing NaN losses. Depends on T051/T057.
**Dependencies:** [T051, T052, T053, T057]
**Estimated Complexity:** Standard (2 hours - integration + testing)

## Completion Checklist

- [ ] Gradient clipping added to `_run_training_epoch()`
- [ ] Logs pre/post-clip norms to MetricsTracker
- [ ] Unit tests pass
- [ ] Validated: Large model trains without NaN losses
- [ ] CLAUDE.md updated with clipping documentation

**Definition of Done:** Gradient clipping prevents explosions, logs norms to W&B, large models train stably.
