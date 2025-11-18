---
id: T056
title: Implement Weight Decay Exclusion for Biases and LayerNorm
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

est_tokens: 6000
actual_tokens: null
---

## Description

Exclude bias parameters and LayerNorm weights from weight decay, following best practices from BERT/GPT training. Applying weight decay to biases/LayerNorm degrades performance—these parameters should not be regularized.

Current state: `AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)` applies weight decay to all parameters uniformly. This incorrectly regularizes biases and LayerNorm parameters.

Target state: `_get_optimizer_grouped_parameters()` utility creates parameter groups:
- Group 1: Weights (excluding LayerNorm) → weight_decay=0.01
- Group 2: Biases + LayerNorm weights → weight_decay=0.0

**Integration Points:**
- Used in `test_fine_tuning()`, `test_hyperparameter_search()`
- Works with T052's loss calculation (both improve final quality)
- Standard in HuggingFace transformers library

## Business Context

**User Story:** As an ML practitioner, I want bias/LayerNorm parameters excluded from weight decay, so that my models achieve better final accuracy matching published results.

**Why This Matters:**
Research shows excluding biases/LayerNorm from weight decay improves final accuracy by 0.5-2%. All major transformer papers (BERT, GPT-2/3, T5) use this technique. Without it, models underperform published baselines.

**What It Unblocks:**
- Match published baseline performance (reproducible results)
- Professional training configuration (industry standard)
- Better final model quality (0.5-2% PPL improvement)

**Priority Justification:**
P2 (Important) - Training best practice that improves results. Depends on T051-T053 being complete.

## Acceptance Criteria

- [ ] `_get_optimizer_grouped_parameters(model, weight_decay)` function created
- [ ] Function returns list of 2 parameter groups: `[{params: decay_params, weight_decay: 0.01}, {params: no_decay_params, weight_decay: 0.0}]`
- [ ] No-decay group includes: `.bias` parameters, `LayerNorm.weight`, `LayerNorm.bias`
- [ ] Decay group includes: all other `.weight` parameters
- [ ] `test_fine_tuning()` uses grouped parameters: `AdamW(_get_optimizer_grouped_parameters(model, 0.01), lr=5e-5)`
- [ ] Validation: Count parameters in each group, verify biases excluded
- [ ] Validation: Train with/without exclusion, verify exclusion version ≤ baseline PPL
- [ ] Unit test: Mock model with bias/LayerNorm, verify correct grouping

## Test Scenarios

**Test Case 1: Parameter Grouping**
- Given: GPT-2 124M model with 163 total parameters (124M values)
- When: Call `_get_optimizer_grouped_parameters(model, 0.01)`
- Then: Group 1 (decay) has ~148M params (weights), Group 2 (no decay) has ~15M params (biases + LayerNorm)

**Test Case 2: Bias Exclusion**
- Given: Model with parameter named `transformer.h.0.attn.c_attn.bias`
- When: Group parameters
- Then: Bias in no-decay group (weight_decay=0.0)

**Test Case 3: LayerNorm Exclusion**
- Given: Parameter `transformer.ln_f.weight` (final LayerNorm)
- When: Group parameters
- Then: LayerNorm weight in no-decay group

**Test Case 4: Weight Inclusion**
- Given: Parameter `transformer.h.0.attn.c_attn.weight`
- When: Group parameters
- Then: Weight in decay group (weight_decay=0.01)

**Test Case 5: Performance Improvement**
- Given: Train two models - A: uniform decay, B: grouped decay
- When: Compare final validation PPL after 10 epochs
- Then: B ≤ A PPL (typically 0.5-2% better)

**Test Case 6: No Parameters Lost**
- Given: Model with N total parameters
- When: Group into decay/no-decay
- Then: sum(len(group['params']) for group in groups) == N (all params accounted for)

## Technical Implementation

**Required Components:**

1. **Create `_get_optimizer_grouped_parameters()` in `utils/tier3_training_utilities.py`:**
```python
def _get_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float = 0.01
) -> list:
    """
    Create parameter groups for AdamW optimizer with selective weight decay.

    Excludes bias parameters and LayerNorm weights from weight decay, following
    BERT/GPT-2/GPT-3 best practices.

    Args:
        model: PyTorch model
        weight_decay: Weight decay value for applicable parameters

    Returns:
        List of 2 parameter groups for optimizer:
        - Group 1: Weights (with weight_decay)
        - Group 2: Biases + LayerNorm (no weight_decay)

    Example:
        >>> param_groups = _get_optimizer_grouped_parameters(model, weight_decay=0.01)
        >>> optimizer = torch.optim.AdamW(param_groups, lr=5e-5)
    """
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters
```

2. **Update `test_fine_tuning()` to use grouped parameters:**
```python
def test_fine_tuning(
    model: nn.Module,
    config: SimpleNamespace,
    n_epochs: int = 10,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,  # NEW parameter
    batch_size: int = 4,
    use_wandb: bool = False,
    random_seed: int = 42,
    deterministic: bool = False,
    use_lr_schedule: bool = True
) -> dict:
    """Fine-tune model with weight decay exclusion for biases/LayerNorm."""

    # ... existing setup ...

    # Create optimizer with grouped parameters
    param_groups = _get_optimizer_grouped_parameters(model, weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

    print(f"📊 Optimizer: {len(param_groups[0]['params'])} decay params, "
          f"{len(param_groups[1]['params'])} no-decay params")

    # ... rest of training loop ...
```

3. **Add unit test in `tests/test_optimizer.py`:**
```python
import torch.nn as nn
from utils.tier3_training_utilities import _get_optimizer_grouped_parameters

def test_parameter_grouping():
    """Verify biases and LayerNorm excluded from weight decay"""
    # Simple model with bias and LayerNorm
    model = nn.Sequential(
        nn.Linear(10, 10, bias=True),  # Has .weight and .bias
        nn.LayerNorm(10),  # Has .weight and .bias
        nn.Linear(10, 5, bias=True)
    )

    param_groups = _get_optimizer_grouped_parameters(model, weight_decay=0.01)

    # Check group structure
    assert len(param_groups) == 2
    assert param_groups[0]['weight_decay'] == 0.01  # Decay group
    assert param_groups[1]['weight_decay'] == 0.0   # No-decay group

    # Count parameters
    decay_params = len(param_groups[0]['params'])
    no_decay_params = len(param_groups[1]['params'])

    # Model has: 2 linear weights (decay), 2 linear biases + 2 LayerNorm params (no decay)
    assert decay_params == 2, f"Expected 2 weight params, got {decay_params}"
    assert no_decay_params == 4, f"Expected 4 bias/LN params, got {no_decay_params}"

def test_all_parameters_grouped():
    """Verify no parameters are lost during grouping"""
    model = nn.Linear(100, 50, bias=True)
    param_groups = _get_optimizer_grouped_parameters(model, 0.01)

    total_grouped = sum(len(group['params']) for group in param_groups)
    total_model = sum(1 for _ in model.parameters())

    assert total_grouped == total_model, "Some parameters not grouped"
```

**Validation Commands:**

```bash
# Unit tests
pytest tests/test_optimizer.py -v

# Integration test (manual)
# 1. Train with weight_decay=0.01 (grouped parameters)
# 2. Compare to baseline (uniform decay)
# 3. Verify grouped version achieves equal or better PPL
```

**Code Patterns:**
- Use `model.named_parameters()` to inspect parameter names
- Filter by substring matching (`"bias"` in name, `"LayerNorm"` in name)
- Check `p.requires_grad` to exclude frozen parameters
- Return list of dicts (standard PyTorch optimizer API)

## Dependencies

**Hard Dependencies:**
- [T051] log_scalar() - Independent but both improve training
- [T052] Padding fix - Independent but both improve training
- [T053] Reproducibility - Independent but both improve training

**Soft Dependencies:**
- [T054] Extract duplicated code - Makes optimizer setup cleaner

**External Dependencies:**
- PyTorch AdamW optimizer (built-in)

## Design Decisions

**Decision 1: String Matching vs. Type Checking**
- **Rationale:** String matching (`"bias"` in name) simpler and more robust across architectures than `isinstance(param, nn.Parameter)` type checking.
- **Alternatives:** Type checking - fragile, misses custom modules
- **Trade-offs:** Pro: Works across custom architectures. Con: May miss non-standard naming (acceptable, document convention)

**Decision 2: Default weight_decay=0.01**
- **Rationale:** Standard value from BERT/GPT papers. Works well across model sizes.
- **Alternatives:** 0.0 (no decay) - underregularizes, 0.1 (high decay) - overregularizes
- **Trade-offs:** Pro: Proven default. Con: May not be optimal for all datasets (users can override)

**Decision 3: Two Groups (Decay/No-Decay) vs. Per-Layer Groups**
- **Rationale:** Two groups sufficient for 99% of cases. Per-layer groups (different decay per layer) adds complexity with marginal benefit.
- **Alternatives:** Per-layer decay schedules - complex, limited research support
- **Trade-offs:** Pro: Simple, proven. Con: Less flexible (acceptable for MVP)

**Decision 4: Expose weight_decay Parameter in test_fine_tuning()**
- **Rationale:** Users should control weight decay magnitude (e.g., 0.01 vs. 0.1). Transparency preferred.
- **Alternatives:** Hardcode to 0.01 - less flexible
- **Trade-offs:** Pro: User control. Con: More parameters to configure (acceptable, has good default)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Custom architectures with non-standard naming (e.g., `bn.gamma` instead of `LayerNorm.weight`) | Medium | Low | Document naming convention: "Assumes standard PyTorch naming. For custom modules, ensure bias/norm params include 'bias' or 'LayerNorm' in name." Add extensibility hook in future. |
| Weight decay=0.01 too high for small datasets (overfitting) | Low | Low | Expose `weight_decay` parameter (default 0.01). Users can reduce to 0.001 if overfitting. Document in CLAUDE.md. |
| Parameters with requires_grad=False incorrectly filtered | Low | Low | Already handled: `p.requires_grad` check ensures only trainable params grouped. Frozen params skipped. |
| Performance improvement not measurable on synthetic data | Low | Medium | Synthetic data too simple to show benefit. Document: "Test on real datasets (WikiText-2) to verify 0.5-2% PPL improvement." |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 2, Task 2 of 6. Training best practice from BERT/GPT papers. Improves final model quality by 0.5-2% PPL. Depends on T051-T053.
**Dependencies:** [T051, T052, T053] (training improvements stack together)
**Estimated Complexity:** Standard (2-hour implementation with parameter grouping, testing, validation)

## Completion Checklist

**Code Quality:**
- [ ] `_get_optimizer_grouped_parameters()` function with docstring
- [ ] Type hints on parameters
- [ ] Clean integration into `test_fine_tuning()`

**Testing:**
- [ ] Unit tests pass: `test_parameter_grouping()`, `test_all_parameters_grouped()`
- [ ] Integration test: Train with grouped params, verify PPL ≤ baseline
- [ ] Verified: All model parameters accounted for in groups

**Documentation:**
- [ ] Docstring explains bias/LayerNorm exclusion rationale
- [ ] CLAUDE.md notes weight decay best practice
- [ ] Comments explain no_decay list

**Integration:**
- [ ] Works with T055's LR scheduler (both optimizations compatible)
- [ ] Default weight_decay=0.01 exposed as parameter
- [ ] Logs parameter group sizes for visibility

**Definition of Done:**
Task is complete when weight decay selectively applied, biases/LayerNorm excluded, unit tests pass, and validation shows equal or better PPL.
