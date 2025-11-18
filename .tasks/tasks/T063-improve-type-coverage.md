---
id: T063
title: Improve Type Hint Coverage to 90%
status: pending
priority: 3
agent: backend
dependencies: [T061, T062]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [code-quality, typing, phase4, refactor]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - utils/tier1_critical_validation.py
  - utils/tier2_advanced_analysis.py
  - utils/tier3_training_utilities.py

est_tokens: 27000
actual_tokens: null
---

## Description

Add comprehensive type hints to all public functions in utils/, achieving 90% coverage. Enables static type checking with mypy, improves IDE autocomplete, and catches bugs before runtime.

Current state: ~40% of functions have type hints. Missing hints on return types, complex types (Union, Optional), and internal utilities.

Target state: 90% coverage verified by mypy, all public functions fully typed, complex types properly annotated (e.g., `Optional[torch.optim.Optimizer]`, `Union[dict, pd.DataFrame]`).

## Business Context

**Why This Matters:** Type hints catch bugs during development (before CI/testing). Missing hints lead to runtime errors like `AttributeError: 'NoneType' has no attribute 'step'` (forgot to check if scheduler is None).

**Priority:** P3 - Code quality improvement. Not critical but improves developer experience.

## Acceptance Criteria

- [ ] All public functions in tier1/tier2/tier3 have full type hints (args + return)
- [ ] Complex types properly annotated: `Optional[T]`, `Union[A, B]`, `List[dict]`, etc.
- [ ] mypy runs without errors: `mypy utils/ --strict`
- [ ] Type stubs for PyTorch types: `torch.Tensor`, `torch.nn.Module`, `torch.optim.Optimizer`
- [ ] Validation: mypy coverage report shows ≥90% (run `mypy --html-report coverage`)
- [ ] Validation: IDE autocomplete works for all typed functions
- [ ] CI check: mypy added to pre-commit or GitHub Actions (future task)

## Test Scenarios

**Test Case 1:** mypy validation
- Given: Type hints added to all public functions
- When: Run `mypy utils/ --strict`
- Then: 0 errors, warnings only for unavoidable issues (3rd-party library stubs)

**Test Case 2:** Catch type errors
- Given: Function expects `torch.nn.Module`, receives `int`
- When: mypy checks code
- Then: Raises error: "Argument 1 has incompatible type 'int'; expected 'Module'"

**Test Case 3:** IDE autocomplete
- Given: Call `test_fine_tuning(model, config, `
- When: Trigger autocomplete in VSCode/PyCharm
- Then: Shows all parameters with types: `n_epochs: int = 10`, `learning_rate: float = 5e-5`

## Technical Implementation

```python
# Before
def test_fine_tuning(model, config, n_epochs=10, learning_rate=5e-5):
    ...

# After
from typing import Optional, Union
import torch
import torch.nn as nn
from types import SimpleNamespace
import pandas as pd

def test_fine_tuning(
    model: nn.Module,
    config: SimpleNamespace,
    n_epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    use_wandb: bool = False,
    random_seed: int = 42,
    deterministic: bool = False,
    use_lr_schedule: bool = True,
    max_grad_norm: Optional[float] = 1.0
) -> dict:
    """Fine-tune model with full type safety."""
    ...
```

**Coverage Target:**
- tier1_critical_validation.py: 15 functions, all typed
- tier2_advanced_analysis.py: 8 functions, all typed
- tier3_training_utilities.py: 25 functions, all typed
- Total: ~48 functions → ≥43 typed (90%)

## Dependencies

**Hard Dependencies:**
- [T061] Logging refactor - Cleaner code easier to type
- [T062] Decomposition - Smaller functions easier to type

**Blocks:** None

## Design Decisions

**Decision 1:** Use Optional[T] vs. Union[T, None]
- **Rationale:** `Optional[T]` more concise, standard in typing community.
- **Trade-offs:** Pro: Readable. Con: Requires `from typing import Optional`

**Decision 2:** 90% coverage target (not 100%)
- **Rationale:** Some internal utilities hard to type (dynamic imports, monkey-patching). 90% covers all user-facing code.
- **Trade-offs:** Pro: Achievable. Con: Not perfect (acceptable)

**Decision 3:** Use mypy --strict mode**
- **Rationale:** Catches more issues (implicit Optional, Any types, untyped decorators).
- **Trade-offs:** Pro: Maximum safety. Con: Requires more work (acceptable for quality)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| PyTorch types not recognized by mypy (no stubs) | Medium | Low | Install `torch` with type stubs or add `# type: ignore` for unavoidable cases. |
| Complex types too verbose (e.g., nested dicts) | Low | Medium | Use `TypedDict` or `dataclass` for complex structures. Document in comments. |
| Type hints break backward compatibility | Low | Very Low | Type hints are annotations only, don't affect runtime. Python <3.5 users unaffected. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 4, Task 2 of 2. Code quality refactor adding comprehensive type hints. Runs after T061/T062 for cleaner code base.
**Dependencies:** [T061, T062] (cleaner code easier to type)
**Estimated Complexity:** Complex (9 hours - 48 functions, complex types, mypy configuration)

## Completion Checklist

- [ ] 90% of functions have full type hints
- [ ] mypy --strict runs without errors
- [ ] Complex types properly annotated (Optional, Union, List)
- [ ] IDE autocomplete works
- [ ] mypy coverage report shows ≥90%

**Definition of Done:** 90% type coverage verified by mypy, IDE autocomplete works, no mypy errors in strict mode.
