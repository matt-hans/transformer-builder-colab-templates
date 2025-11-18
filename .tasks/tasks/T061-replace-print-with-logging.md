---
id: T061
title: Replace 515 Print Statements with Python Logging
status: pending
priority: 2
agent: backend
dependencies: [T054]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [refactor, code-quality, logging, phase3, large-task]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - utils/tier1_critical_validation.py
  - utils/tier2_advanced_analysis.py
  - utils/tier3_training_utilities.py

est_tokens: 60000
actual_tokens: null
---

## Description

Replace all 515 `print()` statements across utils/ with Python's logging module (logger.info/debug/warning/error), enabling proper log levels, filtering, and file output. This is a **large refactor** (20 hours) but essential for production use.

Current state: 515 print statements scattered across tier1/tier2/tier3 utilities. No log levels (can't filter debug vs. critical), no file output, prints inside training loops break tqdm progress bars.

Target state:
- Configured logger in each module: `logger = logging.getLogger(__name__)`
- Prints replaced: `logger.info()`, `logger.debug()`, `logger.warning()`, `logger.error()`
- Log levels: DEBUG (verbose), INFO (default), WARNING (issues), ERROR (failures)
- Training loop prints use `tqdm.write()` to preserve progress bars
- Optional file output: `logging.FileHandler('training.log')`

## Business Context

**Why This Matters:** 515 print statements create noise, can't be filtered, and break progress bars. Professional ML code uses structured logging for debugging and production monitoring.

**Priority:** P2 - Large refactor improving code quality. Should run after T054 (code extraction) to minimize duplication fixes.

**Estimated Effort:** 20 hours (515 prints ÷ 25 per hour = ~20.6 hours)

## Acceptance Criteria

- [ ] Python logging configured in tier1/tier2/tier3 modules
- [ ] All 515 `print()` statements replaced with `logger.{level}()`
- [ ] Log levels assigned correctly: debug for verbose output, info for normal, warning for issues, error for failures
- [ ] Training loop prints use `tqdm.write()` to preserve progress bars
- [ ] Validation: Run tests with `logging.basicConfig(level=logging.DEBUG)`, verify verbose output
- [ ] Validation: Run with `level=logging.WARNING`, verify only warnings/errors shown
- [ ] Validation: Training progress bars not broken by logging
- [ ] File output example in CLAUDE.md: `logging.basicConfig(filename='training.log')`

## Test Scenarios

**Test Case 1:** Log level filtering
- Given: Set `logging.basicConfig(level=logging.WARNING)`
- When: Run test_fine_tuning()
- Then: Only warnings/errors shown, info/debug suppressed

**Test Case 2:** Progress bar preservation
- Given: Training with tqdm progress bar
- When: Logging occurs during training loop
- Then: Progress bar remains intact, logs appear above bar

**Test Case 3:** File output
- Given: Configure `FileHandler('training.log')`
- When: Run training for 3 epochs
- Then: Log file contains all messages, rotates properly

## Technical Implementation

```python
# In utils/tier3_training_utilities.py
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def test_fine_tuning(...):
    logger.info(f"Starting training: {n_epochs} epochs, LR={learning_rate}")

    for epoch in tqdm(range(n_epochs), desc="Training"):
        for batch_idx, batch in enumerate(train_loader):
            # ... training code ...

            if batch_idx % 10 == 0:
                tqdm.write(f"Batch {batch_idx}: loss={loss.item():.4f}")  # Preserves progress bar

        logger.info(f"Epoch {epoch + 1}/{n_epochs} complete - "
                    f"Train loss: {avg_train_loss:.4f}, Val loss: {val_loss:.4f}")

# Configure logging (in notebook cells or main script)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('training.log')  # File output
    ]
)
```

**Mapping Guidelines:**
- `print(f"Training {model}")` → `logger.info(f"Training {model}")`
- `print(f"WARNING: ...")` → `logger.warning(...)`
- `print(f"ERROR: ...")` → `logger.error(...)`
- `print(f"DEBUG: ...")` → `logger.debug(...)`
- Prints inside tqdm loops → `tqdm.write(...)`

## Dependencies

**Hard Dependencies:**
- [T054] Extract duplicated code - Must complete first to avoid duplicate logging fixes

**Blocks:**
- None (but improves debugging for all future tasks)

## Design Decisions

**Decision 1:** Standard logging module vs. structlog/loguru
- **Rationale:** Standard library sufficient for current needs. Avoid external dependencies.
- **Trade-offs:** Pro: Zero deps. Con: Less features than loguru (acceptable)

**Decision 2:** tqdm.write() in training loops
- **Rationale:** Preserves progress bars. Standard practice in ML code.
- **Trade-offs:** Pro: Clean UX. Con: Requires tqdm import (already used)

**Decision 3:** Log levels by severity
- **Rationale:** DEBUG = verbose diagnostics, INFO = normal flow, WARNING = potential issues, ERROR = failures
- **Trade-offs:** Pro: Industry standard. Con: Requires judgment on categorization (acceptable, guidelines provided)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 515 replacements introduce typos/bugs | Medium | Medium | Automated search-replace where possible. Test each module after conversion. Code review focuses on logging changes. |
| Incorrect log level assignments | Low | Medium | Provide guidelines doc: "INFO = user-facing, DEBUG = developer diagnostics, WARNING = degraded mode, ERROR = failure." Review samples from each tier. |
| Performance overhead from excessive logging | Low | Low | Use logger.debug() for verbose output (disabled in production). Profile before/after to verify <1% overhead. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 3, Task 1 of 1. Large refactor (20 hours) replacing 515 print statements with structured logging. Should run after T054 to minimize duplication.
**Dependencies:** [T054] Extract duplicated code first
**Estimated Complexity:** Major (20 hours - systematic replacement across 3 tier modules)

**Expert Note:** Use `tqdm.write()` inside training loops to preserve progress bars. This is standard practice in ML libraries.

## Completion Checklist

- [ ] Logging configured in tier1/tier2/tier3 modules
- [ ] All 515 prints replaced (verified via `grep -r "print(" utils/`)
- [ ] Log levels assigned correctly (info/debug/warning/error)
- [ ] tqdm.write() used in training loops
- [ ] Tests pass with DEBUG and WARNING log levels
- [ ] CLAUDE.md updated with logging examples
- [ ] File output example tested

**Definition of Done:** All 515 prints replaced, log levels work, progress bars preserved, tests pass.
