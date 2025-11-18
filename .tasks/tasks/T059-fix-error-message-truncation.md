---
id: T059
title: Fix Error Message Truncation in Notebook Cells
status: pending
priority: 2
agent: frontend
dependencies: [T051, T052, T053]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [ux, error-handling, phase2, refactor, quick-win]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - template.ipynb
  - training.ipynb

est_tokens: 6000
actual_tokens: null
---

## Description

Fix error message truncation in Colab notebooks by adding `sys.tracebacklimit` configuration and custom exception formatting. Currently, long tracebacks (>10 frames) get truncated, hiding root cause of errors.

Current state: Colab truncates stack traces at 10 frames with `... (X frames hidden)`. Users cannot diagnose import errors, model initialization failures, or deep call stack issues.

Target state: Custom exception handler in notebooks displays full traceback with syntax highlighting, shows last 50 lines of relevant context, formats nicely in Colab output cells.

## Business Context

**Why This Matters:** Truncated errors waste hours of debugging time. Users report issues like "model fails to load" without seeing the actual ImportError 15 frames deep.

**Priority:** P2 - UX improvement that saves debugging time.

## Acceptance Criteria

- [ ] Add `sys.tracebacklimit = 50` in notebook setup cells
- [ ] Custom `format_exception()` function formats errors with syntax highlighting
- [ ] Try/except blocks in critical cells (model loading, training) use custom formatting
- [ ] Validation: Trigger deep error (15+ frames), verify full traceback shown
- [ ] Validation: Error messages include file path, line number, relevant code context
- [ ] Notebook cells include "Troubleshooting" markdown with common error patterns

## Technical Implementation

```python
# Cell 1: Setup Error Handling
import sys
import traceback

sys.tracebacklimit = 50  # Show up to 50 stack frames

def format_exception(e: Exception, context_lines: int = 5) -> str:
    """Format exception with full traceback and code context."""
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    return "".join(tb_lines)

# Cell 3: Model Loading (with error handling)
try:
    model = load_model_from_gist(gist_id, model_name)
except Exception as e:
    print("❌ Model Loading Failed")
    print("=" * 60)
    print(format_exception(e))
    print("=" * 60)
    print("\n📚 Troubleshooting:")
    print("  - Check Gist ID is correct")
    print("  - Verify model.py syntax (run through Python parser)")
    print("  - Ensure all imports available in Colab")
    raise
```

## Dependencies

**Hard Dependencies:** None
**Blocks:** None

## Completion Checklist

- [ ] `sys.tracebacklimit = 50` added to notebooks
- [ ] Custom error formatting in critical cells
- [ ] Troubleshooting guides added as markdown
- [ ] Validated: Deep errors show full traceback

**Definition of Done:** Full stack traces visible, users can diagnose errors without external tools.
