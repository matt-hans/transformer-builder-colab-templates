# Testing Guide: v3.3.1 - Pre-Corrupted NumPy Fix

**Version:** v3.3.1
**Date:** 2025-01-13
**Purpose:** Verify the 4-layer defense system handles all numpy corruption scenarios

---

## What Changed in v3.3.1

### Problem Solved
User tested v3.3.0 and got immediate failure:
```
❌ NumPy is already corrupted! Recommend: Runtime → Restart runtime
ImportError: NumPy corruption detected before installation
```

This error occurred in the **pre-flight check**, meaning numpy was corrupted BEFORE Cell 3 ran.

### Root Cause
**90% probability:** User didn't restart runtime after previous v3.2.0 test (corrupted runtime persisted)
**10% probability:** Colab startup corruption (unlikely but handled)

### Solution: 4-Layer Defense System

```
Layer 1: Cell 1 (Markdown)
  ↓ Prominent warning about runtime restarts

Layer 2: Cell 2 (Version Check)
  ↓ Runtime freshness detection (marker file)
  ↓ Requires user confirmation to continue with reused runtime

Layer 3: Cell 3 (Installation - Pre-flight)
  ↓ Detect corruption BEFORE installation
  ↓ Attempt automatic repair (2 strategies)
  ↓ If repair fails: clear error message + instructions

Layer 4: Cell 3 (Installation - Post-flight)
  ↓ Verify numpy still intact AFTER installation
  ↓ If corrupted during install: critical bug report
```

---

## Test Scenarios

### Scenario 1: Fresh Runtime (Expected: ✅ PASS)

**Setup:**
```
1. Runtime → Restart runtime
2. Edit → Clear all outputs
```

**Steps:**
1. Run Cell 1 (markdown) - should display warning
2. Run Cell 2 (version check)
   - Expected: "No marker file found, creating..."
   - Creates `/tmp/transformer_builder_runtime_used`
3. Run Cell 3 (installation)
   - Expected: Pre-flight ✅ pass
   - Installation proceeds normally
   - Post-flight ✅ pass
   - All imports succeed

**Success Criteria:**
- ✅ No errors
- ✅ Installation completes in 5-10 seconds
- ✅ All dependencies verified
- ✅ Marker file created at `/tmp/transformer_builder_runtime_used`

---

### Scenario 2: Reused Runtime - User Continues (Expected: ⚠️ PASS with warning)

**Setup:**
```
Do NOT restart runtime (reuse from Scenario 1)
```

**Steps:**
1. Run Cell 2 again
   - Expected: Detects marker file
   - Shows prominent warning
   - Prompts: "Do you want to continue anyway? (type 'yes' to proceed):"
2. User types: `yes`
3. Run Cell 3
   - Expected: Pre-flight ✅ pass (assuming numpy still intact)
   - Installation proceeds
   - Post-flight ✅ pass

**Success Criteria:**
- ⚠️ Warning displayed correctly
- ✅ User can override and continue
- ✅ Installation succeeds (if numpy still intact)

---

### Scenario 3: Reused Runtime - User Declines (Expected: ❌ STOPS)

**Setup:**
```
Do NOT restart runtime
```

**Steps:**
1. Run Cell 2 again
   - Expected: Detects marker file
   - Shows prominent warning
   - Prompts: "Do you want to continue anyway? (type 'yes' to proceed):"
2. User types: `no` (or anything other than 'yes')
3. Expected: Execution stops with RuntimeError

**Success Criteria:**
- ✅ Clear error message: "Runtime restart required. Please: Runtime → Restart runtime"
- ✅ Cell execution halted
- ✅ User guided to restart

---

### Scenario 4: Pre-Corrupted Runtime - Auto-Repair Succeeds (Expected: ✅ PASS after repair)

**Setup:**
```
1. Runtime → Restart runtime
2. Manually corrupt numpy (simulate v3.2.0):
   !pip install -q onnx onnxruntime
```

**Steps:**
1. Run Cell 2 (version check)
   - Expected: Marker created (fresh runtime)
2. Run Cell 3 (installation)
   - Expected: Pre-flight ❌ detects corruption
   - Shows "CORRUPTION DETECTED BEFORE INSTALLATION"
   - Attempts automatic repair
   - Expected: "✅ Strategy 1 successful!" (or Strategy 2)
   - Continues with installation
   - Post-flight ✅ pass

**Success Criteria:**
- ✅ Corruption detected in pre-flight
- ✅ Auto-repair succeeds
- ✅ Installation completes successfully
- ✅ User sees clear messaging about repair

---

### Scenario 5: Pre-Corrupted Runtime - Auto-Repair Fails (Expected: ❌ FAILS with clear instructions)

**Setup:**
```
1. Runtime → Restart runtime
2. Corrupt numpy in a way that's unrecoverable:
   !pip uninstall -y numpy
   !pip install numpy==1.24.0  # Incompatible version
```

**Steps:**
1. Run Cell 2 (version check)
2. Run Cell 3 (installation)
   - Expected: Pre-flight ❌ detects corruption
   - Attempts automatic repair
   - Expected: "❌ Both repair strategies failed"
   - Shows clear recovery instructions
   - Raises ImportError

**Success Criteria:**
- ✅ Corruption detected
- ✅ Auto-repair attempts made
- ✅ Clear error message with recovery steps:
   ```
   REQUIRED ACTION:
     1. Runtime → Restart runtime
     2. Edit → Clear all outputs
     3. Runtime → Run all
   ```
- ✅ ImportError raised to halt execution

---

### Scenario 6: Corruption During Installation (Expected: ❌ CRITICAL BUG)

**Setup:**
```
This scenario tests if requirements-colab.txt still has problematic packages
```

**Steps:**
1. Runtime → Restart runtime
2. Run Cell 2 (version check)
3. Run Cell 3 (installation)
   - Expected: Pre-flight ✅ pass
   - Installation runs...
   - Post-flight ❌ detects corruption (hypothetically)
   - Shows "CRITICAL BUG" message
   - Provides debug info (Python version, numpy version)
   - Asks to report bug

**Success Criteria:**
- ✅ Clear messaging: "This is a CRITICAL BUG in v3.3.1 - this should NOT happen"
- ✅ Debug information provided
- ✅ Bug report URL shown
- ✅ ImportError raised

**Note:** This should NOT happen with v3.3.1's minimal requirements. If it does, it's a real bug.

---

## Testing Checklist

### Pre-Test Setup
- [ ] Ensure you have access to Google Colab
- [ ] Have v3.3.1 notebook ready
- [ ] Clear your browser cache (optional, but recommended)

### Test Execution

**Fresh Runtime Test:**
- [ ] Scenario 1: Fresh runtime (expected: ✅ pass)

**Runtime Reuse Tests:**
- [ ] Scenario 2: Reused runtime, user continues (expected: ⚠️ pass with warning)
- [ ] Scenario 3: Reused runtime, user declines (expected: ❌ stops)

**Corruption Tests:**
- [ ] Scenario 4: Pre-corrupted, auto-repair succeeds (expected: ✅ pass after repair)
- [ ] Scenario 5: Pre-corrupted, auto-repair fails (expected: ❌ fails with clear instructions)
- [ ] Scenario 6: Corruption during install (expected: ❌ critical bug - should NOT happen)

### Post-Test Validation
- [ ] All expected scenarios behaved correctly
- [ ] Error messages were clear and actionable
- [ ] No confusing or misleading output
- [ ] User experience was smooth (for successful scenarios)

---

## Expected Test Results

### Success Metrics

**Primary Goals:**
- ✅ 90% of users never see an error (Scenarios 1, 4 with auto-repair)
- ✅ 10% who hit errors get clear, actionable instructions (Scenario 5)
- ✅ 0% of users hit confusing error messages

**Technical Goals:**
- ✅ Detect pre-corrupted numpy 100% of the time (Scenarios 4, 5)
- ✅ Auto-repair succeeds in 70%+ of corruption cases (Scenario 4)
- ✅ Provide debug info for bug reports in remaining cases (Scenario 5, 6)

---

## How to Run Tests in Colab

### Method 1: Manual Testing

1. **Open v3.3.1 notebook in Colab**
   ```
   File → Upload notebook → Select template.ipynb
   ```

2. **For each scenario:**
   - Follow the "Setup" steps
   - Execute cells as described in "Steps"
   - Verify "Success Criteria"
   - Document results

### Method 2: Automated Testing (Using Playwright MCP)

```python
# In Claude Code with Playwright MCP
# Load Transformer Builder
# Click "Open in Colab"
# Execute cells programmatically
# Verify output matches expected results
```

---

## Debugging Failed Tests

### If Scenario 1 fails (Fresh runtime should pass):
**Possible causes:**
1. requirements-colab.txt still has problematic packages → Check file contents
2. Colab updated pre-installed packages → Check numpy version in Colab
3. pip/pip cache issues → Try clearing pip cache

**Debug steps:**
```python
# In a fresh Colab cell:
import numpy as np
print(f"NumPy version: {np.__version__}")

from numpy._core.umath import _center
print("✅ numpy C extensions intact")
```

### If Scenario 4 fails (Auto-repair should succeed):
**Possible causes:**
1. Corruption is too severe for force-reinstall
2. Pip cache has corrupted packages
3. Python import cache isn't clearing properly

**Debug steps:**
```python
# Check if force reinstall works manually:
!pip install --force-reinstall --no-deps numpy==2.3.4

# Clear Python import cache:
import sys
for module in list(sys.modules.keys()):
    if 'numpy' in module:
        del sys.modules[module]

# Test:
from numpy._core.umath import _center
print("✅ Manual repair worked")
```

### If Scenario 6 occurs (Should NOT happen):
**This is a CRITICAL BUG** - one of the "safe" packages is corrupting numpy.

**Immediate action:**
1. Document exact package versions installed
2. Run diagnostic script to identify culprit:
   ```python
   !wget https://raw.githubusercontent.com/matt-hans/transformer-builder-colab-templates/main/test-numpy-corruption.py
   !python test-numpy-corruption.py
   ```
3. Remove culprit from requirements-colab.txt
4. Release v3.3.2 hotfix

---

## Post-Test Actions

### If All Tests Pass:
1. ✅ Mark v3.3.1 as stable
2. ✅ Deploy to production (update main branch)
3. ✅ Update CHANGELOG.md
4. ✅ Monitor user feedback for 48 hours

### If Some Tests Fail:
1. ❌ Document failure details in bug report
2. ❌ Identify root cause (see "Debugging Failed Tests")
3. ❌ Fix issues → Release v3.3.2
4. ❌ Re-run full test suite

---

## Test Report Template

```markdown
# v3.3.1 Test Report

**Tester:** [Your Name]
**Date:** [YYYY-MM-DD]
**Environment:** Google Colab (Python [version], numpy [version])

## Test Results

### Scenario 1: Fresh Runtime
- Status: [ ] PASS / [ ] FAIL
- Notes:

### Scenario 2: Reused Runtime (User Continues)
- Status: [ ] PASS / [ ] FAIL
- Notes:

### Scenario 3: Reused Runtime (User Declines)
- Status: [ ] PASS / [ ] FAIL
- Notes:

### Scenario 4: Pre-Corrupted (Auto-Repair Succeeds)
- Status: [ ] PASS / [ ] FAIL
- Repair strategy used: [ ] Strategy 1 / [ ] Strategy 2 / [ ] Failed
- Notes:

### Scenario 5: Pre-Corrupted (Auto-Repair Fails)
- Status: [ ] PASS / [ ] FAIL
- Notes:

### Scenario 6: Corruption During Install
- Status: [ ] DID NOT OCCUR (expected) / [ ] OCCURRED (CRITICAL BUG)
- Notes:

## Overall Assessment

- [ ] Ready for production
- [ ] Needs fixes (see notes)

## Recommendations

[Your recommendations here]
```

---

## Contact & Support

**Bug Reports:** https://github.com/matt-hans/transformer-builder-colab-templates/issues
**Documentation:** See COMPREHENSIVE_FIX_NUMPY_PRECORRUPTION.md
**Version History:** See CHANGELOG.md
