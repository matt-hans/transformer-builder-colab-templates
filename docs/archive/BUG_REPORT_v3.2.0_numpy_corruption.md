# BUG REPORT: v3.2.0 Numpy Corruption Still Occurring

**Date:** 2025-01-13
**Version:** v3.2.0
**Status:** 🔴 CRITICAL - Notebook fails at Cell 3
**Test Environment:** Google Colab (Python 3.12, numpy 2.3.4)

---

## Executive Summary

Despite removing `onnx/onnxruntime` in v3.2.0, **numpy corruption still occurs** during dependency installation at Cell 3. The error manifests when importing `pytorch_lightning`, indicating that one or more packages in `requirements-colab.txt` are corrupting Colab's pre-installed numpy 2.3.4.

---

## Error Details

### Error Message
```python
ImportError: cannot import name '_center' from 'numpy._core.umath'
(/usr/local/lib/python3.12/dist-packages/numpy/_core/umath.py)
```

### Stack Trace
```
Cell 3 execution failed at line 38:
  import pytorch_lightning as pl

Full trace:
  /usr/local/lib/python3.12/dist-packages/numpy/_core/strings.py
  from numpy._core.umath import _center
  ImportError: cannot import name '_center' from 'numpy._core.umath'
```

### Execution Timeline
1. ✅ Step 1/3: pip upgrade completed (0s)
2. ✅ Step 2/3: Install safe dependencies from requirements-colab.txt (~15s)
3. ✅ Step 3/3: Install pytorch-lightning with --no-deps (~3s)
4. ❌ **VERIFICATION FAILED**: numpy C extensions corrupted

**Total execution time:** 20.523s
**Cell status:** Execution ended unsuccessfully

---

## Root Cause Analysis

### Hypothesis
One or more packages in `requirements-colab.txt` have transitive dependencies that conflict with numpy 2.x, despite being labeled as "safe":

```python
# Current requirements-colab.txt (v3.2.0)
datasets>=2.16.0,<3.0.0          # SUSPECT: Large package with many deps
tokenizers>=0.15.0,<1.0.0        # SUSPECT: May pull in incompatible deps
huggingface-hub>=0.20.0,<1.0.0   # Likely safe
torchinfo>=1.8.0,<3.0.0          # Likely safe
optuna>=3.0.0,<4.0.0             # SUSPECT: scipy/numpy dep conflicts
pytest>=7.4.0,<8.0.0             # Likely safe
pytest-cov>=4.1.0,<5.0.0         # Likely safe
```

### Primary Suspects

1. **datasets** (Highest priority)
   - Known issue: Has many dependencies including `pyarrow`, `dill`, `xxhash`
   - These may require specific numpy versions

2. **optuna** (Medium priority)
   - Depends on scipy, which has strict numpy version requirements
   - May conflict with Colab's numpy 2.3.4

3. **tokenizers** (Lower priority)
   - Rust-based with potential C extension conflicts

---

## Testing Strategy

### Immediate Action: Isolate the Culprit

Run the diagnostic script `test-numpy-corruption.py` in a fresh Colab environment to test each package individually:

```python
# In fresh Colab cell:
!wget https://raw.githubusercontent.com/matt-hans/transformer-builder-colab-templates/main/test-numpy-corruption.py
!python test-numpy-corruption.py
```

This will identify which package(s) corrupt numpy.

### Alternative: Manual Binary Search

If unable to run automated test, manually test in Colab:

```python
# Cell 1: Verify baseline
from numpy._core.umath import _center
print("✅ numpy intact")

# Cell 2: Test datasets
!pip install -q datasets
from numpy._core.umath import _center  # Will fail if datasets is culprit

# Cell 3: Factory reset runtime, test tokenizers
# Runtime → Factory reset runtime
!pip install -q tokenizers
from numpy._core.umath import _center  # Will fail if tokenizers is culprit

# Repeat for each package...
```

---

## Proposed Solutions

### Option 1: Remove Problematic Packages (v3.3.0 - Quick Fix)

**Strategy:** Eliminate packages that corrupt numpy, add fallback instructions

```python
# requirements-colab.txt v3.3.0 (MINIMAL)
# Only absolutely essential packages that are verified numpy-safe

# Core utilities (verified safe)
torchinfo>=1.8.0,<3.0.0
pytest>=7.4.0,<8.0.0
pytest-cov>=4.1.0,<5.0.0

# ==============================================================================
# INSTALL MANUALLY IF NEEDED (to avoid numpy corruption):
# - datasets (likely corrupts numpy - install only if using HF datasets)
# - tokenizers (may corrupt numpy)
# - optuna (may corrupt numpy - use for hyperparameter tuning only)
# - huggingface-hub (install only if uploading to HF Hub)
# ==============================================================================
```

**Pros:**
- Guaranteed to work (minimal dependencies = minimal corruption risk)
- Fast installation (<5s)

**Cons:**
- Users lose automatic HuggingFace dataset loading
- No built-in hyperparameter optimization (Optuna)

---

### Option 2: Pin Specific Versions (v3.3.0 - Targeted Fix)

**Strategy:** Pin exact versions that are known to work with numpy 2.3.4

```python
# requirements-colab.txt v3.3.0 (PINNED)
# Exact versions verified to work with Colab's numpy 2.3.4

datasets==2.16.1        # Pinned version compatible with numpy 2.x
tokenizers==0.15.2      # Pinned version
huggingface-hub==0.20.3 # Pinned version
torchinfo==1.8.0
optuna==3.5.0          # Pinned version compatible with numpy 2.x
pytest==7.4.3
pytest-cov==4.1.0
```

**Pros:**
- Keeps all functionality
- More reproducible builds

**Cons:**
- Requires testing to find working versions
- May break when Colab updates pre-installed packages

---

### Option 3: Use Conda Environment (v3.3.0 - Nuclear Option)

**Strategy:** Create isolated conda environment to avoid Colab's package conflicts

```python
# Cell 3 (NEW approach)
!pip install -q condacolab
import condacolab
condacolab.install()

# Then install all packages via conda to avoid pip dependency hell
!conda install -c conda-forge -y numpy pytorch-lightning datasets optuna
```

**Pros:**
- Complete isolation from Colab's packages
- Conda handles binary compatibility better than pip

**Cons:**
- Slower installation (~2-3 minutes)
- More complex for users
- Larger disk footprint

---

## Recommended Next Steps

1. **[URGENT]** Run `test-numpy-corruption.py` to identify exact culprit(s)
2. **[HIGH]** Implement v3.3.0 with Option 1 (minimal requirements) as immediate fix
3. **[MEDIUM]** Test Option 2 (pinned versions) in parallel for more feature-complete solution
4. **[LOW]** Document workaround for users who need removed packages

---

## Additional Context

### Colab Environment Details
- Python: 3.12
- numpy (pre-installed): 2.3.4
- torch (pre-installed): 2.6-2.8
- transformers (pre-installed): 4.37+

### Previous Fixes Attempted
- v3.0.0: Removed numpy from requirements → Still failed
- v3.1.0: Added --no-deps for pytorch-lightning → Still failed
- v3.2.0: Removed onnx/onnxruntime → **Still failing** (current)

### Lessons Learned
- Removing explicit numpy doesn't prevent corruption
- Using --no-deps on one package isn't enough
- Need to audit **all** dependencies, not just the obvious ones
- Colab's pre-installed packages have hidden constraints

---

## Success Criteria for v3.3.0

- [ ] Cell 3 completes without numpy corruption errors
- [ ] All numpy C extensions intact: `from numpy._core.umath import _center` succeeds
- [ ] pytorch-lightning imports successfully
- [ ] Tier 1 tests can run
- [ ] Installation time < 30 seconds

---

## Files to Update for v3.3.0

1. `requirements-colab.txt` - Remove/pin problematic packages
2. `template.ipynb` Cell 3 - Update installation instructions
3. `CHANGELOG.md` - Document the fix
4. `README.md` - Add troubleshooting section

---

**Reporter:** Claude Code (Automated Testing)
**Priority:** P0 - Blocks all users
**Assignee:** Development team
