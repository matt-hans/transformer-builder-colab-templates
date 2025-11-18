---
id: T048
title: Create requirements.txt for Reproducibility
status: pending
priority: 1
agent: infrastructure
dependencies: []
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [reproducibility, infrastructure, phase1, refactor, critical]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - CLAUDE.md

est_tokens: 4500
actual_tokens: null
---

## Description

Create a proper `requirements.txt` file that pins exact versions of all training dependencies, enabling reproducible environments across different Colab sessions and local development. This is **critical prerequisite** for reproducibility—must come before all other training work.

Current state: No requirements file exists. Dependencies are installed ad-hoc in training.ipynb cells, leading to version drift and irreproducible results when notebooks are rerun months later.

Target state: Single source of truth for all Python dependencies with exact version pins (`==` not `>=`), organized by category (core, training, MLOps, optional), with inline comments explaining each package's purpose.

**Integration Points:**
- Referenced in training.ipynb installation cell
- Used by T016's environment snapshot feature
- Basis for deterministic pip installs in fresh Colab sessions
- Aligns with zero-installation strategy (template.ipynb remains dependency-free)

## Business Context

**User Story:** As an ML practitioner, I want to reproduce training results from 6 months ago, so that I can verify my published metrics and build on previous experiments.

**Why This Matters:**
Without pinned dependencies, code that worked in January may fail in June due to breaking changes in PyTorch, transformers, or W&B. This creates a reproducibility crisis where users cannot trust their own results.

**What It Unblocks:**
- T016's `pip freeze` can diff against this canonical list
- T017's config versioning can reference requirements.txt version
- Professional users gain confidence in training pipeline stability

**Priority Justification:**
P1 (Critical) - This is the foundation of reproducibility. Creating requirements.txt should be the **first** task before any training work, as it prevents future version conflicts and ensures all subsequent tasks build on a stable base.

## Acceptance Criteria

- [ ] `requirements.txt` created at repository root with exact version pins (`==`)
- [ ] All core dependencies pinned: `torch==2.6.0`, `numpy==2.3.4`, `pandas==2.2.3`
- [ ] Training dependencies pinned: `pytorch-lightning==2.4.0`, `optuna==3.0.0`, `torchmetrics==1.3.0`
- [ ] MLOps dependencies pinned: `wandb==0.18.7`, `huggingface-hub==0.27.0`, `datasets==3.2.0`
- [ ] Optional dependencies in separate section with comments (e.g., `# Optional: jupyter lab`)
- [ ] File includes header comment explaining version philosophy (exact pins for reproducibility)
- [ ] Verified compatible with Colab's pre-installed packages (torch, numpy don't conflict)
- [ ] training.ipynb updated to install from requirements.txt: `!pip install -r requirements.txt`
- [ ] Validation: Fresh Colab runtime can install all deps without conflicts or downgrades
- [ ] Validation: Local venv can install all deps: `python -m venv test_venv && source test_venv/bin/activate && pip install -r requirements.txt`

## Test Scenarios

**Test Case 1: Fresh Colab Installation**
- Given: New Colab runtime with pre-installed torch 2.6.0, numpy 2.3.4
- When: Run `!pip install -r requirements.txt`
- Then: All packages install without version conflicts, no downgrades of pre-installed packages, installation completes in <60 seconds

**Test Case 2: Local Development Setup**
- Given: Clean Python 3.10 virtual environment
- When: Run `pip install -r requirements.txt`
- Then: All dependencies install successfully, can import torch/wandb/optuna without errors

**Test Case 3: Version Compatibility Check**
- Given: requirements.txt specifies torch==2.6.0, numpy==2.3.4
- When: Check Colab's default versions (as of Nov 2025)
- Then: No version conflicts, torch/numpy pre-installed versions match pinned versions

**Test Case 4: Downstream Compatibility (T016 Integration)**
- Given: requirements.txt with 15 pinned dependencies
- When: T016's `pip freeze` runs and diffs against requirements.txt
- Then: Only expected additional packages listed (auto-installed sub-dependencies), no unexpected version changes

**Test Case 5: Six-Month Reproduction**
- Given: requirements.txt from today, archived for 6 months
- When: User creates fresh Colab session in May 2026 and installs from archived requirements.txt
- Then: Exact same package versions install (via PyPI archive), training code runs identically

**Test Case 6: Conflict Detection**
- Given: requirements.txt specifying incompatible versions (e.g., wandb==0.18.7 with python 3.7)
- When: Attempt `pip install -r requirements.txt`
- Then: Pip raises clear error, prevents broken environment (validation catches this during testing)

**Test Case 7: Optional Dependencies**
- Given: requirements.txt with optional section for `jupyter==1.0.0 # Optional: local notebook dev`
- When: User installs only required deps: `pip install -r requirements.txt --no-deps` (testing excluding optionals)
- Then: Core + training + MLOps deps install, jupyter skipped, training utilities still work

## Technical Implementation

**Required Components:**

1. **Create `requirements.txt` at repository root:**
```txt
# Transformer Builder - Training Pipeline Dependencies
# Version Philosophy: Exact pins (==) for full reproducibility
# Last verified: 2025-11-16 on Google Colab (Python 3.10, CUDA 12.1)

# ============================================================================
# CORE DEPENDENCIES (Pre-installed in Colab, pinned for local dev)
# ============================================================================
torch==2.6.0
numpy==2.3.4
pandas==2.2.3
matplotlib==3.8.2
seaborn==0.13.0
scipy==1.11.4

# ============================================================================
# TRAINING FRAMEWORK
# ============================================================================
pytorch-lightning==2.4.0      # Training loop abstraction, callbacks
optuna==3.0.0                  # Hyperparameter optimization
torchmetrics==1.3.0            # Metrics computation (accuracy, F1, etc.)
tqdm==4.66.1                   # Progress bars

# ============================================================================
# MLOPS & EXPERIMENT TRACKING
# ============================================================================
wandb==0.18.7                  # Experiment tracking, metrics logging
huggingface-hub==0.27.0        # Model registry, push/pull trained models
datasets==3.2.0                # Real dataset loading (WikiText, GLUE)

# ============================================================================
# OPTIONAL (Local Development)
# ============================================================================
# jupyter==1.0.0               # Notebook interface (not needed in Colab)
# ipywidgets==8.1.1            # Interactive widgets (Colab has this)
```

2. **Update training.ipynb installation cell (Cell 2):**
```python
# Cell 2: Install Training Dependencies
# Note: Uses requirements.txt for reproducible environment

# Download requirements.txt from repository
!wget https://raw.githubusercontent.com/YOUR_USERNAME/transformer-builder-colab-templates/main/requirements.txt -O requirements.txt

# Install all dependencies with exact versions
!pip install -r requirements.txt

# Verify installation
import torch
import pytorch_lightning as pl
import wandb
print(f"✅ Environment ready - PyTorch {torch.__version__}, Lightning {pl.__version__}")
```

3. **Update CLAUDE.md "Common Development Commands" section:**
```markdown
### Local Development Setup
```bash
# Create virtual environment and install from requirements.txt
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt  # Single command for all deps
```
```

**Validation Commands:**

```bash
# Test 1: Colab compatibility (manual - run in Colab)
# 1. Open blank Colab notebook
# 2. Run: !pip install -r requirements.txt
# 3. Verify no version conflicts in output

# Test 2: Local installation
python -m venv test_venv
source test_venv/bin/activate  # Windows: test_venv\Scripts\activate
pip install -r requirements.txt
python -c "import torch, wandb, optuna, pytorch_lightning; print('✅ All imports successful')"
deactivate
rm -rf test_venv

# Test 3: Verify versions match Colab defaults (Nov 2025)
grep "torch==" requirements.txt  # Should be 2.6.0
grep "numpy==" requirements.txt  # Should be 2.3.4
```

**Code Patterns:**
- Use exact version pins (`==`) not ranges (`>=`) for full reproducibility
- Group dependencies by category with clear comments
- Match Colab's pre-installed versions to avoid conflicts
- Optional deps in commented section, not separate requirements-dev.txt (simplicity)

## Dependencies

**Hard Dependencies** (must be complete first):
- None - This is the foundational task

**Soft Dependencies** (nice to have):
- None

**External Dependencies:**
- PyPI package availability (all packages exist in Nov 2025)
- Colab's default Python 3.10 + CUDA 12.1 environment

**Blocks Future Tasks:**
- [T016] Reproducibility - Environment Snapshot will reference this
- [T017] Reproducibility - Config Versioning will link to requirements version
- All training tasks benefit from stable dependency base

## Design Decisions

**Decision 1: Exact Version Pins (`==`) vs. Ranges (`>=`)**
- **Rationale:** Full reproducibility requires exact versions. Using `>=` leads to version drift over time as new releases appear.
- **Alternatives:**
  - `>=` with upper bounds (`>=2.6.0,<3.0.0`) - still allows drift within range
  - No pins at all - unacceptable for reproducibility
- **Trade-offs:**
  - Pro: 100% reproducible environments
  - Con: Requires periodic updates to get bug fixes (acceptable trade-off)

**Decision 2: Single requirements.txt (No requirements-dev.txt Split)**
- **Rationale:** Simplicity for users. Optional deps clearly commented, can be manually excluded if needed.
- **Alternatives:**
  - requirements.txt + requirements-dev.txt - adds complexity
  - pyproject.toml with optional groups - overkill for notebook-based project
- **Trade-offs:**
  - Pro: Single file, easier to maintain
  - Con: Slightly larger install if users include optionals (minimal impact)

**Decision 3: Match Colab Pre-installed Versions**
- **Rationale:** Avoid downgrading torch/numpy which can cause binary compatibility issues in Colab.
- **Alternatives:**
  - Use latest versions - may conflict with Colab's pre-installed packages
  - Require manual torch install - adds friction
- **Trade-offs:**
  - Pro: Fast installs in Colab (skips torch/numpy re-download)
  - Con: Tied to Colab's update cadence (acceptable, they update regularly)

**Decision 4: Download via wget vs. GitHub Raw URL**
- **Rationale:** Ensures training.ipynb always uses latest requirements.txt from main branch.
- **Alternatives:**
  - Copy-paste requirements into notebook cell - causes drift over time
  - Upload file manually to Colab - adds user friction
- **Trade-offs:**
  - Pro: Always in sync with repository
  - Con: Requires internet access (Colab has this by default)

**Decision 5: Three Requirements Files (Hybrid Strategy)**
- **Rationale:** Zero-installation strategy (v3.4.0) for template.ipynb prevents NumPy corruption in Colab, but reproducibility still needed for training and local dev
- **Implemented Files:**
  1. `requirements.txt` - Local development with exact pins (`==`) for all dependencies
  2. `requirements-training.txt` - Training notebook with exact pins for training-specific packages
  3. `requirements-colab-v3.4.0.txt` - Documentation of zero-installation strategy with range pins
- **Alternatives:**
  - Single requirements.txt - conflicts with v3.4.0 zero-installation architecture
  - No requirements files - unacceptable for reproducibility
- **Trade-offs:**
  - Pro: Supports both zero-installation (template) and exact reproducibility (training/local)
  - Con: More files to maintain (3 files), but clear separation of concerns
- **Documentation:** VERSION STRATEGY NOTES section in requirements-colab-v3.4.0.txt documents version deviations and intentional omissions

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Colab updates pre-installed packages, breaking version pins | High | Medium | Document tested Colab environment (Python 3.10, CUDA 12.1, Nov 2025). Include update instructions in requirements.txt header. Test quarterly and update pins. |
| PyPI removes old package versions, breaking future installs | High | Low | Pin versions that are 6+ months old (stable releases). Document alternative pip install from archives. Consider vendoring critical packages. |
| User has conflicting packages in local environment | Medium | Medium | Validate instructions specify fresh venv. Include troubleshooting section in CLAUDE.md for version conflicts. |
| requirements.txt diverges from training.ipynb actual usage | Medium | High | Establish update policy: when adding new import to training code, immediately update requirements.txt. Add CI check (future task) to verify imports match pins. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** User approved comprehensive refactor plan - Phase 1, Task 1 of 18. Critical prerequisite for reproducibility framework.
**Dependencies:** None (foundational task)
**Estimated Complexity:** Simple (single file creation, but critical for downstream tasks)

## Completion Checklist

**Code Quality:**
- [ ] requirements.txt follows standard format (package==version per line)
- [ ] Comments explain each dependency's purpose
- [ ] Grouped logically (core, training, MLOps, optional)
- [ ] No trailing whitespace or blank lines between pins

**Testing:**
- [ ] Fresh Colab runtime installs all deps without conflicts
- [ ] Local venv installs successfully: `pip install -r requirements.txt`
- [ ] All imports work: `python -c "import torch, wandb, optuna"`
- [ ] No version downgrades in Colab (check pip output)

**Documentation:**
- [ ] CLAUDE.md updated with `pip install -r requirements.txt` in setup instructions
- [ ] training.ipynb Cell 2 updated to wget and install from requirements.txt
- [ ] Header comment in requirements.txt explains version philosophy

**Integration:**
- [ ] Compatible with Colab's pre-installed torch 2.6.0, numpy 2.3.4
- [ ] All pinned versions exist on PyPI (verified via `pip index versions <package>`)
- [ ] File committed to main branch (for wget raw URL access)

**Definition of Done:**
Task is complete when requirements.txt exists with all deps pinned, training.ipynb uses it, fresh Colab installs work without errors, and local venv setup succeeds.
