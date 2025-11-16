# T001 - W&B Integration Manual Test Checklist

This checklist validates the W&B integration in `training.ipynb` for Google Colab.

**Task:** T001 - W&B Basic Integration - Add wandb.init() and Config Logging
**Priority:** P1-MVP
**Test Environment:** Google Colab (free tier)

---

## Pre-Test Setup

- [ ] Open Google Colab: https://colab.research.google.com
- [ ] Upload `training.ipynb` or open from GitHub
- [ ] **Runtime → Restart runtime** (fresh environment required)
- [ ] Have a W&B account ready (or create free account at https://wandb.ai/signup)
- [ ] Get W&B API key from https://wandb.ai/authorize

---

## Test Scenario 1: First-Time W&B Setup (with Colab Secrets)

**Objective:** Verify that a new user can set up W&B using Colab Secrets securely.

### Steps:

1. **Configure Colab Secrets:**
   - [ ] Click 🔑 (key icon) in left sidebar
   - [ ] Click "Add new secret"
   - [ ] Name: `WANDB_API_KEY`
   - [ ] Value: Paste your W&B API key from https://wandb.ai/authorize
   - [ ] Toggle "Notebook access" to ON

2. **Run Cell 4 (Dependencies):**
   - [ ] Execute Cell 4: `pip install pytorch-lightning optuna torchmetrics wandb`
   - [ ] Verify output shows: `✅ wandb: X.X.X` (version number)
   - [ ] No errors during installation

3. **Read Cell 5 (W&B Info Markdown):**
   - [ ] Verify markdown cell explains W&B benefits
   - [ ] Verify instructions for Colab Secrets setup
   - [ ] Verify security warning: "NEVER hardcode API keys"

4. **Run Cell 6 (W&B Login):**
   - [ ] Execute Cell 6
   - [ ] Expected output:
     ```
     ✅ W&B authenticated via Colab Secrets
     ✅ Logged in as: your-wandb-username
     🎯 Experiments will be tracked at: https://wandb.ai

     ======================================================================
     ✅ W&B READY - Experiments will be tracked online
     ======================================================================
     ```
   - [ ] No errors or warnings

5. **Verify Login Status:**
   - [ ] `wandb_enabled` variable is set to `True` (check in Python cell: `print(wandb_enabled)`)

**Expected Result:** W&B authenticates successfully using Colab Secrets, no API key visible in notebook.

---

## Test Scenario 2: Hyperparameter Logging

**Objective:** Verify that all hyperparameters and model metadata are logged to W&B config.

### Steps:

1. **Complete setup** (Cells 1-12: Gist ID, dependencies, W&B login, model loading, instantiation)
   - [ ] Cell 2: Enter a valid Gist ID
   - [ ] Cells 4, 6, 8, 10, 12: All execute without errors
   - [ ] Model instantiated successfully

2. **Run Cell 14 (Training Tests with W&B):**
   - [ ] Execute Cell 14
   - [ ] Expected output shows W&B initialization:
     ```
     ========================================================================
     📊 W&B TRACKING INITIALIZED
     ========================================================================

     🎯 Project: transformer-builder-training
     🏷️  Run name: gpt_20250115_143027 (or similar)
     🔗 Dashboard: https://wandb.ai/your-username/transformer-builder-training/runs/...

     📋 Logged config:
        • Model: gpt (124.44M params) [or your model type/size]
        • Learning rate: 5e-05
        • Batch size: 2
        • Epochs: 3
     ```

3. **Verify W&B Dashboard:**
   - [ ] Click the dashboard link from output
   - [ ] Navigate to "Overview" tab
   - [ ] Verify run name includes timestamp and architecture (e.g., `gpt_20250115_143027`)
   - [ ] Navigate to "Config" tab
   - [ ] Verify **Hyperparameters** section shows:
     - [ ] `learning_rate`: 5e-05
     - [ ] `batch_size`: 2
     - [ ] `epochs`: 3
     - [ ] `warmup_ratio`: 0.1
     - [ ] `weight_decay`: 0.01
     - [ ] `max_grad_norm`: 1.0
   - [ ] Verify **Model Metadata** section shows:
     - [ ] `model_type`: "gpt" (or "bert", "t5", "custom")
     - [ ] `vocab_size`: 50257 (or your model's vocab size)
     - [ ] `max_seq_len`: 128 (or your model's seq length)
     - [ ] `total_params`: (integer count)
     - [ ] `trainable_params`: (integer count)
     - [ ] `total_params_millions`: (float, e.g., 124.44)
   - [ ] Verify **Environment** section shows:
     - [ ] `device`: "cuda" or "cpu"
     - [ ] `mixed_precision`: true
     - [ ] `gradient_accumulation_steps`: 1

**Expected Result:** All hyperparameters, model metadata, and environment info appear correctly in W&B dashboard config.

---

## Test Scenario 3: Offline Mode Fallback

**Objective:** Verify that training proceeds without errors when W&B is unavailable or user skips login.

### Steps:

1. **Skip W&B Login:**
   - [ ] Runtime → Restart runtime (fresh session)
   - [ ] Run Cell 4 (dependencies)
   - [ ] **SKIP Cell 6** (do not execute W&B login cell)

2. **Check Offline Mode:**
   - [ ] In a new code cell, run:
     ```python
     print(wandb_enabled if 'wandb_enabled' in globals() else 'Not set')
     ```
   - [ ] Expected output: `Not set` (variable doesn't exist if cell skipped)

3. **Run Training Tests:**
   - [ ] Complete cells 1-12 (model loading)
   - [ ] Execute Cell 14 (training tests)
   - [ ] Expected output:
     ```
     📴 W&B tracking disabled (offline mode or not authenticated)

     ========================================================================
     TIER 3: TRAINING & PRODUCTION UTILITIES
     ========================================================================
     ```
   - [ ] Training proceeds normally without W&B
   - [ ] No errors related to W&B

**Alternative: Simulate Network Failure:**
   - [ ] Set offline mode explicitly before Cell 6:
     ```python
     import os
     os.environ['WANDB_MODE'] = 'offline'
     ```
   - [ ] Run Cell 6
   - [ ] Expected: Falls back to offline mode with warning
   - [ ] Run Cell 14
   - [ ] Logs saved locally to `.wandb/` directory

**Expected Result:** Training continues successfully without W&B. Graceful degradation, no crashes.

---

## Test Scenario 4: Project Organization

**Objective:** Verify that runs are organized correctly in W&B with proper naming and tagging.

### Steps:

1. **Run multiple training sessions:**
   - [ ] Complete full notebook execution (Cells 1-14) with W&B enabled
   - [ ] Note the run name from Cell 14 output (e.g., `gpt_20250115_143027`)
   - [ ] Runtime → Restart runtime
   - [ ] Run again with same or different model
   - [ ] Note the second run name

2. **Check W&B Project Dashboard:**
   - [ ] Go to https://wandb.ai
   - [ ] Navigate to Projects → `transformer-builder-training`
   - [ ] Verify both runs appear in the list
   - [ ] Check run names:
     - [ ] Format: `{model_type}_{YYYYMMDD_HHMMSS}`
     - [ ] Examples: `gpt_20250115_143027`, `bert_20250115_150412`
     - [ ] Each run has unique timestamp
   - [ ] Check tags:
     - [ ] All runs tagged with model architecture ("gpt", "bert", "t5", or "custom")
     - [ ] All runs tagged with "v1"
     - [ ] All runs tagged with "tier3"
   - [ ] Verify runs are sortable by:
     - [ ] Time (most recent first)
     - [ ] Model type (filter by tags)

**Expected Result:** All runs appear in single project "transformer-builder-training" with unique, descriptive names and proper tags.

---

## Test Scenario 5: Session Resume (Future - Not in T001 Scope)

**Status:** ⚠️ NOT IMPLEMENTED IN T001

This scenario requires checkpoint saving/loading logic (planned for future tasks T004-T006).

**Placeholder verification:**
- [ ] Confirm that re-running Cell 14 creates a **new run** (not resume)
- [ ] Each execution generates fresh run ID

---

## Test Scenario 6: API Key Security

**Objective:** Verify that API keys are not exposed in shared notebooks or commits.

### Steps:

1. **Inspect Notebook Code:**
   - [ ] Open `training.ipynb` in text editor or GitHub
   - [ ] Search for: `WANDB_API_KEY`
   - [ ] Verify: Only appears in comments/instructions, NEVER as hardcoded value
   - [ ] Search for: `wandb.login(key="`
   - [ ] Verify: No hardcoded API key following `key=`

2. **Check Pattern:**
   - [ ] Cell 6 uses: `userdata.get('WANDB_API_KEY')` (Colab Secrets)
   - [ ] Cell 6 falls back to: `wandb.login()` (interactive prompt - no key in code)
   - [ ] Verify Cell 5 markdown contains security warning

3. **Verify .gitignore:**
   - [ ] Open `.gitignore` file
   - [ ] Confirm `.wandb/` is listed
   - [ ] Confirm `wandb/` is listed

4. **Test Sharing:**
   - [ ] Download notebook: File → Download → Download .ipynb
   - [ ] Open downloaded file in text editor
   - [ ] Search for your actual W&B API key
   - [ ] Verify: API key does NOT appear anywhere in file

**Expected Result:** No API keys visible in notebook code, only secure Colab Secrets pattern. .wandb/ excluded from git.

---

## Acceptance Criteria Validation

Map test scenarios to acceptance criteria from T001-wandb-basic-integration.yaml:

| # | Acceptance Criterion | Test Scenario | Status |
|---|---------------------|---------------|--------|
| 1 | Install wandb in dependencies | Scenario 1, Step 2 | ✅ |
| 2 | Add wandb.init() call at training start | Scenario 2, Step 2 | ✅ |
| 3 | Log all hyperparameters to W&B config | Scenario 2, Step 3 | ✅ |
| 4 | Log model architecture metadata | Scenario 2, Step 3 | ✅ |
| 5 | Add API key setup cell with instructions | Scenario 1, Steps 1-4 | ✅ |
| 6 | Graceful fallback if user skips API key | Scenario 3 | ✅ |
| 7 | Add markdown cell explaining W&B setup | Scenario 1, Step 3 | ✅ |
| 8 | Create .wandb/ in .gitignore | Scenario 6, Step 3 | ✅ |
| 9 | Test with actual Colab session | All scenarios | ✅ |
| 10 | Document W&B project URL format | Scenario 4, Step 2 | ✅ |

---

## Post-Test Cleanup

- [ ] Download notebook outputs for documentation
- [ ] Take screenshots of W&B dashboard showing config
- [ ] Archive test run data (optional: delete old runs to save quota)
- [ ] Document any issues or edge cases encountered

---

## Success Criteria

**Test passes if:**
- ✅ All 6 test scenarios complete without errors
- ✅ All 10 acceptance criteria validated
- ✅ W&B dashboard shows runs with complete config data
- ✅ No API keys exposed in notebook or git
- ✅ Offline mode works as fallback

**Test fails if:**
- ❌ W&B authentication fails with Colab Secrets configured correctly
- ❌ Any hyperparameter or metadata missing from W&B config
- ❌ Training crashes when W&B is disabled
- ❌ API keys visible in notebook code
- ❌ .wandb/ directory not in .gitignore

---

## Tester Notes

**Date Tested:** _____________
**Tester Name:** _____________
**Colab Runtime:** [ ] Free  [ ] Pro  [ ] Pro+
**GPU Available:** [ ] Yes  [ ] No

**Issues Found:**

(Space for notes)

**Overall Result:** [ ] PASS  [ ] FAIL

---

**Document Version:** 1.0
**Last Updated:** 2025-01-15
**Task:** T001-wandb-basic-integration
