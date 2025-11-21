# Google Colab Validation Instructions

**Task:** P3-5 - Colab Notebook Validation
**Validator:** [YOUR NAME]
**Date:** 2025-11-20

---

## Overview

This document provides step-by-step instructions for validating `template.ipynb` and `training.ipynb` in actual Google Colab environments.

**Estimated Time:**
- Template validation: 30-45 minutes
- Training validation: 45-60 minutes
- **Total: ~2 hours**

**Requirements:**
- Google account (for Colab access)
- Valid Transformer Builder Gist ID (or use test Gist)
- Screenshots tool (for documentation)
- This validation guide

---

## Pre-Validation Checklist

Before starting validation, ensure you have:

- [ ] Google account with Colab access
- [ ] Test Gist ID from Transformer Builder (or create one)
- [ ] Screenshots folder ready: `docs/screenshots/`
- [ ] Validation report open: `docs/COLAB_VALIDATION_REPORT.md`
- [ ] Troubleshooting guide open: `docs/COLAB_TROUBLESHOOTING.md`
- [ ] Internet connection stable
- [ ] 2+ hours of focused time

---

## Part 1: Template.ipynb Validation

### Step 1.1: Open Template in Fresh Colab Runtime

1. **Open Colab:** https://colab.research.google.com
2. **Upload Notebook:**
   - File → Upload notebook
   - Select `template.ipynb` from local repo
   - OR open from GitHub: `https://github.com/matt-hans/transformer-builder-colab-templates/blob/main/template.ipynb`

3. **Enable GPU (Optional for Tier 1):**
   - Runtime → Change runtime type
   - Hardware accelerator → GPU
   - Save

4. **Verify Fresh Runtime:**
   - Runtime → View runtime logs
   - Should show "Initializing runtime..."

**Screenshot:** Save as `docs/screenshots/template_01_colab_open.png`

---

### Step 1.2: Verify Zero-Installation Strategy

**Objective:** Confirm no pip installs in template.ipynb

1. **Scan Notebook Cells:**
   - Look for any `!pip install` commands
   - Look for any `%pip install` magic commands
   - Verify Cell 6 (Dependency Verification) says "Zero Installation Strategy"

2. **Expected Result:**
   - ✅ No pip install commands found
   - ✅ Cell 6 output: "No installation needed - using Colab pre-installed packages"

3. **Record in Report:**
   ```
   Template.ipynb > Zero-Installation Strategy > Status: PASS/FAIL
   ```

**Screenshot:** Save Cell 6 output as `docs/screenshots/template_02_zero_install.png`

---

### Step 1.3: Check Colab Runtime Versions

1. **Run Version Check Cell (create new cell):**

```python
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

print("=" * 70)
print("COLAB RUNTIME VERSIONS")
print("=" * 70)
print()
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {sns.__version__}")
print()

# GPU info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU: Not allocated (CPU runtime)")
print()
print("=" * 70)
```

2. **Record Versions in Report:**
   - Copy output to `COLAB_VALIDATION_REPORT.md > Environment Specifications`
   - Update `requirements-colab-v3.4.0.txt` if versions differ significantly

**Screenshot:** Save output as `docs/screenshots/template_03_versions.png`

---

### Step 1.4: Test Gist Loading (Valid ID)

**Objective:** Load model from valid Transformer Builder Gist

1. **Get Test Gist ID:**
   - Option A: Use your own Transformer Builder export
   - Option B: Use test Gist: `[INSERT TEST GIST ID HERE]`

2. **Run Cell 5 (Gist ID Input):**
   - Paste Gist ID in the text field
   - Run cell
   - Expected output: "✅ GIST ID SAVED"

3. **Run Cell 8 (Load Model):**
   - Expected output:
     ```
     ✅ Model loaded successfully!
     ✅ Gist URL: https://gist.github.com/...
     ✅ Model code: X,XXX bytes
     ✅ Config: XXX bytes
     ```

4. **Verify Files Created:**
   ```python
   !ls -lh custom_transformer.py config.json
   ```
   - Should show both files exist

5. **Record in Report:**
   ```
   Template.ipynb > Transformer Builder Integration > Status: PASS/FAIL
   Test Gist ID: [INSERT ID]
   ```

**Screenshot:** Save Cell 8 output as `docs/screenshots/template_04_gist_valid.png`

---

### Step 1.5: Test Gist Loading (Invalid ID)

**Objective:** Verify graceful error handling

1. **Create New Cell Above Cell 8:**

```python
# Test invalid Gist ID
GIST_ID = "invalid_test_12345"
```

2. **Run Cell 8 (Load Model):**
   - Expected output: User-friendly error message
   - Should mention:
     - "Gist not found (check your Gist ID)"
     - Troubleshooting steps
     - No Python traceback (handled gracefully)

3. **Restore Valid Gist ID:**
   - Re-run Cell 5 with valid ID

4. **Record in Report:**
   ```
   Template.ipynb > Error Handling > Invalid Gist ID > Status: PASS/FAIL
   Error Message User-Friendly: YES/NO
   ```

**Screenshot:** Save error output as `docs/screenshots/template_05_gist_invalid.png`

---

### Step 1.6: View Model Code

1. **Run Cell 11 (View Loaded Model Code):**
   - Should display syntax-highlighted Python code
   - Should display JSON config

2. **Verify:**
   - Code is readable
   - No obvious syntax errors
   - Config contains expected fields (vocab_size, max_seq_len, etc.)

3. **Record in Report:**
   ```
   Template.ipynb > Model Code Display > Status: PASS/FAIL
   ```

**Screenshot:** Save (partial) as `docs/screenshots/template_06_model_code.png`

---

### Step 1.7: Model Instantiation

1. **Run Cell 13 (Import and Instantiate Model):**
   - Expected output:
     ```
     ✅ Model instantiated: CustomTransformer
     ✅ Total parameters: X,XXX,XXX
     ✅ Trainable parameters: X,XXX,XXX
     ✅ Device: cuda:0 (or cpu)
     ```

2. **Verify:**
   - No errors during instantiation
   - Parameter count reasonable (not 0)
   - Model moved to GPU if available

3. **Record in Report:**
   ```
   Template.ipynb > Model Instantiation > Status: PASS/FAIL
   Total Parameters: [NUMBER]
   Device: [cuda:0 / cpu]
   ```

**Screenshot:** Save as `docs/screenshots/template_07_instantiation.png`

---

### Step 1.8: Run Tier 1 Tests

**Objective:** All 6 Tier 1 tests pass in under 2 minutes

1. **Start Timer**

2. **Run Cell 16 (Tier 1 Tests):**
   - Test 1/6: Shape Validation
   - Test 2/6: Gradient Flow Analysis
   - Test 3/6: Numerical Stability
   - Test 4/6: Parameter Initialization
   - Test 5/6: Memory Footprint Analysis
   - Test 6/6: Inference Speed Benchmark

3. **Stop Timer**

4. **Verify Each Test:**

| Test | Expected Result |
|------|----------------|
| Shape Robustness | All input shapes produce valid outputs |
| Gradient Flow | No vanishing/exploding gradients |
| Output Stability | No NaN/Inf, standard deviation reasonable |
| Parameter Init | Mean ~0, std ~0.02-0.1 |
| Memory Footprint | Linear growth, no leaks |
| Inference Speed | Reasonable throughput (varies by model) |

5. **Record in Report:**
   ```
   Template.ipynb > Tier 1 Tests > Status: PASS/FAIL
   Total Duration: [TIME]

   Individual Tests:
   - Shape Robustness: [PASS/FAIL]
   - Gradient Flow: [PASS/FAIL]
   - Output Stability: [PASS/FAIL]
   - Parameter Init: [PASS/FAIL]
   - Memory Footprint: [PASS/FAIL]
   - Inference Speed: [PASS/FAIL]
   ```

**Screenshot:** Save Tier 1 summary as `docs/screenshots/template_08_tier1.png`

---

### Step 1.9: Run Tier 2 Tests

**Objective:** Tier 2 tests run without errors (results may vary by model)

1. **Run Cell 19 (Tier 2 Tests):**
   - Test 1/2: Attention Pattern Analysis
   - Test 2/2: Robustness Under Noise

2. **Verify:**
   - Tests complete without crashes
   - Results displayed (may show "⚠️ skipped" if model lacks attention)

3. **Record in Report:**
   ```
   Template.ipynb > Tier 2 Tests > Status: PASS/FAIL
   Attention Patterns: [PASS/FAIL/SKIPPED]
   Robustness Testing: [PASS/FAIL]
   ```

**Screenshot:** Save as `docs/screenshots/template_09_tier2.png`

---

### Step 1.10: Check NumPy Integrity (Post-Testing)

**Objective:** Ensure NumPy remains uncorrupted after all tests

1. **Create New Cell:**

```python
print("=" * 70)
print("NUMPY INTEGRITY CHECK (POST-TESTING)")
print("=" * 70)
print()

try:
    from numpy._core.umath import _center
    print("✅ NumPy C extensions intact")
    print("✅ No corruption detected")
except ImportError as e:
    print("❌ NumPy corrupted!")
    print(f"Error: {e}")
    print()
    print("This indicates:")
    print("  1. A package was installed that corrupted NumPy")
    print("  2. Runtime needs restart")

print()
print("=" * 70)
```

2. **Expected Result:** ✅ NumPy C extensions intact

3. **Record in Report:**
   ```
   Template.ipynb > NumPy Corruption Check > Status: PASS/FAIL
   ```

**Screenshot:** Save as `docs/screenshots/template_10_numpy_check.png`

---

### Step 1.11: Test Cell-by-Cell Execution

**Objective:** Verify no cell-order dependencies

1. **Runtime → Restart runtime**
2. **Clear all outputs**
3. **Execute cells ONE BY ONE:**
   - Click each cell individually
   - Press Shift+Enter
   - Do NOT use "Run All"

4. **Verify:**
   - Each cell completes successfully
   - No "NameError: name 'X' is not defined"
   - State builds correctly across cells

5. **Record in Report:**
   ```
   Template.ipynb > Cell-by-Cell Execution > Status: PASS/FAIL
   Issues: [LIST ANY CELL-ORDER DEPENDENCIES]
   ```

---

### Step 1.12: Test "Run All" Execution

**Objective:** Verify "Run All" works end-to-end

1. **Runtime → Restart runtime**
2. **Edit → Clear all outputs**
3. **Runtime → Run all**
4. **Wait for completion** (should take ~2-3 minutes)

5. **Verify:**
   - All cells execute without errors
   - Gist ID cell prompts for input (you'll need to paste ID manually)
   - Final output: "✅ TIER 2 ANALYSIS COMPLETE"

6. **Record in Report:**
   ```
   Template.ipynb > Run All Execution > Status: PASS/FAIL
   Total Duration: [TIME]
   ```

**Screenshot:** Save final cell output as `docs/screenshots/template_11_run_all.png`

---

### Step 1.13: Memory Usage Check

**Objective:** Verify memory usage within 12GB Colab free tier limit

1. **During Testing, Monitor:**
   - Top right corner: RAM usage bar
   - Should stay well below 12GB

2. **Create New Cell (After All Tests):**

```python
import psutil

mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / 1e9:.1f} GB")
print(f"Used RAM: {mem.used / 1e9:.1f} GB")
print(f"Available RAM: {mem.available / 1e9:.1f} GB")
print(f"Usage: {mem.percent}%")

if mem.percent > 80:
    print("⚠️ High memory usage - approaching 12GB limit")
else:
    print("✅ Memory usage acceptable")
```

3. **Record in Report:**
   ```
   Template.ipynb > Memory Usage > Peak Memory: [X.X GB]
   Status: PASS/FAIL (under 12GB)
   ```

---

## Part 2: Training.ipynb Validation

### Step 2.1: Open Training Notebook in FRESH Runtime

**CRITICAL: Must use fresh runtime, NOT same session as template.ipynb**

1. **Close Template.ipynb tab** (or just open new tab)
2. **Open Colab:** https://colab.research.google.com
3. **Upload training.ipynb** or open from GitHub
4. **Enable GPU (REQUIRED):**
   - Runtime → Change runtime type
   - Hardware accelerator → GPU
   - Save

5. **Verify Fresh Runtime:**
   - Should NOT have `model` or `config` variables from template

**Screenshot:** Save as `docs/screenshots/training_01_fresh_runtime.png`

---

### Step 2.2: Install Training Dependencies

1. **Create Installation Cell (if not present):**

```python
print("=" * 70)
print("INSTALLING TRAINING DEPENDENCIES")
print("=" * 70)
print()

!pip install -r https://raw.githubusercontent.com/matt-hans/transformer-builder-colab-templates/main/requirements-training.txt

print()
print("=" * 70)
print("✅ INSTALLATION COMPLETE")
print("=" * 70)
```

2. **Run Installation Cell**
   - Should take 2-3 minutes
   - Expected: All packages install without conflicts

3. **Verify Installation:**

```python
import pytorch_lightning as pl
import optuna
import torchmetrics
import wandb

print(f"PyTorch Lightning: {pl.__version__}")
print(f"Optuna: {optuna.__version__}")
print(f"TorchMetrics: {torchmetrics.__version__}")
print(f"W&B: {wandb.__version__}")
```

4. **Record in Report:**
   ```
   Training.ipynb > Dependency Installation > Status: PASS/FAIL
   pytorch-lightning: [VERSION]
   optuna: [VERSION]
   torchmetrics: [VERSION]
   wandb: [VERSION]
   ```

**Screenshot:** Save installation output as `docs/screenshots/training_02_install.png`

---

### Step 2.3: Download Utils Package

1. **Run Utils Download Cell:**

```python
!rm -rf utils/
!git clone --depth 1 https://github.com/matt-hans/transformer-builder-colab-templates.git temp_repo
!cp -r temp_repo/utils ./
!rm -rf temp_repo

# Verify
import sys
sys.path.insert(0, './')
from utils.tier3_training_utilities import test_fine_tuning
print("✅ Utils package loaded")
```

2. **Expected Result:** No errors, utils imported successfully

**Screenshot:** Save as `docs/screenshots/training_03_utils.png`

---

### Step 2.4: Load Model from Gist

1. **Run Gist Loading Cells** (same as template.ipynb)
   - Paste same Gist ID from template validation
   - Verify model loads successfully

2. **Expected Result:**
   - Same model as in template.ipynb
   - `model` and `config` variables created

3. **Record in Report:**
   ```
   Training.ipynb > Model Loading > Status: PASS/FAIL
   ```

---

### Step 2.5: Run Small Training Experiment (3 Epochs)

**Objective:** Verify training loop works end-to-end

1. **Create Training Cell:**

```python
from utils.tier3_training_utilities import test_fine_tuning
from utils.training.training_config import TrainingConfig
import time

print("=" * 70)
print("SMALL TRAINING EXPERIMENT (3 EPOCHS)")
print("=" * 70)
print()

# Configure for quick test
training_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=3,
    random_seed=42,
    deterministic=False,  # Fast mode
    use_wandb=False  # Disable W&B for quick test
)

print("Training Configuration:")
print(f"  Epochs: {training_config.epochs}")
print(f"  Batch Size: {training_config.batch_size}")
print(f"  Learning Rate: {training_config.learning_rate}")
print()

# Run training
start_time = time.time()
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=training_config.epochs,
    learning_rate=training_config.learning_rate,
    batch_size=training_config.batch_size,
    deterministic=training_config.deterministic,
    use_wandb=False
)
duration = time.time() - start_time

print()
print("=" * 70)
print("TRAINING RESULTS")
print("=" * 70)
print(f"Duration: {duration:.1f}s")
print(f"Final Loss: {results['final_loss']:.4f}")
print(f"Loss History: {results['loss_history']}")
print()
print("✅ Training completed successfully!")
```

2. **Expected Results:**
   - Training completes in 2-5 minutes
   - Loss decreases across epochs
   - No OOM errors
   - No NaN/Inf losses

3. **Record in Report:**
   ```
   Training.ipynb > Small Training Experiment > Status: PASS/FAIL
   Duration: [TIME]
   Epoch 1 Loss: [VALUE]
   Epoch 2 Loss: [VALUE]
   Epoch 3 Loss: [VALUE]
   Final Loss: [VALUE]
   ```

**Screenshot:** Save training output as `docs/screenshots/training_04_train_3epochs.png`

---

### Step 2.6: Test Checkpoint Save/Resume

1. **Create Checkpoint Test Cell:**

```python
import torch
import os

print("=" * 70)
print("CHECKPOINT SAVE/RESUME TEST")
print("=" * 70)
print()

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

# Save checkpoint
checkpoint_path = 'checkpoints/test_checkpoint.pt'
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': None,  # Would include optimizer in real scenario
    'loss': results['loss_history'][0]
}, checkpoint_path)

print(f"✅ Checkpoint saved: {checkpoint_path}")
print(f"   Size: {os.path.getsize(checkpoint_path) / 1e6:.2f} MB")
print()

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

print("✅ Checkpoint loaded successfully")
print(f"   Saved at epoch: {checkpoint['epoch']}")
print(f"   Saved loss: {checkpoint['loss']:.4f}")
print()

# Verify model still works after loading
test_input = torch.randint(0, config.vocab_size, (1, 10))
if torch.cuda.is_available():
    test_input = test_input.cuda()
    model = model.cuda()

with torch.no_grad():
    test_output = model(test_input)

print("✅ Model inference works after checkpoint load")
print("=" * 70)
```

2. **Expected Result:**
   - Checkpoint saves without errors
   - Checkpoint loads successfully
   - Model works after loading

3. **Record in Report:**
   ```
   Training.ipynb > Checkpoint Save/Resume > Status: PASS/FAIL
   Checkpoint Size: [X.X MB]
   ```

**Screenshot:** Save as `docs/screenshots/training_05_checkpoint.png`

---

### Step 2.7: Test W&B Integration (Optional)

**Option A: With W&B Account**

1. **Login to W&B:**
```python
import wandb
wandb.login()  # Follow prompts to enter API key
```

2. **Run Training with W&B:**
```python
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=3,
    use_wandb=True,
    wandb_project="colab-validation-test"
)
```

3. **Verify:**
   - W&B run created
   - Metrics logged
   - Dashboard accessible

**Option B: Offline Mode**

1. **Set Offline Mode:**
```python
import os
os.environ['WANDB_MODE'] = 'offline'
```

2. **Run Training:**
```python
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=3,
    use_wandb=True
)
```

3. **Verify:**
   - No errors
   - Offline logs created in `wandb/` directory

**Option C: Disabled**

```python
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=3,
    use_wandb=False
)
```

4. **Record in Report:**
   ```
   Training.ipynb > W&B Integration > Status: PASS/FAIL/SKIPPED
   Mode: [Online/Offline/Disabled]
   W&B Run URL: [URL or N/A]
   ```

---

### Step 2.8: Test Export Bundle Generation

1. **Create Export Test Cell:**

```python
from utils.training.export_utilities import create_export_bundle
from utils.training.training_config import TrainingConfig
from utils.tokenization.task_spec import TaskSpec

print("=" * 70)
print("EXPORT BUNDLE GENERATION TEST")
print("=" * 70)
print()

# Configure export
export_config = TrainingConfig(
    export_bundle=True,
    export_formats=["pytorch", "torchscript"],  # Skip ONNX for speed
    export_dir="exports"
)

# Create minimal task spec
task_spec = TaskSpec.text_tiny()

# Generate export bundle
try:
    export_path = create_export_bundle(
        model=model,
        config=config,
        task_spec=task_spec,
        training_config=export_config
    )

    print(f"✅ Export bundle created: {export_path}")
    print()

    # List generated files
    import os
    print("Generated files:")
    for root, dirs, files in os.walk(export_path):
        level = root.replace(export_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

    print()
    print("=" * 70)
    print("✅ EXPORT BUNDLE TEST COMPLETE")
    print("=" * 70)

except Exception as e:
    print(f"❌ Export failed: {e}")
    import traceback
    traceback.print_exc()
```

2. **Expected Result:**
   - Export bundle created
   - Directory structure matches spec
   - Files include: model files, configs, inference.py, README.md, Dockerfile

3. **Verify Key Files:**
```python
# Check if key files exist
import os
key_files = [
    'artifacts/model.pytorch.pt',
    'artifacts/model.torchscript.pt',
    'configs/task_spec.json',
    'configs/training_config.json',
    'inference.py',
    'README.md',
    'Dockerfile',
    'requirements.txt'
]

for file in key_files:
    file_path = os.path.join(export_path, file)
    exists = "✅" if os.path.exists(file_path) else "❌"
    print(f"{exists} {file}")
```

4. **Record in Report:**
   ```
   Training.ipynb > Export Bundle > Status: PASS/FAIL
   Exported Formats: [LIST]
   Missing Files: [LIST or NONE]
   ```

**Screenshot:** Save file listing as `docs/screenshots/training_06_export.png`

---

### Step 2.9: Monitor GPU Memory

1. **Create Memory Monitoring Cell:**

```python
import torch

print("=" * 70)
print("GPU MEMORY USAGE")
print("=" * 70)
print()

if torch.cuda.is_available():
    # Get GPU properties
    gpu_props = torch.cuda.get_device_properties(0)
    total_memory = gpu_props.total_memory / 1e9

    # Current usage
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    # Peak usage
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {total_memory:.1f} GB")
    print()
    print(f"Current Allocated: {allocated:.2f} GB")
    print(f"Current Reserved: {reserved:.2f} GB")
    print(f"Peak Allocated: {peak_allocated:.2f} GB")
    print()

    if peak_allocated < total_memory * 0.8:
        print("✅ Memory usage healthy (< 80% of total)")
    else:
        print("⚠️ High memory usage (> 80% of total)")

    # Detailed memory summary
    print()
    print("Detailed Memory Summary:")
    print(torch.cuda.memory_summary(device=0, abbreviated=True))
else:
    print("❌ No GPU available")

print("=" * 70)
```

2. **Record in Report:**
   ```
   Training.ipynb > GPU Memory > Peak Allocated: [X.X GB]
   Total GPU Memory: [X.X GB]
   Status: PASS/FAIL (under 80%)
   ```

**Screenshot:** Save as `docs/screenshots/training_07_gpu_memory.png`

---

## Part 3: Cross-Cutting Validation

### Step 3.1: Runtime Selection Testing

**Objective:** Verify notebooks work on different runtime types

1. **Template.ipynb - CPU Runtime:**
   - Runtime → Change runtime type → None
   - Run all cells
   - Expected: Works but slower (Tier 1 may take 3-5 minutes)

2. **Template.ipynb - GPU Runtime:**
   - Runtime → Change runtime type → T4 GPU
   - Run all cells
   - Expected: Faster (Tier 1 completes in ~1 minute)

3. **Training.ipynb - GPU Required:**
   - Must use GPU runtime
   - CPU runtime too slow for practical validation

4. **Record in Report:**
   ```
   Cross-Cutting > Runtime Selection:
   - Template CPU: [PASS/FAIL]
   - Template GPU: [PASS/FAIL]
   - Training GPU: [PASS/FAIL]
   ```

---

### Step 3.2: Session Timeout Testing

**Objective:** Estimate if workflows complete within 12-hour limit

1. **Calculate Total Durations:**
   - Template.ipynb (all tiers): [TIME from validation]
   - Training.ipynb (3 epochs): [TIME from validation]
   - Training.ipynb (10 epochs - estimated): [TIME × 3.33]

2. **Project for Full Workflow:**
   - Template validation: ~3-5 minutes
   - Training (10 epochs): ~10-15 minutes
   - Export generation: ~2-3 minutes
   - **Total: ~15-23 minutes** (well under 12 hours)

3. **Record in Report:**
   ```
   Cross-Cutting > Session Timeout:
   - Template All Tiers: [TIME] (12h limit: PASS)
   - Training 3 Epochs: [TIME] (12h limit: PASS)
   - Training 10 Epochs (est): [TIME] (12h limit: PASS/FAIL)
   ```

---

## Part 4: Post-Validation Tasks

### Step 4.1: Update Validation Report

1. **Fill All [TO BE FILLED] Sections:**
   - Environment specifications
   - All test results
   - All screenshots
   - Issues summary

2. **Complete Validation Checklist:**
   - Check all boxes that passed
   - Note all failures

3. **Add Screenshots:**
   - Ensure all screenshots captured
   - Name consistently: `template_XX_description.png`
   - Place in `docs/screenshots/`

---

### Step 4.2: Update Requirements File (If Needed)

1. **Compare Detected Versions with requirements-colab-v3.4.0.txt:**

```python
# From Step 1.3 version check
# torch: [ACTUAL] vs requirements: 2.6+
# numpy: [ACTUAL] vs requirements: 2.3+
# etc.
```

2. **If Significant Deviation:**
   - Update `requirements-colab-v3.4.0.txt`
   - Document changes in git commit message

---

### Step 4.3: Document Issues and Workarounds

1. **For Each Issue Found:**
   - Add to Issues Summary in validation report
   - Add to troubleshooting guide if reproducible
   - Create GitHub issue if blocking

2. **Categorize Issues:**
   - Critical (blocking): Prevents notebook from working
   - High: Degrades user experience significantly
   - Medium: Minor annoyance with workaround
   - Low: Enhancement opportunity

---

### Step 4.4: Create Summary Document

1. **Create `COLAB_VALIDATION_SUMMARY.md`:**

```markdown
# Colab Validation Summary

**Date:** 2025-11-20
**Validator:** [NAME]
**Duration:** [X hours]

## Results

### Template.ipynb
- Status: PASS/FAIL
- Key Issues: [SUMMARY]
- Recommendations: [SUMMARY]

### Training.ipynb
- Status: PASS/FAIL
- Key Issues: [SUMMARY]
- Recommendations: [SUMMARY]

## Action Items
1. [Action item with owner]
2. [Action item with owner]

## Approval
Ready for production: YES/NO
Blockers: [LIST or NONE]
```

---

## Validation Complete!

**Final Checklist:**

- [ ] Both notebooks validated in actual Colab environment
- [ ] All screenshots captured and organized
- [ ] Validation report filled completely
- [ ] Requirements file updated (if needed)
- [ ] Troubleshooting guide updated (if issues found)
- [ ] Summary document created
- [ ] GitHub issues created for blockers
- [ ] Validation results communicated to team

**Estimated Completion Time:** 2 hours

**Next Steps:**
- Address critical issues
- Update documentation based on findings
- Plan for regression testing in CI/CD

---

**Document Version:** 1.0
**Last Updated:** 2025-11-20
