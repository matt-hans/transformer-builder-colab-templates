# Load Model Weights Cell Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Cell 34 that loads trained model weights from checkpoints into the initialized model, making it ready for inference and additional training.

**Architecture:** Reuse existing `list_checkpoints()` utility from recovery.py. Default to Cell 33's checkpoint selection if available, otherwise auto-select best by val_loss. Load weights with strict=True, migrate to GPU, set eval() mode, display comprehensive info (checkpoint metadata, model summary, architecture preview).

**Tech Stack:** Python 3.x, PyTorch, Jupyter notebook JSON structure, existing recovery utilities

---

## Task 1: Read Cell 34 JSON to Understand Insertion Point

**Files:**
- Read: `training.ipynb` (find Cell 34 structure)

**Step 1: Extract Cell 34 from notebook**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
cell_34 = nb['cells'][34]
print('Cell 34 title:', cell_34['source'][0][:80])
print('Cell 34 index:', 34)
print('Total cells:', len(nb['cells']))
"
```

Expected output:
```
Cell 34 title: # @title 🔧 Extract Session Variables { display-mode: "form" }
Cell 34 index: 34
Total cells: 45
```

**Step 2: Verify Cell 33 is Checkpoint Recovery**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
cell_33 = nb['cells'][33]
print('Cell 33 title:', cell_33['source'][0][:80])
"
```

Expected output:
```
Cell 33 title: # @title 🔧 Optional: Recover Training Results from Checkpoint { display-
```

**Step 3: Commit reconnaissance findings**

```bash
git add -A
git commit -m "docs: add implementation plan for load model weights cell"
```

---

## Task 2: Create New Cell 34 JSON Structure

**Files:**
- Read: `training.ipynb` (to get template from Cell 33)
- Read: `/tmp/cell_34.json` (from session context - the fixed Cell 33 structure)
- Create: `scripts/insert_cell_34.py` (helper script)

**Step 1: Write script to insert new cell at index 34**

Create `scripts/insert_cell_34.py`:

```python
#!/usr/bin/env python3
"""Insert new Cell 34 (Load Model Weights) into training.ipynb."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

# Read notebook
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

# Create new Cell 34 structure (placeholder content)
new_cell_34 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "cellView": "form"
    },
    "outputs": [],
    "source": [
        "# @title 🔧 Optional: Load Model Weights from Checkpoint { display-mode: \"form\" }\n",
        "\n",
        "# PLACEHOLDER - Will be populated in next task\n",
        "print(\"Cell 34 placeholder\")\n"
    ]
}

# Insert at index 34 (between Cell 33 and current Cell 34)
nb['cells'].insert(34, new_cell_34)

# Save notebook
with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✅ Inserted new Cell 34 at index 34")
print(f"   Total cells: {len(nb['cells'])}")
print(f"   Cell 33: {nb['cells'][33]['source'][0][:60]}")
print(f"   Cell 34 (NEW): {nb['cells'][34]['source'][0][:60]}")
print(f"   Cell 35 (was 34): {nb['cells'][35]['source'][0][:60]}")
```

**Step 2: Run script to insert placeholder cell**

```bash
python3 scripts/insert_cell_34.py
```

Expected output:
```
✅ Inserted new Cell 34 at index 34
   Total cells: 46
   Cell 33: # @title 🔧 Optional: Recover Training Results from Ch
   Cell 34 (NEW): # @title 🔧 Optional: Load Model Weights from Checkpo
   Cell 35 (was 34): # @title 🔧 Extract Session Variables { display-mode: "
```

**Step 3: Verify notebook structure**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
for i in [33, 34, 35]:
    title = nb['cells'][i]['source'][0][:70]
    print(f'Cell {i}: {title}')
"
```

Expected output shows Cell 34 inserted correctly.

**Step 4: Commit cell insertion**

```bash
git add training.ipynb scripts/insert_cell_34.py
git commit -m "feat(notebook): insert Cell 34 placeholder for model weight loading

Inserted new cell at index 34 between:
- Cell 33: Checkpoint Recovery (metrics only)
- Cell 35 (was 34): Extract Session Variables

Cell 34 will load model_state_dict from checkpoint into model.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Implement Cell 34 - Prerequisites Validation Section

**Files:**
- Modify: `training.ipynb` (Cell 34 source)
- Create: `scripts/update_cell_34.py` (updater script)

**Step 1: Write script to update Cell 34 source code**

Create `scripts/update_cell_34.py`:

```python
#!/usr/bin/env python3
"""Update Cell 34 source with full implementation."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')
CELL_INDEX = 34

# Full Cell 34 source code (prerequisites section)
CELL_34_SOURCE = [
    "# @title 🔧 Optional: Load Model Weights from Checkpoint { display-mode: \"form\" }\n",
    "\n",
    "# =============================================================================\n",
    "# LOAD MODEL WEIGHTS FROM CHECKPOINT\n",
    "# =============================================================================\n",
    "# This cell loads model weights from a checkpoint into an existing model instance.\n",
    "#\n",
    "# Prerequisites:\n",
    "#   1. Run Cell 13 (Initialize Model) first to create 'model' variable\n",
    "#   2. Run Cell 32 (Training) OR Cell 33 (Recovery) to have checkpoints\n",
    "#\n",
    "# Use Cases:\n",
    "#   - Inference on trained model\n",
    "#   - Continue training from checkpoint\n",
    "#   - Evaluate model on test data\n",
    "# =============================================================================\n",
    "\n",
    "from pathlib import Path\n",
    "from utils.training.engine.recovery import list_checkpoints\n",
    "import torch\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(\"LOAD MODEL WEIGHTS FROM CHECKPOINT\")\n",
    "print(\"=\" * 70)\n",
    "print()\n",
    "\n",
    "# Validate prerequisites\n",
    "if 'model' not in globals():\n",
    "    print(\"❌ Model instance not found!\")\n",
    "    print()\n",
    "    print(\"💡 To fix:\")\n",
    "    print(\"   1. Run Cell 13 (Initialize Model) to create 'model' variable\")\n",
    "    print(\"   2. Re-run this cell\")\n",
    "    print()\n",
    "    print(\"=\" * 70)\n",
    "\n"
]

def update_cell_34_source(source_lines):
    """Update Cell 34 with new source code."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Update Cell 34 source
    nb['cells'][CELL_INDEX]['source'] = source_lines

    # Save notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"✅ Updated Cell {CELL_INDEX}")
    print(f"   Lines of source: {len(source_lines)}")

if __name__ == "__main__":
    update_cell_34_source(CELL_34_SOURCE)
```

**Step 2: Run script to update Cell 34 (prerequisites only)**

```bash
python3 scripts/update_cell_34.py
```

Expected output:
```
✅ Updated Cell 34
   Lines of source: 38
```

**Step 3: Verify Cell 34 source**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
cell_34 = nb['cells'][34]
print('Cell 34 source lines:', len(cell_34['source']))
print('First 10 lines:')
for i, line in enumerate(cell_34['source'][:10]):
    print(f'{i+1:2}: {line}', end='')
"
```

Expected: Shows Cell 34 with header and prerequisites validation.

**Step 4: Commit prerequisites section**

```bash
git add training.ipynb scripts/update_cell_34.py
git commit -m "feat(cell-34): add prerequisites validation section

Validates model variable exists before attempting to load weights.
Provides clear error message directing user to Cell 13.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Implement Cell 34 - Checkpoint Discovery Section

**Files:**
- Modify: `scripts/update_cell_34.py` (extend source)

**Step 1: Extend CELL_34_SOURCE with checkpoint discovery**

Modify `scripts/update_cell_34.py`, replace `CELL_34_SOURCE` with extended version:

```python
CELL_34_SOURCE = [
    # ... (keep previous lines 1-38) ...
    "else:\n",
    "    # Scan checkpoint directory\n",
    "    checkpoint_dir = './checkpoints'\n",
    "    checkpoints = list_checkpoints(checkpoint_dir)\n",
    "    \n",
    "    if not checkpoints:\n",
    "        print(\"❌ No checkpoints found in ./checkpoints/\")\n",
    "        print()\n",
    "        print(\"💡 To fix:\")\n",
    "        print(\"   1. Run Cell 32 (Run Training) to create checkpoints\")\n",
    "        print(\"   2. Or verify checkpoint directory path\")\n",
    "        print()\n",
    "        print(\"=\" * 70)\n",
    "    \n",
    "    else:\n",
    "        # Determine which checkpoint to load\n",
    "        if 'results' in globals() and 'checkpoint_path' in results:\n",
    "            # Use checkpoint from Cell 33 (Checkpoint Recovery)\n",
    "            checkpoint_path = results['checkpoint_path']\n",
    "            selection_method = \"From Cell 33 (Checkpoint Recovery)\"\n",
    "        else:\n",
    "            # Auto-select best checkpoint (first in list = best val_loss)\n",
    "            checkpoint_path = checkpoints[0]['path']\n",
    "            selection_method = \"Auto-selected (Best val_loss)\"\n",
    "        \n",
    "        print(\"=\" * 70)\n",
    "        print(\"AVAILABLE CHECKPOINTS\")\n",
    "        print(\"=\" * 70)\n",
    "        print()\n",
    "        \n",
    "        # Display checkpoint list\n",
    "        for i, ckpt in enumerate(checkpoints[:10]):  # Show top 10\n",
    "            # Mark selected checkpoint\n",
    "            marker = \"→\" if ckpt['path'] == checkpoint_path else \" \"\n",
    "            \n",
    "            print(f\"{marker} [{i}] Epoch {ckpt['epoch']:2d} | \"\n",
    "                  f\"Step {ckpt['global_step']:5d} | \"\n",
    "                  f\"train_loss={ckpt['train_loss']:.4f} | \"\n",
    "                  f\"val_loss={ckpt['val_loss']:.4f}\")\n",
    "            print(f\"     {ckpt['filename']}\")\n",
    "            print()\n",
    "        \n",
    "        if len(checkpoints) > 10:\n",
    "            print(f\"... and {len(checkpoints) - 10} more checkpoints\")\n",
    "            print()\n",
    "        \n",
    "        print(f\"📂 Selected: {Path(checkpoint_path).name}\")\n",
    "        print(f\"🔍 Selection Method: {selection_method}\")\n",
    "        print()\n",
    "        \n",
    "        # Manual override option (commented)\n",
    "        print(\"# Override: Uncomment and modify to load different checkpoint\")\n",
    "        print(\"# checkpoint_path = checkpoints[2]['path']  # Load epoch 7 checkpoint\")\n",
    "        print()\n",
    "\n"
]
```

**Step 2: Run script to update Cell 34**

```bash
python3 scripts/update_cell_34.py
```

Expected output:
```
✅ Updated Cell 34
   Lines of source: 93
```

**Step 3: Commit checkpoint discovery section**

```bash
git add scripts/update_cell_34.py
git commit -m "feat(cell-34): add checkpoint discovery and selection logic

Uses list_checkpoints() from recovery.py to scan ./checkpoints/.
Selection priority:
1. results['checkpoint_path'] (from Cell 33)
2. Auto-select best by val_loss
3. Manual override via uncommented code

Displays top 10 checkpoints with metrics for informed selection.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Implement Cell 34 - Weight Loading Section

**Files:**
- Modify: `scripts/update_cell_34.py` (extend source)

**Step 1: Extend CELL_34_SOURCE with weight loading**

Add to `CELL_34_SOURCE` after checkpoint discovery section:

```python
        # Load checkpoint and weights
        try:
            print(\"=\" * 70)\n",
            print(\"LOADING MODEL WEIGHTS\")\n",
            print(\"=\" * 70)\n",
            print()\n",
            \n",
            # Load checkpoint\n",
            print(f\"📂 Loading checkpoint: {Path(checkpoint_path).name}\")\n",
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)\n",
            \n",
            # Extract model weights\n",
            model_state_dict = checkpoint['model_state_dict']\n",
            \n",
            # Load weights into model (strict=True ensures architecture match)\n",
            model.load_state_dict(model_state_dict, strict=True)\n",
            \n",
            print(\"✅ Weights loaded successfully!\")\n",
            print()\n",
            \n",
            # Determine target device\n",
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            \n",
            # Move model to device\n",
            model = model.to(device)\n",
            \n",
            # Set to evaluation mode (disables dropout, batchnorm updates)\n",
            model.eval()\n",
            \n",
            print(f\"📍 Device: {device}\")\n",
            print(f\"🎯 Mode: Evaluation (dropout/batchnorm frozen)\")\n",
            print()\n",

```

**Step 2: Run script to update Cell 34**

```bash
python3 scripts/update_cell_34.py
```

**Step 3: Commit weight loading section**

```bash
git add scripts/update_cell_34.py
git commit -m "feat(cell-34): add weight loading and device migration

Loads model_state_dict with strict=True for architecture validation.
Uses map_location='cpu' to handle GPU checkpoints on CPU sessions.
Migrates model to GPU if available, sets eval() mode by default.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Implement Cell 34 - Information Display Section

**Files:**
- Modify: `scripts/update_cell_34.py` (extend source)

**Step 1: Extend CELL_34_SOURCE with comprehensive display**

Add to `CELL_34_SOURCE` after weight loading:

```python
            # Display checkpoint info\n",
            print(\"=\" * 70)\n",
            print(\"CHECKPOINT INFO\")\n",
            print(\"=\" * 70)\n",
            print()\n",
            \n",
            epoch = checkpoint.get('epoch', 'Unknown')\n",
            global_step = checkpoint.get('global_step', 'Unknown')\n",
            metrics = checkpoint.get('metrics', {})\n",
            timestamp = checkpoint.get('timestamp', 'Unknown')\n",
            \n",
            print(f\"📊 Training Progress:\")\n",
            print(f\"   Epoch: {epoch}\")\n",
            print(f\"   Global Step: {global_step}\")\n",
            print(f\"   Timestamp: {timestamp}\")\n",
            print()\n",
            \n",
            print(f\"📈 Metrics:\")\n",
            if 'train_loss' in metrics:\n",
            "    print(f\"   Train Loss: {metrics['train_loss']:.4f}\")\n",
            if 'val_loss' in metrics:\n",
            "    print(f\"   Val Loss: {metrics['val_loss']:.4f}\")\n",
            if 'learning_rate' in metrics:\n",
            "    print(f\"   Learning Rate: {metrics['learning_rate']:.6f}\")\n",
            print()\n",
            \n",
            # Display model info\n",
            print(\"=\" * 70)\n",
            print(\"MODEL INFO\")\n",
            print(\"=\" * 70)\n",
            print()\n",
            \n",
            # Count parameters\n",
            total_params = sum(p.numel() for p in model.parameters())\n",
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
            frozen_params = total_params - trainable_params\n",
            \n",
            print(f\"🧠 Parameters:\")\n",
            print(f\"   Total: {total_params:,}\")\n",
            print(f\"   Trainable: {trainable_params:,}\")\n",
            print(f\"   Frozen: {frozen_params:,}\")\n",
            print()\n",
            \n",
            print(f\"💾 Memory:\")\n",
            print(f\"   Model Size: {total_params * 4 / 1024**2:.2f} MB (FP32)\")\n",
            print()\n",
            \n",
            # Display architecture preview\n",
            print(\"=\" * 70)\n",
            print(\"ARCHITECTURE PREVIEW\")\n",
            print(\"=\" * 70)\n",
            print()\n",
            \n",
            print(\"📐 First 5 layers:\")\n",
            for i, (name, param) in enumerate(model.named_parameters()):\n",
            "    if i >= 5:\n",
            "        break\n",
            "    print(f\"   {i+1}. {name}\")\n",
            "    print(f\"      Shape: {list(param.shape)}\")\n",
            "    print(f\"      Device: {param.device}\")\n",
            "    print()\n",
            \n",
            total_layers = sum(1 for _ in model.named_parameters())\n",
            if total_layers > 5:\n",
            "    print(f\"   ... and {total_layers - 5} more layers\")\n",
            "    print()\n",
            \n",
            # Next steps guidance\n",
            print(\"=\" * 70)\n",
            print(\"NEXT STEPS\")\n",
            print(\"=\" * 70)\n",
            print()\n",
            print(\"💡 Model is ready for:\")\n",
            print(\"   → Inference on test data\")\n",
            print(\"   → Evaluation/benchmarking\")\n",
            print(\"   → Feature extraction\")\n",
            print()\n",
            print(\"🔄 To resume training:\")\n",
            print(\"   model.train()  # Switch to training mode\")\n",
            print(\"   # Then run training loop\")\n",
            print()\n",
            print(\"=\" * 70)\n",

```

**Step 2: Run script to update Cell 34**

```bash
python3 scripts/update_cell_34.py
```

**Step 3: Commit information display section**

```bash
git add scripts/update_cell_34.py
git commit -m "feat(cell-34): add comprehensive information display

Displays:
- Checkpoint metadata (epoch, step, timestamp)
- Training metrics (train_loss, val_loss, lr)
- Model summary (total/trainable/frozen params, memory)
- Architecture preview (first 5 layers with shapes/devices)
- Next steps guidance (inference vs training)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Implement Cell 34 - Error Handling Section

**Files:**
- Modify: `scripts/update_cell_34.py` (extend source)

**Step 1: Extend CELL_34_SOURCE with error handling**

Add error handling after the try block:

```python
        except RuntimeError as e:\n",
            if \"size mismatch\" in str(e) or \"Missing key\" in str(e) or \"Unexpected key\" in str(e):\n",
            "    print(\"❌ ERROR: Model architecture mismatch!\")\n",
            "    print()\n",
            "    print(\"💡 Possible causes:\")\n",
            "    print(\"   1. Model config in Cell 13 differs from training config\")\n",
            "    print(\"   2. Checkpoint is from a different model architecture\")\n",
            "    print(\"   3. Model definition was modified after training\")\n",
            "    print()\n",
            "    print(\"🔧 To fix:\")\n",
            "    print(\"   1. Verify config in Cell 13 matches training config\")\n",
            "    print(\"   2. Check model definition hasn't changed\")\n",
            "    print(\"   3. Try loading a different checkpoint\")\n",
            "    print()\n",
            "    print(f\"📋 Technical details: {e}\")\n",
            "    print()\n",
            "    print(\"=\" * 70)\n",
            else:\n",
            "    # Re-raise unexpected RuntimeError\n",
            "    raise\n",
        \n",
        except Exception as e:\n",
            print(f\"❌ ERROR: Failed to load checkpoint!\")\n",
            print()\n",
            print(\"💡 Possible causes:\")\n",
            print(\"   1. Checkpoint file is corrupted\")\n",
            print(\"   2. Incompatible PyTorch versions\")\n",
            print(\"   3. Insufficient memory\")\n",
            print()\n",
            print(\"🔧 To fix:\")\n",
            print(\"   1. Try loading a different checkpoint\")\n",
            print(\"   2. Verify checkpoint file is not corrupted\")\n",
            print(\"   3. Restart runtime and try again\")\n",
            print()\n",
            print(f\"📋 Technical details: {e}\")\n",
            print()\n",
            print(\"=\" * 70)\n",

```

**Step 2: Run script to generate final Cell 34**

```bash
python3 scripts/update_cell_34.py
```

Expected: Cell 34 now complete with all sections.

**Step 3: Verify complete Cell 34 structure**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
cell_34 = nb['cells'][34]
print(f'Cell 34 source lines: {len(cell_34[\"source\"])}')
print(f'Has execution_count: {\"execution_count\" in cell_34}')
print(f'Has outputs: {\"outputs\" in cell_34}')
print(f'Has cellView: {\"cellView\" in cell_34.get(\"metadata\", {})}')
"
```

Expected: All required fields present.

**Step 4: Commit complete Cell 34 implementation**

```bash
git add scripts/update_cell_34.py training.ipynb
git commit -m "feat(cell-34): add error handling for architecture mismatch and corruption

Graceful error handling with actionable guidance:
- RuntimeError (architecture mismatch) -> verify config match
- Generic Exception (corruption/OOM) -> try different checkpoint

Cell 34 implementation complete with:
✅ Prerequisites validation
✅ Checkpoint discovery & selection
✅ Weight loading & device migration
✅ Comprehensive information display
✅ Error handling with recovery guidance

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Fix Cell 34 (Now Cell 35) - Extract Session Variables

**Files:**
- Read: `training.ipynb` (Cell 35, previously Cell 34)
- Modify: `training.ipynb` (Cell 35 source)

**Step 1: Read current Cell 35 source**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
cell_35 = nb['cells'][35]
print('Cell 35 (Extract Variables):')
for i, line in enumerate(cell_35['source']):
    print(f'{i+1:2}: {line}', end='')
"
```

**Step 2: Investigate why no output appears**

Hypothesis: All variables already exist, so nested `if` conditions skip all print statements.

Create diagnostic script `scripts/debug_cell_35.py`:

```python
#!/usr/bin/env python3
"""Debug why Cell 35 doesn't produce output."""

import json

nb = json.load(open('training.ipynb'))
cell_35 = nb['cells'][35]

print("Cell 35 logic flow:")
print("1. if 'results' in globals() → checks for results dict")
print("2.   if 'workspace_root' not in globals() → only prints if missing")
print("3.   if 'run_name' not in globals() → only prints if missing")
print("4.   if 'metrics_df' not in globals() → only prints if missing")
print()
print("Issue: If all variables already exist, nothing prints!")
print("Solution: Always print variable values, not just when creating them.")
```

**Step 3: Run diagnostic**

```bash
python3 scripts/debug_cell_35.py
```

Expected: Confirms hypothesis - nested conditions prevent output.

**Step 4: Write fixed Cell 35 source**

Create `scripts/fix_cell_35.py`:

```python
#!/usr/bin/env python3
"""Fix Cell 35 to always show output."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')
CELL_INDEX = 35

# Fixed Cell 35 source - always prints status
FIXED_SOURCE = [
    "# @title 🔧 Extract Session Variables { display-mode: \"form\" }\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(\"EXTRACT SESSION VARIABLES\")\n",
    "print(\"=\" * 70)\n",
    "print()\n",
    "\n",
    "# Extract workspace_root, run_name, metrics_df from results dict\n",
    "# Works for both training (Cell 32) and recovery (Cell 33) workflows\n",
    "\n",
    "if 'results' in globals():\n",
    "    # Extract workspace_root (for file paths)\n",
    "    workspace_root = results.get('workspace_root', './workspace')\n",
    "    print(f\"✅ workspace_root: {workspace_root}\")\n",
    "    \n",
    "    # Extract run_name (for file naming)\n",
    "    run_name = results.get('run_name', 'training_run')\n",
    "    print(f\"✅ run_name: {run_name}\")\n",
    "    \n",
    "    # Extract metrics_df (for analysis)\n",
    "    metrics_df = results.get('metrics_summary')\n",
    "    if metrics_df is not None:\n",
    "        print(f\"✅ metrics_df: {len(metrics_df)} epochs\")\n",
    "    else:\n",
    "        print(\"⚠️  metrics_df: Not available\")\n",
    "    \n",
    "    print()\n",
    "    print(\"💡 Variables extracted and ready to use\")\n",
    "    \n",
    "else:\n",
    "    print(\"⚠️  'results' not found\")\n",
    "    print()\n",
    "    print(\"💡 To fix:\")\n",
    "    print(\"   1. Run Cell 32 (Training) OR Cell 33 (Recovery) first\")\n",
    "    print(\"   2. Re-run this cell\")\n",
    "\n",
    "print()\n",
    "print(\"=\" * 70)\n",
]

# Update Cell 35
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

nb['cells'][CELL_INDEX]['source'] = FIXED_SOURCE

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✅ Fixed Cell {CELL_INDEX}")
print(f"   Now always prints extraction status")
```

**Step 5: Run fix script**

```bash
python3 scripts/fix_cell_35.py
```

Expected output:
```
✅ Fixed Cell 35
   Now always prints extraction status
```

**Step 6: Commit Cell 35 fix**

```bash
git add training.ipynb scripts/debug_cell_35.py scripts/fix_cell_35.py
git commit -m "fix(cell-35): always print variable extraction status

Previous issue: Cell only printed when variables were missing.
If all variables already existed, cell executed silently (no output).

Fix: Always print extraction status for workspace_root, run_name, metrics_df.

User can now verify:
- Variables were extracted successfully
- Values are correct
- Cell actually executed

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md` (add Cell 34 documentation)

**Step 1: Add Cell 34 section to CLAUDE.md**

Find the section documenting notebook cells, add after Cell 33:

```markdown
#### Checkpoint Recovery (v4.0+)

**NEW**: Recover training results from saved checkpoints for interrupted training, analysis, or resume workflows.

**Quick Example** (in training.ipynb):
```python
from utils.training.engine.recovery import recover_training_results

# Recover from best checkpoint
results = recover_training_results(checkpoint_dir='./checkpoints')

# Use exactly like Trainer.train() return value
print(f"Train Loss: {results['loss_history'][-1]:.4f}")
print(f"Val Loss: {results['val_loss_history'][-1]:.4f}")
```

**Recovery Cell** (training.ipynb Cell 33):
- Lists all checkpoints with metrics
- Recovers best checkpoint automatically
- Logs to ExperimentDB
- Provides results in same format as `Trainer.train()`

**Load Model Weights Cell** (training.ipynb Cell 34):
- Loads `model_state_dict` from checkpoint into model
- Displays available checkpoints with metrics
- Defaults to Cell 33's checkpoint selection
- Migrates model to GPU if available
- Sets model to eval() mode (ready for inference)
- Shows comprehensive info (checkpoint metadata, model summary, architecture)

**Variable Extraction Cell** (training.ipynb Cell 35):
- Extracts `workspace_root`, `run_name`, `metrics_df` from results dict
- Works for both training and recovery workflows
- Always prints extraction status
```

**Step 2: Commit documentation update**

```bash
git add CLAUDE.md
git commit -m "docs: document Cell 34 (Load Model Weights) in CLAUDE.md

Added documentation for new Cell 34 that loads model weights from
checkpoints, completing the recovery workflow (metrics + weights).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Integration Testing

**Files:**
- Create: `scripts/test_cell_34_integration.py`

**Step 1: Write integration test script**

Create `scripts/test_cell_34_integration.py`:

```python
#!/usr/bin/env python3
"""Integration test for Cell 34 (Load Model Weights)."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

def test_cell_structure():
    """Verify Cell 34 has correct Jupyter structure."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_34 = nb['cells'][34]

    # Check required fields
    assert cell_34['cell_type'] == 'code', "Cell 34 must be code cell"
    assert 'execution_count' in cell_34, "Missing execution_count"
    assert 'outputs' in cell_34, "Missing outputs"
    assert 'metadata' in cell_34, "Missing metadata"
    assert cell_34['metadata'].get('cellView') == 'form', "Must be form cell"

    print("✅ Cell 34 structure valid")

def test_cell_content():
    """Verify Cell 34 has all required sections."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_34_source = ''.join(nb['cells'][34]['source'])

    # Check required sections
    required = [
        "LOAD MODEL WEIGHTS FROM CHECKPOINT",
        "if 'model' not in globals():",
        "list_checkpoints",
        "checkpoint_path",
        "torch.load",
        "load_state_dict",
        "model.eval()",
        "CHECKPOINT INFO",
        "MODEL INFO",
        "ARCHITECTURE PREVIEW",
        "NEXT STEPS",
        "RuntimeError",
    ]

    for keyword in required:
        assert keyword in cell_34_source, f"Missing required keyword: {keyword}"

    print("✅ Cell 34 content complete")

def test_cell_order():
    """Verify cells are in correct order."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Extract titles
    titles = []
    for i in [33, 34, 35]:
        title = nb['cells'][i]['source'][0]
        titles.append((i, title[:60]))

    # Verify order
    assert "Recover Training Results" in titles[0][1], "Cell 33 should be Recovery"
    assert "Load Model Weights" in titles[1][1], "Cell 34 should be Load Weights"
    assert "Extract Session Variables" in titles[2][1], "Cell 35 should be Extract"

    print("✅ Cell order correct")
    for i, title in titles:
        print(f"   Cell {i}: {title}")

if __name__ == "__main__":
    test_cell_structure()
    test_cell_content()
    test_cell_order()
    print()
    print("🎉 All integration tests passed!")
```

**Step 2: Run integration tests**

```bash
python3 scripts/test_cell_34_integration.py
```

Expected output:
```
✅ Cell 34 structure valid
✅ Cell 34 content complete
✅ Cell order correct
   Cell 33: Recover Training Results from Checkpoint { display-mode:
   Cell 34: Load Model Weights from Checkpoint { display-mode: "form
   Cell 35: Extract Session Variables { display-mode: "form" }

🎉 All integration tests passed!
```

**Step 3: Commit integration tests**

```bash
git add scripts/test_cell_34_integration.py
git commit -m "test: add integration tests for Cell 34

Validates:
- Jupyter cell structure (execution_count, outputs, metadata)
- Cell content completeness (all required sections)
- Cell order (Cell 33 → 34 → 35)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Cleanup and Final Verification

**Files:**
- Delete: `scripts/insert_cell_34.py`, `scripts/update_cell_34.py`, `scripts/debug_cell_35.py`, `scripts/fix_cell_35.py`
- Keep: `scripts/test_cell_34_integration.py` (for future validation)

**Step 1: Run final verification**

```bash
python3 -c "
import json
nb = json.load(open('training.ipynb'))
print('Total cells:', len(nb['cells']))
print()
print('Cell 33:', nb['cells'][33]['source'][0][:60])
print('Cell 34:', nb['cells'][34]['source'][0][:60])
print('Cell 35:', nb['cells'][35]['source'][0][:60])
print()
print('Cell 34 source lines:', len(nb['cells'][34]['source']))
"
```

Expected: Shows complete notebook structure.

**Step 2: Remove temporary scripts**

```bash
rm scripts/insert_cell_34.py scripts/update_cell_34.py scripts/debug_cell_35.py scripts/fix_cell_35.py
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: remove temporary implementation scripts

Kept scripts/test_cell_34_integration.py for future validation.
Removed one-time-use helper scripts.

Implementation complete:
✅ Cell 34: Load Model Weights from Checkpoint
✅ Cell 35: Extract Session Variables (fixed)
✅ Documentation updated (CLAUDE.md)
✅ Integration tests passing

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Success Criteria Checklist

- [ ] Cell 34 inserted at correct position (between Cell 33 and current Cell 34)
- [ ] Cell 34 has proper Jupyter structure (execution_count, outputs, metadata)
- [ ] Prerequisites validation implemented (model exists check)
- [ ] Checkpoint discovery using `list_checkpoints()` utility
- [ ] Selection priority logic (results → auto-select → manual override)
- [ ] Weight loading with strict=True and device migration
- [ ] Model set to eval() mode by default
- [ ] Comprehensive display (checkpoint info, model summary, architecture)
- [ ] Error handling (architecture mismatch, corruption)
- [ ] Cell 35 (Extract Variables) fixed to always print output
- [ ] CLAUDE.md documentation updated
- [ ] Integration tests passing
- [ ] Temporary scripts cleaned up

---

**Implementation Complete!**

This plan provides step-by-step instructions for implementing Cell 34 (Load Model Weights) and fixing Cell 35 (Extract Session Variables). Each task is broken into bite-sized steps (2-5 minutes) with clear validation and frequent commits.
