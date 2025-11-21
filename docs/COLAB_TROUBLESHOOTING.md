# Google Colab Troubleshooting Guide

**Version:** 1.0
**Last Updated:** 2025-11-20
**Applies To:** `template.ipynb` and `training.ipynb`

---

## Table of Contents

1. [Common Issues - Template.ipynb](#common-issues---templateipynb)
2. [Common Issues - Training.ipynb](#common-issues---trainingipynb)
3. [Dependency and Installation Issues](#dependency-and-installation-issues)
4. [Model Loading Issues](#model-loading-issues)
5. [Memory and Performance Issues](#memory-and-performance-issues)
6. [Network and Connectivity Issues](#network-and-connectivity-issues)
7. [GPU and Runtime Issues](#gpu-and-runtime-issues)
8. [Test Failures](#test-failures)
9. [Export and Checkpoint Issues](#export-and-checkpoint-issues)
10. [Best Practices](#best-practices)

---

## Common Issues - Template.ipynb

### Issue: NumPy Corrupted Error

**Symptoms:**
```
ImportError: cannot import name '_center' from 'numpy._core.umath'
```

**Cause:** Installing packages that reinstall NumPy with incompatible versions corrupts Colab's pre-installed NumPy

**Solution:**
1. **Runtime → Restart runtime** (top menu)
2. **Edit → Clear all outputs** (optional, for clean UI)
3. Run all cells again from the beginning
4. **DO NOT** run any pip install commands in template.ipynb

**Prevention:**
- Template.ipynb uses zero-installation strategy (no pip installs)
- All Tier 1/2 tests use only Colab pre-installed packages
- For Tier 3 (training), use separate `training.ipynb` in fresh runtime

---

### Issue: Gist ID Not Found

**Symptoms:**
```
⚠️ NO GIST ID PROVIDED
Please paste your Gist ID in the field above and re-run this cell
```

**Cause:** Cell 5 (Gist ID input) was skipped or not executed

**Solution:**
1. Scroll to Cell 5 ("📥 Paste Your Gist ID Here")
2. Paste your Gist ID from Transformer Builder export
3. Run Cell 5
4. Continue with subsequent cells

**How to Get Gist ID:**
1. Go to [Transformer Builder](https://transformer-builder.com)
2. Click "Export to Colab"
3. Copy the Gist ID from the modal (alphanumeric string like `abc123def456`)
4. Paste into Cell 5

---

### Issue: Model Loading Fails (HTTP 404)

**Symptoms:**
```
❌ Failed to load model from Gist!
Error: GitHub API error: HTTP 404 - Gist not found
```

**Causes:**
- Incorrect Gist ID (typo)
- Gist was deleted or made private
- Gist not fully created by Transformer Builder

**Solutions:**
1. **Verify Gist ID:** Check for typos (should be alphanumeric, no special characters)
2. **Check Gist URL:** Visit `https://gist.github.com/YOUR_GIST_ID` in browser
   - If 404: Gist doesn't exist, re-export from Transformer Builder
   - If private: Make gist public or re-export
3. **Re-export from Transformer Builder:**
   - Go back to Transformer Builder
   - Click "Export to Colab" again
   - Use new Gist ID

---

### Issue: Model Loading Fails (Missing Files)

**Symptoms:**
```
❌ Gist is missing 'model.py' - please re-export from Transformer Builder
❌ Gist is missing 'config.json' - please re-export from Transformer Builder
```

**Cause:** Incomplete Gist export from Transformer Builder

**Solution:**
1. Go back to Transformer Builder
2. Re-export your model (click "Export to Colab")
3. Verify the Gist contains both `model.py` and `config.json`
4. Use the new Gist ID

---

### Issue: GitHub API Rate Limit

**Symptoms:**
```
❌ GitHub API error: HTTP 429 - GitHub API rate limit
```

**Cause:** Exceeded GitHub's unauthenticated API limit (60 requests/hour)

**Solution:**
1. **Wait 1 hour** for rate limit to reset
2. Use fewer cells with Gist loading
3. Cache the model locally (advanced):
   ```python
   # After successful load, save locally
   !cp custom_transformer.py /content/drive/MyDrive/cached_model.py
   !cp config.json /content/drive/MyDrive/cached_config.json

   # Load from cache next time
   !cp /content/drive/MyDrive/cached_model.py custom_transformer.py
   !cp /content/drive/MyDrive/cached_config.json config.json
   ```

---

### Issue: Tier 1 Tests Fail (Shape Mismatch)

**Symptoms:**
```
RuntimeError: Expected input shape (batch, seq_len), got (batch, seq_len, vocab_size)
```

**Cause:** Model expects different input format than standard

**Solution:**
1. Check model's `forward()` signature in Cell 11 (model code display)
2. Verify `vocab_size` in config matches model embedding layer
3. If custom input format, tests may need manual adjustment

**Workaround:**
- Skip Tier 1 tests and use model for inference only
- Report issue to Transformer Builder team

---

## Common Issues - Training.ipynb

### Issue: Dependency Installation Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
ERROR: Cannot install pytorch-lightning==2.5.6 because these package versions have incompatible dependencies
```

**Cause:** Version conflicts between requirements-training.txt and Colab pre-installed packages

**Solution:**
1. **Restart runtime first:** Runtime → Restart runtime
2. Install in fresh runtime (before running any other cells)
3. Try relaxed version pins:
   ```bash
   !pip install pytorch-lightning>=2.4.0 optuna>=4.0.0 torchmetrics>=1.3.0 wandb>=0.15.0
   ```
4. If still fails, install packages individually:
   ```bash
   !pip install pytorch-lightning
   !pip install optuna
   !pip install torchmetrics
   !pip install wandb
   ```

**Prevention:**
- Always use fresh runtime for training.ipynb
- Don't mix template.ipynb and training.ipynb in same session

---

### Issue: Training OOM (Out of Memory)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause:** Model or batch size too large for Colab GPU (T4: 15GB)

**Solutions:**

**1. Reduce Batch Size:**
```python
config = TrainingConfig(
    batch_size=2,  # Reduce from 4 or 8
    gradient_accumulation_steps=4  # Maintain effective batch size
)
```

**2. Enable Gradient Checkpointing:**
```python
# In model definition (if supported)
model.gradient_checkpointing_enable()
```

**3. Use Smaller Model:**
- Reduce number of layers
- Reduce hidden dimensions
- Test with tiny config first

**4. Clear GPU Cache:**
```python
import torch
torch.cuda.empty_cache()
```

**5. Monitor Memory:**
```python
# Before training
torch.cuda.reset_peak_memory_stats()

# After training
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak memory: {peak_mb:.1f} MB")
```

---

### Issue: Training Too Slow (Session Timeout)

**Symptoms:**
- Training doesn't complete within 12-hour Colab session
- Colab disconnects before training finishes

**Solutions:**

**1. Reduce Epochs:**
```python
config = TrainingConfig(
    epochs=3  # Start small, increase if time allows
)
```

**2. Use Faster Compilation Mode:**
```python
config = TrainingConfig(
    compile_mode="reduce-overhead",  # Faster than "max-autotune"
    deterministic=False  # Fast mode (20% speedup)
)
```

**3. Save Frequent Checkpoints:**
```python
# In training loop
if epoch % 2 == 0:  # Every 2 epochs
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
```

**4. Use Colab Pro:**
- Longer session timeouts (24 hours)
- Faster GPUs (V100, A100)
- Background execution

---

### Issue: W&B Login Required

**Symptoms:**
```
wandb: ERROR Please run `wandb login` to get your API key
```

**Solutions:**

**1. Login with API Key:**
```python
import wandb
wandb.login(key='YOUR_API_KEY')  # Get from https://wandb.ai/authorize
```

**2. Use Offline Mode:**
```python
import os
os.environ['WANDB_MODE'] = 'offline'

# Training will save logs locally
# Sync later: !wandb sync wandb/offline-run-*
```

**3. Disable W&B:**
```python
config = TrainingConfig(
    use_wandb=False  # Disable W&B entirely
)
```

---

### Issue: Checkpoint Save/Resume Fails

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/best.pt'
```

**Solutions:**

**1. Create Checkpoint Directory:**
```python
import os
os.makedirs('checkpoints', exist_ok=True)

# Then save
torch.save(model.state_dict(), 'checkpoints/best.pt')
```

**2. Use Google Drive for Persistence:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
torch.save(model.state_dict(), '/content/drive/MyDrive/checkpoints/best.pt')
```

**3. Verify Save Path:**
```python
import os
save_path = 'checkpoints/best.pt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
print(f"Saving to: {os.path.abspath(save_path)}")
torch.save(model.state_dict(), save_path)
```

---

## Dependency and Installation Issues

### Issue: pip install Takes Too Long

**Symptoms:** Installation hangs or takes more than 5 minutes

**Solutions:**
1. **Check Internet Connection:** Colab → Runtime → View runtime logs
2. **Use Faster Mirrors:**
   ```bash
   !pip install --index-url https://pypi.org/simple pytorch-lightning
   ```
3. **Install Without Dependencies (if safe):**
   ```bash
   !pip install --no-deps pytorch-lightning
   ```
4. **Clear pip Cache:**
   ```bash
   !rm -rf ~/.cache/pip
   !pip install pytorch-lightning
   ```

---

### Issue: Package Not Found

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement package-name
```

**Solutions:**
1. **Check Package Name Spelling**
2. **Update pip:**
   ```bash
   !pip install --upgrade pip
   ```
3. **Install from GitHub (latest):**
   ```bash
   !pip install git+https://github.com/owner/repo.git
   ```

---

## Model Loading Issues

### Issue: Model Code Has Syntax Errors

**Symptoms:**
```
SyntaxError: invalid syntax
```

**Cause:** Corrupted export from Transformer Builder

**Solutions:**
1. **View Model Code:** Check Cell 11 (model code display)
2. **Re-export from Transformer Builder**
3. **Manual Fix (advanced):**
   - Edit `custom_transformer.py` directly in Colab
   - Fix syntax errors
   - Re-run model instantiation cell

---

### Issue: Model Instantiation Fails

**Symptoms:**
```
TypeError: __init__() missing 1 required positional argument: 'config'
RuntimeError: Expected tensor for argument #1 'self' to have one of ...
```

**Causes:**
- Model expects config dict but receives SimpleNamespace
- Missing required parameters

**Solutions:**

**1. Check Constructor Signature:**
```python
import inspect
sig = inspect.signature(model_class.__init__)
print(sig)
```

**2. Try Different Instantiation Methods:**
```python
# Method 1: Parameterless (Transformer Builder default)
model = model_class()

# Method 2: With config dict
model = model_class(**config_dict)

# Method 3: With config object
from types import SimpleNamespace
config_obj = SimpleNamespace(**config_dict)
model = model_class(config_obj)
```

---

## Memory and Performance Issues

### Issue: Colab Disconnects During Long Operations

**Symptoms:** Browser tab shows "Reconnecting..." or "Runtime disconnected"

**Causes:**
- Inactive browser tab (Colab assumes you left)
- Session timeout
- Resource limits exceeded

**Solutions:**

**1. Keep Browser Tab Active:**
```javascript
// Run in browser console (F12)
function ClickConnect() {
    console.log("Clicking 'Connect' button");
    document.querySelector("colab-connect-button").shadowRoot.querySelector("#connect").click();
}
setInterval(ClickConnect, 60000);  // Click every 60 seconds
```

**2. Mount Google Drive (Saves Progress):**
```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
checkpoint_dir = '/content/drive/MyDrive/colab_checkpoints'
```

**3. Use Background Execution (Colab Pro):**
- Colab Pro allows background execution
- Sessions continue even if browser closes

---

### Issue: Inference Too Slow

**Symptoms:** Model prediction takes several seconds per batch

**Solutions:**

**1. Use GPU:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)
```

**2. Use Mixed Precision:**
```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)
```

**3. Batch Inputs:**
```python
# Instead of processing one at a time
for text in texts:
    output = model(text)

# Process in batches
batch_size = 8
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    outputs = model(batch)
```

**4. Use torch.compile (PyTorch 2.0+):**
```python
model = torch.compile(model, mode='reduce-overhead')
```

---

## Network and Connectivity Issues

### Issue: Cannot Download utils Package from GitHub

**Symptoms:**
```
fatal: unable to access 'https://github.com/...': Could not resolve host
```

**Solutions:**

**1. Retry with Timeout:**
```bash
!git clone --depth 1 https://github.com/matt-hans/transformer-builder-colab-templates.git --timeout=60
```

**2. Use Raw GitHub URL (fallback):**
```bash
!wget https://raw.githubusercontent.com/matt-hans/transformer-builder-colab-templates/main/utils/test_functions.py
```

**3. Check Colab Network Status:**
```python
import urllib.request
try:
    urllib.request.urlopen('https://www.google.com', timeout=5)
    print("✅ Network OK")
except:
    print("❌ Network issue - try restarting runtime")
```

---

### Issue: GitHub Gist API Slow or Timing Out

**Symptoms:**
```
TimeoutError: Request timed out after 20 seconds
```

**Solutions:**

**1. Increase Timeout:**
```python
# Cell 1 (network retry monkey-patch) already handles this
# Default timeout: 20s
# If still failing, manually increase in Cell 1:
def urlopen_with_retry(req, timeout=60):  # Increase to 60s
    return _retrying_urlopen(req, timeout=timeout)
```

**2. Check GitHub Status:**
- Visit https://www.githubstatus.com
- If degraded performance, wait and retry

**3. Use Cached Copy:**
```python
# Save model locally after first successful load
import json
with open('model_backup.py', 'w') as f:
    f.write(model_code)
with open('config_backup.json', 'w') as f:
    json.dump(config_dict, f)

# Load from backup next time
with open('model_backup.py') as f:
    exec(f.read())
```

---

## GPU and Runtime Issues

### Issue: GPU Not Available

**Symptoms:**
```
RuntimeError: No CUDA GPUs are available
torch.cuda.is_available() returns False
```

**Solutions:**

**1. Enable GPU Runtime:**
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save

**2. Verify GPU Allocation:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not allocated - check runtime type")
```

**3. GPU Quota Exhausted:**
- Free tier: Limited GPU hours per day
- Wait 12-24 hours or upgrade to Colab Pro

---

### Issue: Wrong GPU Type

**Symptoms:** Allocated K80 instead of T4 (free tier should get T4)

**Solutions:**
1. **Runtime → Disconnect and delete runtime**
2. **Runtime → Run all** (reconnect and get new GPU)
3. Repeat until you get T4 (modern GPU)
4. Use Colab Pro for guaranteed V100/A100

---

## Test Failures

### Issue: Gradient Flow Test Fails (Vanishing Gradients)

**Symptoms:**
```
⚠️ Vanishing gradients detected (min gradient < 1e-6)
```

**Causes:**
- Poor weight initialization
- Deep network without residual connections
- Activation functions (sigmoid, tanh can saturate)

**Solutions:**

**1. Check Initialization:**
```python
# In model definition
nn.init.xavier_uniform_(self.linear.weight)
nn.init.zeros_(self.linear.bias)
```

**2. Add Residual Connections:**
```python
class Block(nn.Module):
    def forward(self, x):
        return x + self.layer(x)  # Residual connection
```

**3. Use Better Activations:**
```python
# Instead of sigmoid/tanh
self.activation = nn.GELU()  # or nn.ReLU()
```

**4. Reduce Network Depth:**
- Fewer layers for testing
- Increase gradually

---

### Issue: Memory Footprint Test Shows Excessive Growth

**Symptoms:**
```
⚠️ Memory growth rate: 15.2 MB per token (expected < 1 MB)
```

**Causes:**
- Memory leak (activations not released)
- Gradient accumulation without proper clearing
- Caching in model

**Solutions:**

**1. Use torch.no_grad for Inference:**
```python
with torch.no_grad():
    output = model(inputs)
```

**2. Clear Gradients:**
```python
model.zero_grad()
```

**3. Delete Intermediate Tensors:**
```python
loss.backward()
optimizer.step()
del loss  # Explicitly delete
torch.cuda.empty_cache()
```

---

## Export and Checkpoint Issues

### Issue: ONNX Export Fails

**Symptoms:**
```
RuntimeError: ONNX export failed: Unsupported operator 'aten::some_op'
```

**Solutions:**

**1. Use TorchScript Instead:**
```python
config = TrainingConfig(
    export_formats=["torchscript"],  # Skip ONNX
    export_bundle=True
)
```

**2. Simplify Model for ONNX:**
- Avoid dynamic control flow (if/else in forward())
- Use static shapes
- Avoid unsupported ops

**3. Export with Verbose Logging:**
```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    verbose=True,  # See detailed error
    opset_version=14  # Try different opset
)
```

---

### Issue: TorchScript Export Fails

**Symptoms:**
```
RuntimeError: Could not trace module: Tracing failed
```

**Solutions:**

**1. Use Scripting Instead of Tracing:**
```python
# Instead of torch.jit.trace
scripted_model = torch.jit.script(model)
```

**2. Add Type Annotations:**
```python
from typing import Tuple

def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return output, hidden
```

**3. Skip Export:**
```python
config = TrainingConfig(
    export_bundle=False  # Disable export
)
```

---

## Best Practices

### For Template.ipynb

1. **Always Start Fresh:**
   - Runtime → Restart runtime
   - Edit → Clear all outputs
   - Run all cells from beginning

2. **Never Install Packages:**
   - Template uses zero-installation strategy
   - All tests work with pre-installed packages
   - For training, switch to training.ipynb

3. **Verify Gist Before Loading:**
   - Visit `https://gist.github.com/YOUR_GIST_ID`
   - Check model.py and config.json exist
   - Download and review code if suspicious

4. **Monitor Memory:**
   - Keep an eye on RAM usage (top right)
   - If approaching 12GB, restart runtime

### For Training.ipynb

1. **Use Fresh Runtime:**
   - Don't run training.ipynb in same session as template.ipynb
   - Restart runtime before starting training

2. **Start Small:**
   - Test with 1 epoch first
   - Verify training loop works
   - Scale up gradually (3 epochs → 10 epochs)

3. **Save Checkpoints to Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   checkpoint_dir = '/content/drive/MyDrive/checkpoints'
   ```

4. **Monitor GPU Memory:**
   ```python
   # Check available memory
   !nvidia-smi

   # Or in Python
   torch.cuda.memory_summary()
   ```

5. **Use W&B Offline Mode (if disconnected):**
   ```python
   os.environ['WANDB_MODE'] = 'offline'
   # Sync later: !wandb sync wandb/offline-run-*
   ```

### General Tips

1. **Keep Browser Tab Active:** Colab disconnects inactive tabs after 90 minutes

2. **Use Keyboard Shortcuts:**
   - `Ctrl+Enter`: Run current cell
   - `Shift+Enter`: Run current cell and move to next
   - `Ctrl+M B`: Insert cell below

3. **Check Runtime Logs:**
   - Runtime → View runtime logs
   - Shows detailed error messages and system info

4. **Export Important Results:**
   - Download checkpoints before session ends
   - Save plots/metrics to Drive
   - Sessions are ephemeral (12-hour limit)

5. **Report Bugs:**
   - If Transformer Builder export fails, report to team
   - If test functions fail unexpectedly, file GitHub issue
   - Include full error traceback and model configuration

---

## Getting Help

### Resources

1. **GitHub Issues:** https://github.com/matt-hans/transformer-builder-colab-templates/issues
2. **Transformer Builder Docs:** https://transformer-builder.com/docs
3. **Colab FAQ:** https://research.google.com/colaboratory/faq.html
4. **PyTorch Forums:** https://discuss.pytorch.org

### Before Asking for Help

**Include the following:**

1. **Error Message (full traceback):**
   ```python
   import traceback
   traceback.print_exc()
   ```

2. **Environment Info:**
   ```python
   !python --version
   !pip list | grep -E 'torch|numpy|pandas'
   !nvidia-smi
   ```

3. **Minimal Reproducible Example:**
   - Simplify to smallest code that reproduces issue
   - Include Gist ID or model config
   - Specify which cell failed

4. **What You've Tried:**
   - List solutions from this guide you've attempted
   - Include any workarounds that partially worked

---

**Document Version:** 1.0
**Last Updated:** 2025-11-20
**Maintainer:** Transformer Builder Team
