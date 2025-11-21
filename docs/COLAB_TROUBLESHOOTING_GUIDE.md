# Colab Notebook Troubleshooting Guide

**Last Updated:** 2025-11-20
**Notebooks:** `template.ipynb` and `training.ipynb`

This guide addresses common issues encountered when running the Colab notebooks and provides step-by-step solutions.

---

## Quick Diagnostic Checklist

Before diving into specific issues, check these basics:

- [ ] Running in Google Colab? (works on colab.research.google.com)
- [ ] Fresh runtime? (Click **Runtime → Restart runtime** first)
- [ ] Pasted Gist ID correctly? (Cell 3 or Cell 11)
- [ ] Gist is public? (Private Gists won't load)
- [ ] Internet connection stable? (Stable for GitHub API calls)

---

## Section 1: Dependency Issues

### Issue 1.1: "No module named 'utils'"

**Symptom:**
```
ModuleNotFoundError: No module named 'utils'
```

**Cause:** Utils package didn't download properly from GitHub.

**Solution:**

1. **Try running Cell 7 again:**
   ```python
   !rm -rf utils/
   !git clone --depth 1 --branch main \
       https://github.com/matt-hans/transformer-builder-colab-templates.git temp_repo
   !cp -r temp_repo/utils ./
   !rm -rf temp_repo
   ```

2. **If still failing, use fallback:**
   ```python
   !wget -q https://raw.githubusercontent.com/matt-hans/transformer-builder-colab-templates/main/utils/test_functions.py
   ```

3. **Last resort - manually install:**
   ```python
   !pip install transformer-builder-colab-templates
   ```

**Prevention:** Check your internet connection before running the notebook.

---

### Issue 1.2: "ModuleNotFoundError: No module named 'numpy._core.umath'"

**Symptom:**
```
ModuleNotFoundError: No module named 'numpy._core.umath'
NumPy corrupted - please restart runtime
```

**Cause:** NumPy C extensions are corrupted (usually from reinstalling numpy).

**Solution:**

1. **Click in Colab menu:**
   - Go to **Runtime** → **Restart runtime**
   - Wait 5 seconds for runtime to reset

2. **Clear all outputs:**
   - Go to **Edit** → **Clear all outputs**

3. **Run cells again from the top:**
   - Click on Cell 1
   - Click **Runtime** → **Run all**

**Prevention:**
- Never install numpy versions in `template.ipynb`
- The zero-installation strategy prevents this
- Always restart runtime before re-running

**Why this happens:**
- Colab has numpy pre-installed
- Installing different versions can overwrite C extensions
- Extensions are needed for performance-critical operations

---

### Issue 1.3: "ImportError: cannot import name 'test_shape_robustness'"

**Symptom:**
```
ImportError: cannot import name 'test_shape_robustness' from 'utils.test_functions'
```

**Cause:** Utils package structure is incomplete or outdated.

**Solution:**

1. **Force reload utils package:**
   ```python
   import sys
   if 'utils' in sys.modules:
       del sys.modules['utils']

   # Then re-import
   from utils.test_functions import test_shape_robustness
   ```

2. **Or re-download from source:**
   ```python
   !rm -rf utils/
   # Run Cell 7 again
   ```

**Prevention:** Use `git clone --depth 1` to get the latest version.

---

### Issue 1.4: "PyTorchLightning not found" (training.ipynb)

**Symptom:**
```
ModuleNotFoundError: No module named 'pytorch_lightning'
```

**Cause:** Cell 3 wasn't run to install training dependencies.

**Solution:**

1. **Run Cell 3:**
   ```python
   !pip install -r requirements-training.txt
   ```

2. **Wait for installation to complete** (usually 2-3 minutes)

3. **If still failing:**
   ```python
   !pip install --upgrade pytorch-lightning>=2.4.0
   !pip install optuna>=3.0.0
   !pip install wandb>=0.15.0
   !pip install torchmetrics>=1.3.0
   ```

**Prevention:** Always run Cell 3 before training cells in fresh runtime.

---

## Section 2: Gist Loading Issues

### Issue 2.1: "ValueError: Gist ID is required"

**Symptom:**
```
ValueError: Gist ID is required to load your custom model
```

**Cause:** You skipped Cell 5 (template) or Cell 11 (training) that asks for Gist ID.

**Solution:**

1. **Find your Gist ID:**
   - Go to [Transformer Builder](https://transformer-builder.com)
   - Click **Export to Colab**
   - Copy the Gist ID from the modal (looks like: `abc123def456`)

2. **Paste it in the notebook:**
   - Scroll to Cell 5 (template) or Cell 11 (training)
   - Find the field: `GIST_ID = ""`
   - Replace `""` with your ID: `GIST_ID = "abc123def456"`
   - Click the play button to run the cell

3. **Continue with next cells**

**What is a Gist ID?**
- A unique identifier for your exported model
- Found in the URL: `https://gist.github.com/username/abc123def456`
- The ID is: `abc123def456`

---

### Issue 2.2: "ValueError: Invalid Gist ID format"

**Symptom:**
```
ValueError: Invalid Gist ID format
The Gist ID you entered: 'invalid@gist#id'
Gist IDs should be alphanumeric (e.g., 'abc123def456')
```

**Cause:** Gist ID contains special characters or spaces.

**Solution:**

1. **Check the format:**
   - Valid: `a1b2c3d4e5f6` (letters and numbers only)
   - Invalid: `abc-123` (contains dash)
   - Invalid: `abc123 def456` (contains space)

2. **Copy from Transformer Builder again:**
   - Go back to Transformer Builder
   - Click **Export to Colab**
   - Copy the ID **exactly as shown** (no extra spaces)

3. **Paste and re-run the cell:**
   - Clear the field completely
   - Paste the ID
   - Run the cell

**Prevention:** Always copy Gist IDs directly from the official export flow.

---

### Issue 2.3: "HTTP 404: Gist not found"

**Symptom:**
```
RuntimeError: GitHub API error: HTTP 404 - Gist not found (check your Gist ID)
```

**Cause:**
1. Gist ID is incorrect
2. Gist has been deleted
3. Gist is private (not public)

**Solution:**

1. **Verify the Gist exists:**
   ```
   https://gist.github.com/{your-username}/{gist-id}
   ```
   - Replace `{your-username}` and `{gist-id}` with your values
   - Open in browser
   - You should see your `model.py` and `config.json` files

2. **Check the Gist is public:**
   - In GitHub, Gists are public by default
   - If private: Click **File** → **Make public** in Gist editor
   - Wait 30 seconds for GitHub to propagate

3. **Re-export from Transformer Builder:**
   - Go to Transformer Builder
   - Click **Export to Colab**
   - Get a fresh Gist ID
   - Use that ID instead

4. **Re-run the notebook with new ID:**
   - Update Cell 5 or Cell 11 with the new Gist ID
   - Run the cell again

---

### Issue 2.4: "HTTP 429: GitHub API rate limit exceeded"

**Symptom:**
```
RuntimeError: GitHub API error: HTTP 429 - GitHub API rate limit (try again in an hour)
```

**Cause:** Too many GitHub API requests in the last hour.

**Solution (Automatic):**
- The notebook has **automatic retry logic** with backoff
- You'll see: `⏳ Network retry 1/5 in 2.3s (HTTP 429)`
- The notebook will **automatically retry** up to 5 times
- Wait 30-60 seconds and let it retry

**If automatic retry exhausted:**

1. **Wait 1 hour:**
   - GitHub rate limit resets every hour
   - Come back after 1 hour and try again

2. **Or authenticate with GitHub:**
   ```python
   # In Cell 8, modify the request:
   req = urllib.request.Request(
       url,
       headers={
           "Accept": "application/vnd.github+json",
           "Authorization": "token YOUR_GITHUB_TOKEN"  # Add this
       }
   )
   ```
   - Get a token: https://github.com/settings/tokens
   - Select "gist" scope only
   - Replace `YOUR_GITHUB_TOKEN` with your token

**Prevention:**
- Don't run the notebook multiple times in quick succession
- Share notebooks with friends instead of having them export separately
- Use GitHub authentication for production/shared workflows

---

## Section 3: Model Loading Issues

### Issue 3.1: "SyntaxError: invalid syntax" in model.py

**Symptom:**
```
SyntaxError: invalid syntax
  File "custom_transformer.py", line 42, in <module>
    def forward(x)  # Missing colon
    ^
```

**Cause:** The Transformer Builder export generated invalid Python code.

**Solution:**

1. **Check the model code:**
   - Cell 11 displays the loaded `model.py`
   - Look for syntax errors (missing colons, unclosed parentheses, etc.)

2. **Report to Transformer Builder:**
   - This is a bug in the export
   - Open an issue on the Transformer Builder GitHub
   - Include the model.py code (from Cell 11)

3. **Workaround:**
   - Fix the code manually if possible
   - Or re-export the model in Transformer Builder

**Prevention:** This is rare; report if it happens.

---

### Issue 3.2: "RuntimeError: Could not find model class"

**Symptom:**
```
RuntimeError: Could not find model class 'CustomModel' in generated code
```

**Cause:**
1. Model.py doesn't define a class
2. Class name doesn't match config.json
3. Transformer Builder export is incomplete

**Solution:**

1. **Check the model code (Cell 11):**
   - Look for a line like: `class CustomModel(nn.Module):`
   - The class name should match `config['model_name']`

2. **If class name mismatches:**
   - The notebook will warn: "⚠️ Using CustomTransformer (expected CustomModel)"
   - This is **usually OK** - the notebook adapts automatically
   - If it fails anyway, your model may have an incompatible constructor

3. **Check the constructor:**
   - Transformer Builder models should have: `def __init__(self):`
   - With no required parameters
   - If your model needs parameters, pass them via `config`

**Prevention:** Keep model architecture simple and parameterless in Transformer Builder.

---

### Issue 3.3: "OutOfMemoryError: CUDA out of memory"

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause:** Model is too large for available GPU memory.

**Solution:**

1. **Check GPU memory:**
   ```python
   import torch
   print(torch.cuda.get_device_name(0))
   print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   ```

2. **Reduce batch size in tests:**
   ```python
   # In test calls, use smaller config:
   config.max_batch_size = 2  # Instead of 8

   test_shape_robustness(model, config)
   ```

3. **For training (training.ipynb):**
   ```python
   training_config = TrainingConfig(
       batch_size=2,  # Reduce from 4-8
       max_seq_len=128,  # Reduce from 512
       learning_rate=5e-5
   )
   ```

4. **Or upgrade to Colab Pro:**
   - Free tier: Tesla T4 (16 GB VRAM)
   - Pro tier: A100 (40 GB VRAM)

**Prevention:**
- Prototype with small models first
- Use smaller batch sizes in free Colab
- Colab Pro for larger models

---

## Section 4: Training Issues (training.ipynb)

### Issue 4.1: "Training loss stays constant"

**Symptom:**
```
Epoch 1: Loss = 4.6159
Epoch 2: Loss = 4.6157
Epoch 3: Loss = 4.6153
(no significant change)
```

**Cause:**
1. Learning rate too low
2. Model not receiving gradients
3. Training data is random/incompatible

**Solution:**

1. **Increase learning rate:**
   ```python
   training_config = TrainingConfig(
       learning_rate=1e-4,  # Increase from 5e-5
   )
   ```

2. **Check gradients are flowing:**
   ```python
   grad_results = test_gradient_flow(model, config)
   print(grad_results)
   ```

3. **Use real data:**
   - Generated random data won't have patterns to learn
   - Use Hugging Face datasets or real data

4. **Run longer:**
   - Give training at least 10 epochs
   - Check at epoch 10, not epoch 1

**Prevention:**
- Start with default learning rates (5e-5)
- Use real datasets
- Run enough epochs to see convergence

---

### Issue 4.2: "W&B login required"

**Symptom:**
```
wandb: You can find your API key in Settings at https://wandb.ai/settings
wandb: Paste an API token here: █
```

**Cause:** W&B integration is enabled but not authenticated.

**Solution:**

**Option 1: Skip W&B (easiest):**
```python
training_config = TrainingConfig(
    learning_rate=5e-5,
    use_wandb=False,  # Add this
)
```

**Option 2: Enable W&B (optional):**
1. Go to https://wandb.ai/settings
2. Copy your API key
3. Paste it when prompted
4. Training will now log to W&B dashboard

**Prevention:** W&B is optional; training works fine without it.

---

## Section 5: Performance Issues

### Issue 5.1: "GPU not available"

**Symptom:**
```
Device: cpu
(training is very slow)
```

**Cause:** GPU wasn't enabled in Colab runtime.

**Solution:**

1. **Enable GPU in Colab:**
   - Click **Runtime** → **Change runtime type**
   - Set **Hardware accelerator** to "GPU"
   - Click **Save**

2. **Verify GPU is available:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print: True
   print(torch.cuda.get_device_name(0))  # Tesla T4 or A100
   ```

3. **Restart runtime after enabling:**
   - Click **Runtime** → **Restart runtime**

**Prevention:** Always enable GPU before training.

---

### Issue 5.2: "Notebook disconnects after 12 hours"

**Symptom:**
```
Error: Connection lost
```

**Cause:** Free Colab has 12-hour session limit.

**Solution:**

1. **Use checkpointing:**
   ```python
   training_config = TrainingConfig(
       save_checkpoint_every=5,  # Save every 5 epochs
   )
   ```

2. **Or upgrade to Colab Pro** (unlimited sessions)

**Prevention:** For long training, use Colab Pro or split into multiple runs.

---

## Getting Help

### Gather this information before asking:

```markdown
**Problem:** [Brief description]

**Error Message:**
[Full traceback]

**Notebook:** template.ipynb or training.ipynb
**Cell:** [Number]
**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
...

**Environment:**
- GPU: [T4/A100/None]
- Colab: [Free/Pro]
```

### Where to ask

1. **GitHub:** https://github.com/matt-hans/transformer-builder-colab-templates/issues
2. **Transformer Builder:** https://transformer-builder.com/support
3. **Stack Overflow:** Tag: `google-colab`

---

## Diagnostic Commands

```python
# Check environment
import torch, numpy, sys
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Check utils
import utils
print(f"Utils: {utils.__version__}")
```

---

**Last Updated:** 2025-11-20
**Status:** Complete and ready for use
