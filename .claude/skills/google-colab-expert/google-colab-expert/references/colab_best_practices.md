# Google Colab Best Practices

Comprehensive guide to effective Google Colab workflows, environment management, and productivity patterns.

## Environment Setup

### 1. Initial Setup Cell Pattern

```python
# Cell 1: Environment Setup (run first)
import os
import sys
from pathlib import Path

# Check runtime type
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ No GPU - Enable in Runtime → Change runtime type")

# Install system dependencies
!apt -y install -qq aria2 ffmpeg

print("\n✅ Environment ready")
```

### 2. Dependency Installation Pattern

```python
# Cell 2: Install Python Dependencies
%%capture
# Suppress installation output

# Core ML packages
!pip install -q torch torchvision torchaudio
!pip install -q transformers accelerate
!pip install -q datasets

# Visualization
!pip install -q matplotlib seaborn plotly

# Utils
!pip install -q tqdm ipywidgets

print("✅ Dependencies installed")
```

**Best Practice:** Use `%%capture` magic to suppress verbose installation output.

### 3. Google Drive Integration

```python
# Cell 3: Mount Google Drive (optional)
from google.colab import drive

# Mount at standard location
drive.mount('/content/drive', force_remount=False)

# Define workspace in Drive for persistence
DRIVE_ROOT = Path('/content/drive/MyDrive/colab_workspace')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# Organize by project
PROJECT_DIR = DRIVE_ROOT / 'my_project'
PROJECT_DIR.mkdir(exist_ok=True)

# Standard subdirectories
(PROJECT_DIR / 'models').mkdir(exist_ok=True)
(PROJECT_DIR / 'checkpoints').mkdir(exist_ok=True)
(PROJECT_DIR / 'outputs').mkdir(exist_ok=True)
(PROJECT_DIR / 'data').mkdir(exist_ok=True)

print(f"✅ Workspace: {PROJECT_DIR}")
```

**When to use Drive:**
- Need persistence across sessions
- Frequently reuse models/data
- Collaborate with shared Drive folders

**When to avoid:**
- Small files (<100MB) - slower I/O than local disk
- Temporary experiments
- First-time users (adds complexity)

## Cell Organization Patterns

### 1. Configuration Cell

```python
# Cell: Configuration
from types import SimpleNamespace

config = SimpleNamespace(
    # Model
    model_name='gpt2',
    checkpoint='gpt2',

    # Training
    batch_size=16,
    learning_rate=5e-5,
    num_epochs=3,
    gradient_accumulation_steps=4,

    # Paths
    model_dir=Path('/content/models'),
    output_dir=Path('/content/outputs'),
    checkpoint_dir=Path('/content/drive/MyDrive/checkpoints'),

    # Runtime
    use_fp16=True,
    seed=42,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create directories
config.model_dir.mkdir(exist_ok=True)
config.output_dir.mkdir(exist_ok=True)

print("Configuration:")
for key, value in vars(config).items():
    print(f"  {key}: {value}")
```

**Benefits:**
- Single source of truth for hyperparameters
- Easy to modify and experiment
- Self-documenting

### 2. Import Organization

```python
# Cell: Imports
# Standard library
import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments
)

# Utils
from tqdm.auto import tqdm

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Imports complete")
```

### 3. Helper Functions Cell

```python
# Cell: Helper Functions

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_time(seconds):
    """Format seconds as human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def print_model_summary(model):
    """Print model architecture summary."""
    trainable, total = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Total params: {total:,}")
    print(f"  Model size: {total * 4 / 1024**2:.1f} MB (fp32)")

set_seed(config.seed)
print("✅ Helper functions defined")
```

## File Management

### 1. Upload Files

```python
from google.colab import files

# Single file upload
uploaded = files.upload()

# Access uploaded file
for filename, content in uploaded.items():
    with open(filename, 'wb') as f:
        f.write(content)
    print(f"Uploaded: {filename} ({len(content)} bytes)")

# Alternative: drag and drop to Files panel (left sidebar)
```

### 2. Download Results

```python
from google.colab import files

# Download single file
files.download('output.txt')

# Download multiple files (create zip first)
!zip -r outputs.zip /content/outputs/

files.download('outputs.zip')
```

### 3. Sync with GitHub

```python
# Clone repository
!git clone https://github.com/username/repo.git
%cd repo

# Make changes, commit, and push
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

!git add .
!git commit -m "Update from Colab"
!git push
```

## Notebook Management

### 1. Cell Magic Commands

```python
# Timing
%time result = expensive_function()  # Single execution
%timeit result = cheap_function()     # Multiple runs for average

# Memory usage
%memit model = create_large_model()

# Capture output
%%capture output
print("This will be captured")
!ls
# Access later: output.stdout, output.stderr

# Write cell to file
%%writefile script.py
def hello():
    print("Hello from file")

# Run cell as bash script
%%bash
echo "Running bash commands"
ls -la

# HTML/JavaScript execution
%%html
<h1>Custom HTML in notebook</h1>

%%javascript
console.log("JavaScript in Colab");
```

### 2. Display Utilities

```python
from IPython.display import (
    display, HTML, Markdown,
    Image, Video, Audio,
    clear_output
)

# Display images
display(Image('image.png', width=400))

# Display markdown
display(Markdown('## Results\n\nAccuracy: **95.3%**'))

# Display HTML
display(HTML('<h2 style="color: blue">Custom Styling</h2>'))

# Clear previous output (useful in loops)
for i in range(100):
    clear_output(wait=True)
    print(f"Progress: {i+1}/100")
    time.sleep(0.1)
```

### 3. Interactive Widgets

```python
from ipywidgets import interact, widgets

# Simple slider
@interact(learning_rate=(1e-5, 1e-3, 1e-5))
def update_lr(learning_rate):
    config.learning_rate = learning_rate
    print(f"Learning rate: {learning_rate:.2e}")

# Dropdown
@interact(model_name=['gpt2', 'gpt2-medium', 'gpt2-large'])
def select_model(model_name):
    config.model_name = model_name
    print(f"Selected: {model_name}")

# Multiple widgets
@interact(
    batch_size=widgets.IntSlider(min=1, max=64, value=16),
    use_fp16=widgets.Checkbox(value=True),
    epochs=widgets.IntSlider(min=1, max=10, value=3)
)
def configure_training(batch_size, use_fp16, epochs):
    config.batch_size = batch_size
    config.use_fp16 = use_fp16
    config.num_epochs = epochs
    print(f"Batch size: {batch_size}")
    print(f"FP16: {use_fp16}")
    print(f"Epochs: {epochs}")
```

## Training Workflow Patterns

### 1. Training Loop with Progress Tracking

```python
from tqdm.auto import tqdm
import time

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return avg_loss

# Training loop
for epoch in range(config.num_epochs):
    print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

    train_loss = train_epoch(model, train_loader, optimizer, config.device)
    val_loss = evaluate(model, val_loader, config.device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    # Save checkpoint
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch, val_loss)
        best_val_loss = val_loss
```

### 2. Metrics Tracking

```python
class MetricsLogger:
    """Simple metrics tracking for training."""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }

    def log(self, metrics: dict):
        """Log metrics for current epoch."""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def plot(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print summary statistics."""
        print("Training Summary:")
        print(f"  Best Val Loss: {min(self.history['val_loss']):.4f}")
        print(f"  Best Val Acc: {max(self.history['val_acc']):.4f}")
        print(f"  Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"  Final Val Loss: {self.history['val_loss'][-1]:.4f}")

# Usage
logger = MetricsLogger()

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    logger.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'learning_rate': optimizer.param_groups[0]['lr']
    })

logger.plot()
logger.summary()
```

## Debugging and Troubleshooting

### 1. Debug Mode

```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Enable deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Verbose error messages
import traceback
import sys

def debug_run(fn):
    """Run function with full error traceback."""
    try:
        return fn()
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()
        sys.exit(1)

# Usage
debug_run(lambda: train_model())
```

### 2. Memory Debugging

```python
def print_memory_allocated():
    """Print current GPU memory allocation."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# Check memory at checkpoints
print_memory_allocated()
model = create_model()
print_memory_allocated()
optimizer = create_optimizer()
print_memory_allocated()
```

### 3. Tensor Shape Debugging

```python
def print_shapes(tensors, names=None):
    """Print shapes of multiple tensors."""
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    for name, tensor in zip(names, tensors):
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tuple(tensor.shape)} {tensor.dtype} {tensor.device}")
        else:
            print(f"{name}: {type(tensor)}")

# Usage
print_shapes(
    [input_ids, attention_mask, labels],
    ['input_ids', 'attention_mask', 'labels']
)
```

## Best Practices Summary

### Do's
1. **Use configuration cells** for hyperparameters
2. **Mount Drive** for persistence across sessions
3. **Save checkpoints** frequently to Drive
4. **Use progress bars** (tqdm) for long operations
5. **Clear GPU cache** between experiments
6. **Pin pip versions** for reproducibility
7. **Use form fields** (`#@param`) for user inputs
8. **Profile code** before optimizing
9. **Document cells** with markdown
10. **Export notebooks** to Drive/GitHub regularly

### Don'ts
1. **Don't hardcode paths** - use Path objects
2. **Don't mix setup and execution** - separate cells
3. **Don't ignore runtime limits** - monitor usage
4. **Don't store secrets** in notebooks - use Colab secrets
5. **Don't rely on cell execution order** - notebook should run top-to-bottom
6. **Don't use massive print statements** - capture or log to file
7. **Don't ignore warnings** - often indicate real issues
8. **Don't skip seed setting** - reproducibility matters
9. **Don't over-rely on GPU** - some tasks faster on CPU
10. **Don't forget to disconnect** - free resources when done

## Common Patterns

### 1. Notebook Template

```python
# ============================================
# GOOGLE COLAB NOTEBOOK TEMPLATE
# ============================================

# ---- Cell 1: Environment Check ----
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ---- Cell 2: Install Dependencies ----
%%capture
!pip install -q torch transformers datasets

# ---- Cell 3: Mount Drive (Optional) ----
# from google.colab import drive
# drive.mount('/content/drive')

# ---- Cell 4: Imports ----
import os
import sys
from pathlib import Path
import torch
from transformers import AutoModel

# ---- Cell 5: Configuration ----
from types import SimpleNamespace
config = SimpleNamespace(
    model_name='bert-base-uncased',
    batch_size=16,
    learning_rate=5e-5,
    seed=42
)

# ---- Cell 6: Load Data ----
# Load your data here

# ---- Cell 7: Create Model ----
model = AutoModel.from_pretrained(config.model_name)

# ---- Cell 8: Training ----
# Training loop here

# ---- Cell 9: Evaluation ----
# Evaluation code here

# ---- Cell 10: Results & Visualization ----
# Plot results here

# ---- Cell 11: Save/Export ----
# Save model and artifacts
```

### 2. Experiment Tracking

```python
import json
from datetime import datetime

class ExperimentTracker:
    """Track experiment metadata and results."""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.metadata = {
            'name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'config': {},
            'results': {},
            'artifacts': []
        }

    def log_config(self, config):
        """Log experiment configuration."""
        if hasattr(config, '__dict__'):
            self.metadata['config'] = vars(config)
        else:
            self.metadata['config'] = config

    def log_result(self, key, value):
        """Log a result metric."""
        self.metadata['results'][key] = value

    def log_artifact(self, path):
        """Log an artifact path."""
        self.metadata['artifacts'].append(str(path))

    def save(self, path):
        """Save experiment metadata."""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration'] = str(datetime.now() - self.start_time)

        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

# Usage
tracker = ExperimentTracker('exp_001_baseline')
tracker.log_config(config)

# After training
tracker.log_result('val_loss', 0.234)
tracker.log_result('val_acc', 0.932)
tracker.log_artifact('/content/outputs/model.pt')

tracker.save('/content/drive/MyDrive/experiments/exp_001_metadata.json')
```
