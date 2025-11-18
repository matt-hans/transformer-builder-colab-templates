# Model Management in Google Colab

Comprehensive guide to downloading, organizing, and managing ML models in Colab.

## Download Methods

### 1. aria2c (Recommended for Speed)

Fast, resumable downloads with parallel connections:

```python
# Install aria2c
!apt -y install -qq aria2

# Download with optimal settings
!aria2c --console-log-level=error \
    -c \
    -x 16 \
    -s 16 \
    -k 1M \
    {url} \
    -d {destination_directory} \
    -o {filename}
```

**Parameters:**
- `-c`: Continue interrupted downloads
- `-x 16`: Max 16 connections per server
- `-s 16`: Split download into 16 parts
- `-k 1M`: Minimum split size of 1MB

**When to use:** Large models (>1GB), slow connections, resumable downloads needed

### 2. wget (Simple, Authenticated)

For authenticated downloads (Civitai, HuggingFace with tokens):

```python
# Public downloads
!wget {url} -O {filename}

# With authentication (Civitai)
!wget --content-disposition \
    --header="Authorization: Bearer {api_token}" \
    {url} \
    -P {destination_directory}
```

**When to use:** Authenticated downloads, simple single-file downloads

### 3. HuggingFace Hub (Native)

For HuggingFace models:

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Single file
model_path = hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="model.safetensors",
    cache_dir="/content/models"
)

# Entire repository
snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="/content/models/sdxl",
    token=hf_token  # Optional for private models
)
```

**When to use:** HuggingFace hosted models, need version control

### 4. Google Drive (Persistent Storage)

For models you've already downloaded:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive to runtime
!cp /content/drive/MyDrive/models/my_model.safetensors /content/models/
```

**When to use:** Reusing models across sessions, avoiding re-downloads

## Directory Organization

### Standard Structure

```
/content/
├── models/                 # Downloaded models
│   ├── checkpoints/       # Main model files (.safetensors, .ckpt)
│   ├── loras/             # LoRA adapters
│   ├── vae/               # VAE models
│   ├── unet/              # UNet models
│   ├── text_encoders/     # CLIP, T5, etc.
│   └── embeddings/        # Textual inversions
├── outputs/               # Generated outputs
│   ├── images/
│   └── videos/
└── workspace/             # Working directory
    └── temp/
```

### Model Path Management

```python
from pathlib import Path

# Define base directories
MODEL_BASE = Path("/content/models")
OUTPUT_BASE = Path("/content/outputs")

# Create structure
for subdir in ["checkpoints", "loras", "vae", "unet", "text_encoders", "embeddings"]:
    (MODEL_BASE / subdir).mkdir(parents=True, exist_ok=True)

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Use in code
checkpoint_path = MODEL_BASE / "checkpoints" / "model.safetensors"
lora_path = MODEL_BASE / "loras" / "style.safetensors"
```

## Model File Formats

### safetensors (Recommended)

```python
from safetensors.torch import load_file, save_file

# Load
state_dict = load_file("model.safetensors")

# Save
save_file(model.state_dict(), "model.safetensors")
```

**Advantages:**
- Fast loading
- Memory efficient
- Safe (no arbitrary code execution)
- Cross-platform compatible

### PyTorch Checkpoints (.pt, .pth, .ckpt)

```python
import torch

# Load
checkpoint = torch.load("model.ckpt", map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])

# Save
torch.save({
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}, "checkpoint.ckpt")
```

**When to use:** PyTorch native workflows, need optimizer state

### GGUF (Quantized Models)

```python
# Load with llama.cpp Python bindings
from llama_cpp import Llama

llm = Llama(
    model_path="model.gguf",
    n_ctx=2048,
    n_gpu_layers=35  # Offload to GPU
)
```

**When to use:** CPU inference, memory-constrained environments

## Common Model Sources

### 1. Civitai

```python
# Download function
def download_civitai(model_id, api_token=None):
    base_url = f"https://civitai.com/api/download/models/{model_id}"

    if api_token:
        !wget --content-disposition \
            --header="Authorization: Bearer {api_token}" \
            {base_url} \
            -P /content/models/checkpoints
    else:
        !aria2c -x 16 -s 16 -k 1M {base_url} \
            -d /content/models/checkpoints
```

### 2. HuggingFace

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    local_dir="/content/models/sd15",
    allow_patterns=["*.safetensors", "*.json"],  # Filter files
    ignore_patterns=["*.bin"]  # Exclude PyTorch bins
)
```

### 3. Direct URLs

```python
# Using the model_download.py script
from scripts.model_download import download_with_aria2c

download_with_aria2c(
    url="https://example.com/model.safetensors",
    dest_dir="/content/models/checkpoints",
    filename="custom_model.safetensors"
)
```

## Memory Management

### Check Available Space

```python
import shutil

def check_disk_space(path="/content"):
    total, used, free = shutil.disk_usage(path)
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB")
    print(f"Free: {free // (2**30)} GB")

check_disk_space()
```

### Clean Up Models

```python
import os
from pathlib import Path

def cleanup_models(keep_patterns=None):
    """Remove all models except those matching keep_patterns."""
    model_dir = Path("/content/models")

    if keep_patterns is None:
        keep_patterns = []

    for file in model_dir.rglob("*"):
        if file.is_file():
            # Check if file should be kept
            if not any(pattern in str(file) for pattern in keep_patterns):
                file.unlink()
                print(f"Removed: {file.name}")

# Example: keep only SDXL models
cleanup_models(keep_patterns=["sdxl", "xl-base"])
```

### Load Models On-Demand

```python
class ModelManager:
    def __init__(self):
        self.models = {}

    def load_model(self, name, path):
        """Load model only when needed."""
        if name not in self.models:
            from safetensors.torch import load_file
            self.models[name] = load_file(path)
            print(f"Loaded: {name}")
        return self.models[name]

    def unload_model(self, name):
        """Free model from memory."""
        if name in self.models:
            del self.models[name]
            torch.cuda.empty_cache()
            print(f"Unloaded: {name}")

manager = ModelManager()
```

## Best Practices

1. **Use aria2c for large downloads** (>1GB) for speed and resumability
2. **Organize models by type** (checkpoints, loras, vae) for clarity
3. **Prefer safetensors format** for security and speed
4. **Mount Google Drive** for frequently used models to avoid re-downloads
5. **Clean up after use** - Colab has limited disk space (~200GB total)
6. **Verify checksums** when available to ensure download integrity
7. **Use symbolic links** to avoid duplication when same model used in multiple places

## Troubleshooting

### Download Fails

```python
# Check internet connectivity
!ping -c 3 google.com

# Verify URL is accessible
!curl -I {url}

# Use alternative download method
# Try wget if aria2c fails, or vice versa
```

### Out of Disk Space

```python
# Check what's using space
!du -sh /content/* | sort -h

# Clean up outputs
!rm -rf /content/outputs/*

# Remove unused models
cleanup_models()
```

### Model Won't Load

```python
# Verify file integrity
!ls -lh /content/models/model.safetensors

# Check file format
!file /content/models/model.safetensors

# Try loading with error handling
try:
    from safetensors.torch import load_file
    state_dict = load_file("model.safetensors")
except Exception as e:
    print(f"Load failed: {e}")
    # Try alternative loading method
```
