---
name: google-colab-expert
description: This skill should be used when working with Google Colab notebooks for ML/AI workflows, model management, performance optimization, or implementing production-ready Colab environments. Applies to tasks involving GPU optimization, model downloads, training pipelines, memory management, and Colab-specific patterns.
---

# Google Colab Expert

Expert guidance for building production-ready ML/AI workflows in Google Colab. Covers environment setup, model management, GPU optimization, and best practices for efficient Colab development.

## When to Use This Skill

Apply this skill when working on:
- Setting up ML/AI environments in Google Colab
- Downloading and managing large models (HuggingFace, Civitai, custom URLs)
- Optimizing GPU utilization and memory management
- Implementing training pipelines with performance optimization
- Debugging OOM errors and runtime issues
- Structuring Colab notebooks for reproducibility and collaboration
- Integrating Google Drive for persistence
- Building production-ready workflows in Colab

## Core Workflows

### 1. Environment Setup

Follow this pattern for every new Colab notebook:

\`\`\`python
# Cell 1: Environment check and system dependencies
import torch
import sys

print(f"Python: {sys.version}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Install system dependencies
!apt -y install -qq aria2 ffmpeg

# Cell 2: Python dependencies (use %%capture to suppress output)
%%capture
!pip install -q torch torchvision transformers accelerate
!pip install -q datasets tqdm ipywidgets

# Cell 3: Verify installation
import torch
from transformers import AutoModel
print("✅ Environment ready")
\`\`\`

**Key principles:**
- Separate system dependencies (apt) from Python packages (pip)
- Use \`%%capture\` for clean output
- Verify critical imports before proceeding
- Check GPU availability first thing

**Reference:** See \`references/colab_best_practices.md\` for complete environment setup patterns.

### 2. Model Downloads

Use the provided \`scripts/model_download.py\` for efficient model downloads:

\`\`\`python
# Load the script
from scripts.model_download import download_with_aria2c, install_aria2c

# Ensure aria2c is installed
install_aria2c()

# Download model with optimal settings
success = download_with_aria2c(
    url="https://example.com/model.safetensors",
    dest_dir="/content/models/checkpoints",
    filename="custom_model.safetensors"
)
\`\`\`

**When to use different methods:**
- **aria2c** (recommended): Large files (>1GB), resumable downloads, maximum speed
- **wget**: Authenticated downloads (Civitai with API token)
- **HuggingFace hub**: Official HF models, automatic version management
- **Google Drive**: Frequently reused models, avoid re-downloads

**Organization pattern:**
\`\`\`
/content/models/
├── checkpoints/      # Main models (.safetensors, .ckpt)
├── loras/           # LoRA adapters
├── vae/             # VAE models
├── unet/            # UNet models
└── text_encoders/   # CLIP, T5, etc.
\`\`\`

**Reference:** See \`references/model_management.md\` for complete download methods and organization.

### 3. GPU Optimization

Use the provided \`scripts/gpu_check.py\` for GPU monitoring:

\`\`\`python
from scripts.gpu_check import print_gpu_summary, check_memory_for_model

# Check GPU availability and memory
print_gpu_summary()

# Check if model will fit
can_fit, message = check_memory_for_model(
    num_parameters=125_000_000,  # 125M parameters
    dtype='fp16'
)
print(message)
\`\`\`

**Essential optimization techniques:**

1. **Mixed Precision (FP16/BF16)** - Use for ~2x speedup
2. **Gradient Accumulation** - For larger effective batch sizes
3. **Clear GPU cache** - Between experiments

**Reference:** See \`references/performance_optimization.md\` for complete optimization techniques.

### 4. Google Drive Integration

Integrate Drive for persistence across sessions:

\`\`\`python
from google.colab import drive
from pathlib import Path

# Mount Drive
drive.mount('/content/drive')

# Set up workspace
WORKSPACE = Path('/content/drive/MyDrive/colab_projects/my_project')
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(WORKSPACE / 'models').mkdir(exist_ok=True)
(WORKSPACE / 'checkpoints').mkdir(exist_ok=True)
(WORKSPACE / 'outputs').mkdir(exist_ok=True)

print(f"✅ Workspace: {WORKSPACE}")
\`\`\`

**When to use Drive:**
- Models/data used across multiple sessions
- Checkpoints to survive disconnections
- Collaboration via shared folders
- Long-running experiments (>2 hours)

## Bundled Resources

### Scripts

1. **\`scripts/model_download.py\`** - Efficient model downloads using aria2c
2. **\`scripts/gpu_check.py\`** - GPU monitoring and diagnostics
3. **\`scripts/setup_environment.py\`** - Environment initialization utilities

### References

1. **\`references/model_management.md\`** - Complete guide to model downloads and organization
2. **\`references/performance_optimization.md\`** - GPU optimization and performance tuning
3. **\`references/colab_best_practices.md\`** - General Colab workflows and patterns

**When to load references:**
- Read \`model_management.md\` when implementing model downloads or organization
- Read \`performance_optimization.md\` when encountering OOM errors or slow training
- Read \`colab_best_practices.md\` when structuring new notebooks or debugging workflows

## Best Practices Summary

### Do's
1. Check GPU availability first
2. Use aria2c for large downloads (5-10x faster)
3. Enable mixed precision (2x speedup)
4. Save checkpoints to Drive
5. Use configuration cells for hyperparameters
6. Monitor memory usage regularly
7. Use progress bars with tqdm
8. Organize models by type
9. Profile before optimizing
10. Clear GPU cache between experiments

### Don'ts
1. Don't hardcode paths
2. Don't ignore runtime limits (12-hour max)
3. Don't skip environment checks
4. Don't use Drive for small temporary files
5. Don't mix setup and execution cells
6. Don't skip checkpointing
7. Don't ignore warnings
8. Don't use massive print statements
9. Don't forget to disconnect when done
10. Don't over-rely on free tier for production work

## Summary

Use this skill for production-ready Google Colab workflows with focus on efficient environment setup, fast model downloads, GPU optimization, training pipelines with checkpointing, and Google Drive integration.
