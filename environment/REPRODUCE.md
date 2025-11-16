# Environment Reproduction Guide

## Quick Setup

```bash
# Python version required
python --version  # Should be 3.13.5

# Install exact package versions
pip install -r requirements.txt
```

## System Information

- **Python**: 3.13.5
- **Platform**: macOS-15.3-arm64-arm-64bit-Mach-O
- **PyTorch**: 2.9.1
- **CUDA**: N/A (CPU only)
- **GPU**: N/A

## Key Package Versions

- torch==2.9.1
- transformers==4.57.1
- numpy==2.3.4

## Verification

After installation, verify with:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

Expected output:
```
PyTorch: 2.9.1
CUDA available: False
```

## Notes

- If using Google Colab, GPU type may differ (T4 vs A100 vs V100)
- Some CUDA operations are non-deterministic even with same seed
- Use deterministic mode for bit-exact reproduction (slower):
  ```python
  from utils.training.seed_manager import set_random_seed
  set_random_seed(42, deterministic=True)
  ```

## Troubleshooting

### Different Python version
If you have Python 3.13.X instead of 3.13.5:
- Minor version differences (3.10.X) usually work
- Major version differences (3.9 vs 3.10) may cause issues

### CUDA version mismatch
If you get CUDA errors:
- Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
- Or use CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Package conflicts
If pip install fails with conflicts:
1. Create fresh virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)
3. Install: `pip install -r requirements.txt`
