#!/usr/bin/env python3
"""
GPU detection and memory monitoring for Google Colab.

Provides utilities to check GPU availability, memory usage,
and optimize runtime allocation.
"""

import subprocess
import torch
from typing import Dict, Optional, Tuple


def get_gpu_info() -> Dict[str, any]:
    """
    Get comprehensive GPU information.

    Returns:
        Dictionary with GPU details:
        - available: bool
        - device_name: str
        - memory_total: float (GB)
        - memory_allocated: float (GB)
        - memory_reserved: float (GB)
        - memory_free: float (GB)
        - compute_capability: tuple
    """
    info = {
        'available': torch.cuda.is_available(),
        'device_name': None,
        'memory_total': 0.0,
        'memory_allocated': 0.0,
        'memory_reserved': 0.0,
        'memory_free': 0.0,
        'compute_capability': None
    }

    if info['available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3
        info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**3
        info['memory_free'] = info['memory_total'] - info['memory_allocated']
        info['compute_capability'] = torch.cuda.get_device_capability(0)

    return info


def print_gpu_summary():
    """Print formatted GPU information."""
    info = get_gpu_info()

    if not info['available']:
        print("❌ No GPU available")
        print("\nTo enable GPU in Colab:")
        print("  Runtime → Change runtime type → Hardware accelerator → GPU")
        return

    print(f"✅ GPU Available: {info['device_name']}")
    print(f"   Memory Total: {info['memory_total']:.2f} GB")
    print(f"   Memory Allocated: {info['memory_allocated']:.2f} GB")
    print(f"   Memory Free: {info['memory_free']:.2f} GB")
    print(f"   Compute Capability: {info['compute_capability'][0]}.{info['compute_capability'][1]}")


def get_nvidia_smi_output() -> Optional[str]:
    """Get nvidia-smi output if available."""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def clear_gpu_memory():
    """
    Clear GPU memory cache.

    Useful when switching between models or when encountering OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
        print_memory_usage()
    else:
        print("No GPU available to clear")


def print_memory_usage():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU Memory Usage:")
    print(f"  Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  Reserved:  {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")


def estimate_model_memory(num_parameters: int, dtype_bytes: int = 4) -> float:
    """
    Estimate model memory requirements.

    Args:
        num_parameters: Number of model parameters
        dtype_bytes: Bytes per parameter (4 for fp32, 2 for fp16)

    Returns:
        Estimated memory in GB
    """
    return (num_parameters * dtype_bytes) / 1024**3


def check_memory_for_model(num_parameters: int, dtype: str = 'fp32') -> Tuple[bool, str]:
    """
    Check if GPU has enough memory for a model.

    Args:
        num_parameters: Number of model parameters
        dtype: Data type ('fp32', 'fp16', 'int8')

    Returns:
        Tuple of (can_fit: bool, message: str)
    """
    dtype_bytes = {'fp32': 4, 'fp16': 2, 'int8': 1}.get(dtype, 4)

    required_memory = estimate_model_memory(num_parameters, dtype_bytes)
    info = get_gpu_info()

    if not info['available']:
        return False, "No GPU available"

    # Account for ~20% overhead for activations/gradients
    required_with_overhead = required_memory * 1.2

    if info['memory_free'] >= required_with_overhead:
        return True, f"✅ Sufficient memory ({info['memory_free']:.2f} GB available, {required_with_overhead:.2f} GB needed)"
    else:
        return False, f"❌ Insufficient memory ({info['memory_free']:.2f} GB available, {required_with_overhead:.2f} GB needed)"


if __name__ == '__main__':
    print_gpu_summary()
    print()

    nvidia_output = get_nvidia_smi_output()
    if nvidia_output:
        print("nvidia-smi output:")
        print(nvidia_output)
