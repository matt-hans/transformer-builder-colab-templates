"""
Model adapters for handling arbitrary transformer architectures.

This module provides tools to wrap generated models with complex signatures
into a unified interface compatible with PyTorch Lightning.
"""

# Task 1.3 & 1.4 complete
from .model_adapter import (
    ModelSignatureInspector,
    ComputationalGraphExecutor
)

# UniversalModelAdapter will be completed in Task 2.1
# from .model_adapter import UniversalModelAdapter

__all__ = [
    'ModelSignatureInspector',
    'ComputationalGraphExecutor',
    # 'UniversalModelAdapter',  # Task 2.1
]
