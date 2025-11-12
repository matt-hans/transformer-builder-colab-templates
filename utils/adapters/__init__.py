"""
Model adapters for handling arbitrary transformer architectures.

This module provides tools to wrap generated models with complex signatures
into a unified interface compatible with PyTorch Lightning.
"""

# ModelSignatureInspector is complete (Task 1.3)
from .model_adapter import ModelSignatureInspector

# ComputationalGraphExecutor will be completed in Task 1.4
# from .model_adapter import ComputationalGraphExecutor

# UniversalModelAdapter will be completed in Task 2.1
# from .model_adapter import UniversalModelAdapter

__all__ = [
    'ModelSignatureInspector',
    # 'ComputationalGraphExecutor',  # Task 1.4
    # 'UniversalModelAdapter',       # Task 2.1
]
