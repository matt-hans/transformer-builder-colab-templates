"""
Model adapters for handling arbitrary transformer architectures.

This module provides tools to wrap generated models with complex signatures
into a unified interface compatible with PyTorch Lightning.
"""

# Tasks 1.3, 1.4, 2.1 complete
from .model_adapter import (
    ModelSignatureInspector,
    ComputationalGraphExecutor,
    UniversalModelAdapter,
    ModelAdapter,
    DecoderOnlyLMAdapter,
    EncoderOnlyClassificationAdapter,
    EncoderDecoderSeq2SeqAdapter,
)

__all__ = [
    'ModelSignatureInspector',
    'ComputationalGraphExecutor',
    'UniversalModelAdapter',
    'ModelAdapter',
    'DecoderOnlyLMAdapter',
    'EncoderOnlyClassificationAdapter',
    'EncoderDecoderSeq2SeqAdapter',
]
