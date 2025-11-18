"""
Model adapters for handling arbitrary transformer and vision architectures.

This module provides tools to wrap generated models with complex signatures
into a unified interface compatible with PyTorch Lightning, as well as a
family of lightweight task-aware adapters used by the training/eval stack.
"""

from .model_adapter import (
    ModelSignatureInspector,
    ComputationalGraphExecutor,
    UniversalModelAdapter,
    ModelAdapter,
    DecoderOnlyLMAdapter,
    EncoderOnlyClassificationAdapter,
    EncoderDecoderSeq2SeqAdapter,
    VisionClassificationAdapter,
)

__all__ = [
    'ModelSignatureInspector',
    'ComputationalGraphExecutor',
    'UniversalModelAdapter',
    'ModelAdapter',
    'DecoderOnlyLMAdapter',
    'EncoderOnlyClassificationAdapter',
    'EncoderDecoderSeq2SeqAdapter',
    'VisionClassificationAdapter',
]
