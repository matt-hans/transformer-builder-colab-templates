"""
Training infrastructure for production-ready model training.

Includes dataset loading, checkpoint management, training coordination,
and export utilities for ONNX and TorchScript.
"""

# Dataset utilities (Tasks 3.1-3.2)
from .dataset_utilities import DatasetLoader, DatasetUploader

# Checkpoint management (Task 3.3)
from .checkpoint_manager import CheckpointManager

# Training core (Task 4.1)
from .training_core import TrainingCoordinator, train_model

# Export utilities (Tasks 4.2-4.4)
from .export_utilities import ONNXExporter, TorchScriptExporter, ModelCardGenerator

__all__ = [
    # Dataset utilities
    'DatasetLoader',
    'DatasetUploader',

    # Checkpoint management
    'CheckpointManager',

    # Training
    'TrainingCoordinator',
    'train_model',

    # Export
    'ONNXExporter',
    'TorchScriptExporter',
    'ModelCardGenerator',
]
