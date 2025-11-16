"""
Training infrastructure for production-ready model training.

Includes dataset loading, checkpoint management, training coordination,
metrics tracking with W&B integration, and export utilities for ONNX and TorchScript.
"""

# Dataset utilities (Tasks 3.1-3.2)
from .dataset_utilities import DatasetLoader, DatasetUploader

# Checkpoint management (Task 3.3) - requires pytorch_lightning
try:
    from .checkpoint_manager import CheckpointManager
except ImportError:
    CheckpointManager = None

# Training core (Task 4.1) - requires pytorch_lightning
try:
    from .training_core import TrainingCoordinator, train_model
except ImportError:
    TrainingCoordinator = None
    train_model = None

# Metrics tracking (Task T002)
from .metrics_tracker import MetricsTracker

# Export utilities (Tasks 4.2-4.4)
from .export_utilities import ONNXExporter, TorchScriptExporter, ModelCardGenerator

# Environment snapshot (Task T016)
from .environment_snapshot import (
    capture_environment,
    save_environment_snapshot,
    compare_environments,
    log_environment_to_wandb
)

# Seed management (Task T015)
from .seed_manager import set_random_seed, seed_worker, create_seeded_generator

__all__ = [
    # Dataset utilities
    'DatasetLoader',
    'DatasetUploader',

    # Checkpoint management
    'CheckpointManager',

    # Training
    'TrainingCoordinator',
    'train_model',

    # Metrics tracking
    'MetricsTracker',

    # Export
    'ONNXExporter',
    'TorchScriptExporter',
    'ModelCardGenerator',

    # Environment snapshot
    'capture_environment',
    'save_environment_snapshot',
    'compare_environments',
    'log_environment_to_wandb',

    # Seed management
    'set_random_seed',
    'seed_worker',
    'create_seeded_generator',
]
