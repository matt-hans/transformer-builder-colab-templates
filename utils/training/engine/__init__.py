"""
Training Engine Package

Modular training pipeline components for transformer models.

Modules:
- checkpoint: Checkpoint management and state persistence
- loss: Task-aware loss computation strategies
- gradient_monitor: Gradient health checks and monitoring
- gradient_accumulator: Gradient accumulation management
- data: DataLoader and collation strategies
- loop: Training and validation epoch execution (NEW in Phase 1 - P1-4)
- trainer: High-level training orchestration (Phase 1)
- metrics: Metrics tracking with drift detection and alerts (Phase 1)
- visualization: Training visualization and plotting (coming soon in Phase 2)
"""

# Checkpoint management
from utils.training.engine.checkpoint import CheckpointManager, CheckpointMetadata

# Loss computation strategies
from utils.training.engine.loss import (
    LossStrategy,
    LossInputs,
    ModelOutput,
    LanguageModelingLoss,
    ClassificationLoss,
    PEFTAwareLoss,
    QuantizationSafeLoss,
    VisionLoss,
    LossStrategyRegistry,
    get_loss_strategy
)

# Gradient monitoring
from utils.training.engine.gradient_monitor import (
    GradientMonitor,
    GradientHealth,
    check_gradient_health
)

# Gradient accumulation
from utils.training.engine.gradient_accumulator import (
    GradientAccumulator,
    AccumulationStats
)

# Data loading and collation
from utils.training.engine.data import (
    DataModuleProtocol,
    CollatorRegistry,
    CollatorInfo,
    DataLoaderConfig,
    DataLoaderFactory,
    UniversalDataModule
)

# Training and validation loops (Phase 1 - P1-4)
from utils.training.engine.loop import (
    TrainingLoop,
    ValidationLoop,
    EpochResult
)

# Training orchestration (Phase 1)
from utils.training.engine.trainer import (
    Trainer,
    TrainingHooks,
    DefaultHooks
)

# Metrics tracking (Phase 1)
from utils.training.engine.metrics import (
    MetricsEngine,
    DriftMetrics,
    ConfidenceMetrics,
    AlertConfig,
    AlertCallback
)

__all__ = [
    # Checkpoint
    'CheckpointManager',
    'CheckpointMetadata',

    # Loss
    'LossStrategy',
    'LossInputs',
    'ModelOutput',
    'LanguageModelingLoss',
    'ClassificationLoss',
    'PEFTAwareLoss',
    'QuantizationSafeLoss',
    'VisionLoss',
    'LossStrategyRegistry',
    'get_loss_strategy',

    # Gradient monitoring
    'GradientMonitor',
    'GradientHealth',
    'check_gradient_health',

    # Gradient accumulation
    'GradientAccumulator',
    'AccumulationStats',

    # Data loading
    'DataModuleProtocol',
    'CollatorRegistry',
    'CollatorInfo',
    'DataLoaderConfig',
    'DataLoaderFactory',
    'UniversalDataModule',

    # Training and validation loops
    'TrainingLoop',
    'ValidationLoop',
    'EpochResult',

    # Training orchestration
    'Trainer',
    'TrainingHooks',
    'DefaultHooks',

    # Metrics tracking
    'MetricsEngine',
    'DriftMetrics',
    'ConfidenceMetrics',
    'AlertConfig',
    'AlertCallback',
]
