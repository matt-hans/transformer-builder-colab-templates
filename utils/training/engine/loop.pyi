import torch.nn as nn
from _typeshed import Incomplete
from dataclasses import dataclass, field as field
from torch.optim import Optimizer as Optimizer
from torch.optim.lr_scheduler import LRScheduler as LRScheduler
from torch.utils.data import DataLoader as DataLoader
from typing import Any
from utils.training.engine.gradient_accumulator import GradientAccumulator as GradientAccumulator
from utils.training.engine.gradient_monitor import GradientHealth as GradientHealth, GradientMonitor as GradientMonitor
from utils.training.engine.loss import LossInputs as LossInputs, LossStrategy as LossStrategy, ModelOutput as ModelOutput

logger: Incomplete

@dataclass
class EpochResult:
    loss: float
    accuracy: float
    metrics: dict[str, float]
    duration: float
    batch_count: int = ...
    gradient_norms: list[float] | None = ...
    loss_history: list[float] | None = ...
    learning_rate: float | None = ...
    throughput: float | None = ...
    def to_dict(self) -> dict[str, Any]: ...

class TrainingLoop:
    loss_strategy: Incomplete
    gradient_accumulator: Incomplete
    gradient_monitor: Incomplete
    use_amp: Incomplete
    device: Incomplete
    progress_bar: Incomplete
    scaler: Incomplete
    def __init__(self, loss_strategy: LossStrategy, gradient_accumulator: GradientAccumulator, gradient_monitor: GradientMonitor | None = None, use_amp: bool = False, device: str = 'cuda', progress_bar: bool = True) -> None: ...
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler | None = None, epoch: int = 0, metrics_tracker: Any | None = None) -> EpochResult: ...

class ValidationLoop:
    loss_strategy: Incomplete
    device: Incomplete
    progress_bar: Incomplete
    def __init__(self, loss_strategy: LossStrategy, device: str = 'cuda', progress_bar: bool = True) -> None: ...
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader, epoch: int = 0, metrics_tracker: Any | None = None) -> EpochResult: ...
