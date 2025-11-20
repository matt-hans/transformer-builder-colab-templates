"""
Training Loop Orchestrator with Component Delegation

High-level training workflow coordinator that delegates to specialized components:
- CheckpointManager: State persistence and recovery
- MetricsTracker: Logging and monitoring
- LossStrategy: Task-specific loss computation
- DataLoaderFactory: Reproducible data loading
- GradientAccumulator: Gradient accumulation management

Design Principles:
1. Single Responsibility: Orchestrate, don't implement
2. Delegation: Forward work to specialized components
3. Configuration: Accept TrainingConfig, not 30+ parameters
4. Hooks: Extensible via callback system
5. Type Safety: Protocol-based interfaces for flexibility

Example:
    >>> from utils.training.engine import Trainer
    >>> from utils.training.training_config import TrainingConfig
    >>>
    >>> config = TrainingConfig(
    ...     learning_rate=5e-5,
    ...     batch_size=4,
    ...     epochs=10,
    ...     checkpoint_dir='./checkpoints'
    ... )
    >>>
    >>> trainer = Trainer(
    ...     model=model,
    ...     config=model_config,
    ...     training_config=config,
    ...     task_spec=task_spec
    ... )
    >>>
    >>> results = trainer.train(
    ...     train_data=train_dataset,
    ...     val_data=val_dataset
    ... )
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Protocol, Optional, Dict, Any, Union, Callable
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset

# Import Phase 0 components
from utils.training.engine.checkpoint import CheckpointManager
from utils.training.engine.loss import get_loss_strategy, LossStrategy, LossInputs, ModelOutput
from utils.training.engine.gradient_monitor import GradientMonitor
from utils.training.engine.gradient_accumulator import GradientAccumulator
from utils.training.engine.data import DataLoaderFactory, DataLoaderConfig, UniversalDataModule

# Import existing utilities
from utils.training.metrics_tracker import MetricsTracker
from utils.training.seed_manager import set_random_seed
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Hook Protocol for Extensibility
# =============================================================================

class TrainingHooks(Protocol):
    """
    Protocol for training lifecycle hooks.

    Enables custom behavior injection at key points in the training loop
    without modifying core trainer logic. All hooks are optional (default no-op).

    Example:
        >>> class CustomHooks:
        ...     def on_epoch_start(self, epoch: int) -> None:
        ...         print(f"Starting epoch {epoch}")
        ...
        ...     def on_batch_end(self, batch_idx: int, loss: float) -> None:
        ...         if batch_idx % 100 == 0:
        ...             print(f"Batch {batch_idx}, loss={loss:.4f}")
        >>>
        >>> trainer = Trainer(..., hooks=CustomHooks())
    """

    def on_training_start(self) -> None:
        """Called once before training begins."""
        ...

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """Called after each training batch."""
        ...

    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        """Called after validation completes."""
        ...

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch with combined metrics."""
        ...

    def on_training_end(self) -> None:
        """Called once after training completes."""
        ...


class DefaultHooks:
    """Default no-op implementation of TrainingHooks."""

    def on_training_start(self) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        pass

    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_training_end(self) -> None:
        pass


# =============================================================================
# Trainer Orchestrator
# =============================================================================

class Trainer:
    """
    High-level training orchestrator.

    Coordinates all training components without implementing low-level logic.
    Delegates to specialized modules for loss computation, checkpointing,
    metrics tracking, and gradient management.

    Architecture:
        ┌─────────────────────────────────────────┐
        │       Trainer (Orchestrator)            │
        │  - Setup components                     │
        │  - Execute training workflow            │
        │  - Handle resume/checkpoint             │
        └───────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
    ┌─────────┐         ┌──────────────┐
    │ Training│         │  Validation  │
    │  Epoch  │         │    Epoch     │
    └─────────┘         └──────────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
    ┌─────────┐         ┌──────────────┐
    │Checkpoint│         │   Metrics   │
    │ Manager │         │   Tracker   │
    └─────────┘         └──────────────┘

    Example:
        >>> config = TrainingConfig(
        ...     learning_rate=5e-5,
        ...     batch_size=4,
        ...     epochs=10
        ... )
        >>>
        >>> trainer = Trainer(
        ...     model=model,
        ...     config=model_config,
        ...     training_config=config,
        ...     task_spec=task_spec
        ... )
        >>>
        >>> results = trainer.train(
        ...     train_data=train_dataset,
        ...     val_data=val_dataset
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[SimpleNamespace, Any],  # Model config
        training_config: TrainingConfig,
        task_spec: Optional[TaskSpec] = None,
        hooks: Optional[TrainingHooks] = None
    ):
        """
        Initialize trainer with model and configuration.

        Args:
            model: PyTorch model to train
            config: Model configuration (SimpleNamespace or custom config object)
            training_config: Training hyperparameters and settings
            task_spec: Task specification for data loading and loss computation
            hooks: Optional training lifecycle hooks

        Raises:
            ValueError: If training_config validation fails
        """
        # Validate configuration first
        training_config.validate()

        self.model = model
        self.config = config
        self.training_config = training_config
        self.task_spec = task_spec
        self.hooks = hooks or DefaultHooks()

        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Set random seed for reproducibility
        set_random_seed(
            training_config.random_seed,
            training_config.deterministic
        )

        # Setup components (delegate initialization)
        # Note: Order matters! Optimizer must be created before gradient accumulator
        self.checkpoint_manager = self._setup_checkpointing()
        self.metrics_tracker = self._setup_metrics()
        self.loss_strategy = self._setup_loss_strategy()
        self.gradient_monitor = self._setup_gradient_monitor()
        self.optimizer = self._setup_optimizer()
        self.gradient_accumulator = self._setup_gradient_accumulator()
        self.scheduler = self._setup_scheduler()

        # Data module (initialized lazily in train())
        self.data_module: Optional[UniversalDataModule] = None

        logger.info(f"Trainer initialized with device={self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train(
        self,
        train_data: Union[Dataset, HFDataset, DataLoader],
        val_data: Optional[Union[Dataset, HFDataset, DataLoader]] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute complete training workflow.

        Orchestrates the full training process:
        1. Setup data loaders
        2. Resume from checkpoint if specified
        3. Execute training epochs with validation
        4. Save checkpoints at intervals
        5. Return comprehensive results

        Args:
            train_data: Training dataset (PyTorch Dataset, HF Dataset, or DataLoader)
            val_data: Validation dataset (optional)
            resume_from: Path to checkpoint to resume from (optional)

        Returns:
            Dictionary containing:
                - metrics_summary: DataFrame with per-epoch metrics
                - best_epoch: Epoch number with best validation performance
                - final_loss: Final training loss
                - checkpoint_path: Path to best checkpoint
                - training_time: Total training time in seconds

        Example:
            >>> results = trainer.train(
            ...     train_data=train_dataset,
            ...     val_data=val_dataset,
            ...     resume_from='./checkpoints/checkpoint_epoch0005.pt'
            ... )
            >>> print(f"Best model at epoch {results['best_epoch']}")
        """
        # Setup data loaders
        self._setup_data(train_data, val_data)

        # Resume if checkpoint provided
        start_epoch = 0
        if resume_from:
            start_epoch = self._resume_from_checkpoint(resume_from)
            logger.info(f"Resumed training from epoch {start_epoch}")

        # Training loop orchestration
        self.hooks.on_training_start()
        training_start_time = time.time()

        try:
            for epoch in range(start_epoch, self.training_config.epochs):
                self.hooks.on_epoch_start(epoch)
                epoch_start_time = time.time()

                # Execute training epoch
                train_metrics = self._run_training_epoch(epoch)

                # Execute validation epoch if validation data provided
                val_metrics = {}
                if val_data is not None:
                    val_metrics = self._run_validation_epoch(epoch)
                    self.hooks.on_validation_end(val_metrics)

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                epoch_duration = time.time() - epoch_start_time

                # Log metrics
                self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_duration)

                # Checkpoint if needed
                if self._should_save_checkpoint(epoch):
                    self._save_checkpoint(epoch, epoch_metrics)

                self.hooks.on_epoch_end(epoch, epoch_metrics)

        finally:
            self.hooks.on_training_end()

        training_time = time.time() - training_start_time

        # Return comprehensive results
        return self._format_results(training_time)

    def _setup_checkpointing(self) -> CheckpointManager:
        """Initialize checkpoint manager from config."""
        return CheckpointManager(
            checkpoint_dir=self.training_config.checkpoint_dir,
            keep_best_k=3,  # Keep top 3 checkpoints
            keep_last_n=5,  # Keep last 5 checkpoints
            monitor='val_loss' if hasattr(self.training_config, 'monitor') else 'val_loss',
            mode='min',  # Minimize validation loss
            save_interval_epochs=self.training_config.save_every_n_epochs
        )

    def _setup_metrics(self) -> MetricsTracker:
        """Initialize metrics tracker from config."""
        return MetricsTracker(
            use_wandb=self.training_config.wandb_project is not None,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps
        )

    def _setup_loss_strategy(self) -> LossStrategy:
        """Initialize task-specific loss strategy."""
        if self.task_spec:
            # Map task_spec modality to loss strategy name
            modality_to_strategy = {
                'text': 'language_modeling',
                'vision': 'vision_classification',
                'multimodal': 'language_modeling',  # Default to LM for multimodal
            }
            strategy_name = modality_to_strategy.get(
                self.task_spec.modality,
                'language_modeling'  # Fallback
            )
        else:
            strategy_name = 'language_modeling'  # Default fallback

        # Note: Most loss strategies don't need config, so we don't pass it
        # For advanced strategies (PEFT, Quantization), they should be set up explicitly
        return get_loss_strategy(strategy_name)

    def _setup_gradient_monitor(self) -> GradientMonitor:
        """Initialize gradient monitoring."""
        return GradientMonitor(
            vanishing_threshold=1e-7,
            explosion_threshold=10.0,
            max_consecutive_failures=3,
            log_histogram=False  # Disable for performance
        )

    def _setup_gradient_accumulator(self) -> GradientAccumulator:
        """Initialize gradient accumulation manager."""
        return GradientAccumulator(
            optimizer=self.optimizer,
            accumulation_steps=self.training_config.gradient_accumulation_steps,
            max_grad_norm=self.training_config.max_grad_norm,
            batch_size=self.training_config.batch_size
        )

    def _setup_optimizer(self) -> Optimizer:
        """Initialize optimizer from config."""
        return AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )

    def _setup_scheduler(self) -> Optional[LRScheduler]:
        """Initialize learning rate scheduler if enabled."""
        # Check if LR scheduling is enabled (default: disabled for simplicity)
        use_lr_schedule = getattr(self.training_config, 'use_lr_schedule', False)

        if not use_lr_schedule:
            return None

        # Linear warmup + cosine decay (industry standard)
        from torch.optim.lr_scheduler import OneCycleLR

        # Estimate total steps (will be updated after data setup)
        total_steps = self.training_config.epochs * 100  # Rough estimate

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.training_config.learning_rate,
            total_steps=total_steps,
            pct_start=self.training_config.warmup_ratio,
            anneal_strategy='cos'
        )

        return scheduler

    def _setup_data(
        self,
        train_data: Union[Dataset, HFDataset, DataLoader],
        val_data: Optional[Union[Dataset, HFDataset, DataLoader]]
    ) -> None:
        """Setup data loaders with reproducible configuration."""
        # Use UniversalDataModule if task_spec provided
        if self.task_spec:
            self.data_module = UniversalDataModule(
                task_spec=self.task_spec,
                batch_size=self.training_config.batch_size,
                num_workers=0,  # Single-threaded for reproducibility
                seed=self.training_config.random_seed
            )
        else:
            # Fallback: create data loaders directly
            factory = DataLoaderFactory()

            if isinstance(train_data, DataLoader):
                self.train_loader = train_data
            else:
                train_config = DataLoaderConfig(
                    batch_size=self.training_config.batch_size,
                    num_workers=0,
                    shuffle=True,  # Shuffle training data
                    pin_memory=torch.cuda.is_available(),
                    seed=self.training_config.random_seed
                )
                self.train_loader = factory.create_dataloader(
                    train_data,
                    config=train_config,
                    task_spec=None
                )

            if val_data is not None:
                if isinstance(val_data, DataLoader):
                    self.val_loader = val_data
                else:
                    val_config = DataLoaderConfig(
                        batch_size=self.training_config.batch_size,
                        num_workers=0,
                        shuffle=False,  # Don't shuffle validation
                        pin_memory=torch.cuda.is_available(),
                        seed=self.training_config.random_seed
                    )
                    self.val_loader = factory.create_dataloader(
                        val_data,
                        config=val_config,
                        task_spec=None
                    )
            else:
                self.val_loader = None

        # Update scheduler with correct total steps
        if self.scheduler is not None:
            total_steps = self.training_config.epochs * len(self.train_loader)
            # Recreate scheduler with correct total steps
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.training_config.learning_rate,
                total_steps=total_steps,
                pct_start=self.training_config.warmup_ratio,
                anneal_strategy='cos'
            )

    def _run_training_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one training epoch.

        Delegates batch processing to gradient accumulator and loss strategy.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # Get train loader
        if self.data_module:
            train_loader = self.data_module.train_dataloader()
        else:
            train_loader = self.train_loader

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            elif isinstance(batch, (tuple, list)):
                batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x
                            for x in batch)

            # Forward pass
            outputs = self.model(**batch if isinstance(batch, dict) else {'input_ids': batch[0]})
            model_output = ModelOutput.from_raw(outputs)

            # Compute loss using strategy
            loss_inputs = self._prepare_loss_inputs(batch, model_output)
            loss = self.loss_strategy.compute_loss(loss_inputs)

            # Gradient accumulation handles: scaling, backward, clipping, optimizer step
            is_final_batch = (batch_idx == len(train_loader) - 1)
            did_step = self.gradient_accumulator.accumulate(
                loss=loss,
                model=self.model,
                is_final_batch=is_final_batch
            )

            # Scheduler step when optimizer stepped
            if did_step and self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()

            # Compute accuracy if possible
            if hasattr(model_output, 'logits') and 'labels' in batch:
                labels = batch['labels'] if isinstance(batch, dict) else batch[1]
                predictions = model_output.logits.argmax(dim=-1)
                mask = labels != -100  # Exclude padding
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()

            # Hook callback
            self.hooks.on_batch_end(batch_idx, loss.item())

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def _run_validation_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one validation epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # Get val loader
        if self.data_module:
            val_loader = self.data_module.val_dataloader()
        else:
            val_loader = self.val_loader

        if val_loader is None:
            return {}

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (tuple, list)):
                    batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x
                                for x in batch)

                # Forward pass
                outputs = self.model(**batch if isinstance(batch, dict) else {'input_ids': batch[0]})
                model_output = ModelOutput.from_raw(outputs)

                # Compute loss
                loss_inputs = self._prepare_loss_inputs(batch, model_output)
                loss = self.loss_strategy.compute_loss(loss_inputs)

                # Track metrics
                total_loss += loss.item()

                # Compute accuracy if possible
                if hasattr(model_output, 'logits') and 'labels' in batch:
                    labels = batch['labels'] if isinstance(batch, dict) else batch[1]
                    predictions = model_output.logits.argmax(dim=-1)
                    mask = labels != -100
                    correct = (predictions == labels) & mask
                    total_correct += correct.sum().item()
                    total_tokens += mask.sum().item()

        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        # Return with keys expected by MetricsTracker (without 'val_' prefix)
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def _prepare_loss_inputs(
        self,
        batch: Union[Dict, tuple],
        model_output: ModelOutput
    ) -> LossInputs:
        """
        Prepare inputs for loss computation.

        Args:
            batch: Input batch (dict or tuple)
            model_output: Parsed model output

        Returns:
            LossInputs dictionary
        """
        loss_inputs: LossInputs = {
            'logits': model_output.logits,
            'labels': batch['labels'] if isinstance(batch, dict) else batch[1]
        }

        # Add optional inputs if available
        if isinstance(batch, dict):
            if 'attention_mask' in batch:
                loss_inputs['attention_mask'] = batch['attention_mask']

        # Add pad_token_id from config if available
        if hasattr(self.config, 'pad_token_id'):
            loss_inputs['pad_token_id'] = self.config.pad_token_id

        return loss_inputs

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_duration: float
    ) -> None:
        """
        Log metrics for current epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch_duration: Epoch duration in seconds
        """
        # Get current learning rate
        lr = self.optimizer.param_groups[0]['lr']

        # Get gradient norm from last batch
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm = max(grad_norm, p.grad.norm().item())

        # Log to metrics tracker
        self.metrics_tracker.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics if val_metrics else {'loss': 0.0, 'accuracy': 0.0},
            learning_rate=lr,
            gradient_norm=grad_norm,
            epoch_duration=epoch_duration
        )

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Determine if checkpoint should be saved at this epoch."""
        return (epoch + 1) % self.training_config.save_every_n_epochs == 0

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save checkpoint with current training state.

        Args:
            epoch: Current epoch number
            metrics: Current metrics dictionary
        """
        # Add 'val_loss' to metrics for checkpoint manager (required for monitoring)
        checkpoint_metrics = metrics.copy()
        if 'val_loss' not in checkpoint_metrics:
            checkpoint_metrics['val_loss'] = metrics.get('loss', 0.0)

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=checkpoint_metrics,
            custom_state={
                'training_config': self.training_config.to_dict()
            }
        )

    def _resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch to resume from (checkpoint epoch + 1)
        """
        checkpoint = self.checkpoint_manager.load(Path(checkpoint_path))

        # Restore model and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Return next epoch to train
        return checkpoint['epoch'] + 1

    def _format_results(self, training_time: float) -> Dict[str, Any]:
        """
        Format comprehensive training results.

        Args:
            training_time: Total training time in seconds

        Returns:
            Dictionary with results summary
        """
        metrics_df = self.metrics_tracker.get_summary()

        results = {
            'metrics_summary': metrics_df,
            'best_epoch': self.metrics_tracker.get_best_epoch('val/loss', 'min') if 'val/loss' in metrics_df.columns else 0,
            'final_loss': metrics_df['train/loss'].iloc[-1] if not metrics_df.empty else 0.0,
            'checkpoint_path': str(self.checkpoint_manager.get_best()) if self.checkpoint_manager.get_best() else None,
            'training_time': training_time
        }

        logger.info(f"Training complete in {training_time:.1f}s")
        logger.info(f"Best model at epoch {results['best_epoch']}")

        return results


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'Trainer',
    'TrainingHooks',
    'DefaultHooks',
]
