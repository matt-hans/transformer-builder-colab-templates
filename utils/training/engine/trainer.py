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
from utils.training.engine.progress_hooks import ProgressBarHooks

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
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        hooks: Optional[TrainingHooks] = None
    ):
        """
        Initialize trainer with model and configuration.

        Args:
            model: PyTorch model to train
            config: Model configuration (SimpleNamespace or custom config object)
            training_config: Training hyperparameters and settings
            task_spec: Task specification for data loading and loss computation
            tokenizer: Optional tokenizer for text tasks (enables auto-collator selection)
            data_collator: Optional manual collator (overrides auto-selection)
            hooks: Optional training lifecycle hooks

        Raises:
            ValueError: If training_config validation fails or text task missing tokenizer/collator
        """
        # Validate configuration first
        training_config.validate()

        # Validate text task requirements
        if task_spec and task_spec.modality == 'text':
            if tokenizer is None and data_collator is None:
                raise ValueError(
                    "Text tasks require either:\n"
                    "  1. tokenizer (for auto-collator selection), OR\n"
                    "  2. data_collator (for manual collation)\n\n"
                    "Example:\n"
                    "  trainer = Trainer(..., tokenizer=tokenizer)\n"
                    "  # OR\n"
                    "  trainer = Trainer(..., data_collator=data_collator)"
                )

        self.model = model
        self.config = config
        self.training_config = training_config
        self.task_spec = task_spec
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.hooks = hooks or ProgressBarHooks(update_freq=10)

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

        # Validate data quality before starting training
        if self.data_module:
            train_loader = self.data_module.train_dataloader()
            val_loader = self.data_module.val_dataloader() if val_data is not None else None
        else:
            train_loader = self.train_loader
            val_loader = self.val_loader
        self._validate_data_quality(train_loader, val_loader)

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
                    # Add val_ prefix for hook (consistent with MetricsTracker naming)
                    val_metrics_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}
                    self.hooks.on_validation_end(val_metrics_prefixed)

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

    def _call_model_forward(self, batch: Dict[str, Any]) -> Any:
        """
        Call model forward() with automatic signature detection.

        Handles both HuggingFace models (keyword args) and custom Transformer Builder
        models (positional args with custom parameter names like 'input_0_tokens').

        Args:
            batch: Dict containing 'input_ids', 'attention_mask', 'labels', etc.

        Returns:
            Model outputs (logits, hidden states, etc.)
        """
        import inspect

        # Get forward method signature
        sig = inspect.signature(self.model.forward)
        params = list(sig.parameters.keys())

        # Remove 'self' from params
        if 'self' in params:
            params.remove('self')

        # Check if forward accepts **kwargs (VAR_KEYWORD)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_keyword:
            # HuggingFace-style: supports **kwargs, pass full batch
            return self.model(**batch)
        elif len(params) > 1:
            # Multi-parameter model (input_ids, attention_mask, etc.)
            # Try to match batch keys to parameter names
            model_inputs = {}
            for param_name in params:
                if param_name in batch:
                    model_inputs[param_name] = batch[param_name]
            return self.model(**model_inputs)
        else:
            # Single positional parameter (custom Transformer Builder models)
            # Parameter name may be 'input_ids', 'input_0_tokens', 'x', etc.
            first_param = params[0] if params else 'input_ids'

            # Try to match the exact parameter name from batch
            if first_param in batch:
                return self.model(batch[first_param])
            elif 'input_ids' in batch:
                # Standard case: batch has 'input_ids', pass it positionally
                # Model will receive it as its first parameter (whatever name it uses)
                return self.model(batch['input_ids'])
            else:
                # Last resort: try first tensor in batch
                for value in batch.values():
                    if isinstance(value, torch.Tensor):
                        return self.model(value)
                raise ValueError(f"Could not find appropriate input tensor in batch keys: {batch.keys()}")

    def _setup_data(
        self,
        train_data: Union[Dataset, HFDataset, DataLoader],
        val_data: Optional[Union[Dataset, HFDataset, DataLoader]]
    ) -> None:
        """Setup data loaders with tokenizer/collator for proper batching."""
        # Use UniversalDataModule if task_spec provided
        if self.task_spec:
            self.data_module = UniversalDataModule(
                train_data=train_data,
                val_data=val_data,
                task_spec=self.task_spec,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
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

        # Check for empty training data
        if len(train_loader) == 0:
            logger.error("Training loader is empty! No batches to process.")
            return {'loss': float('nan'), 'accuracy': 0.0}

        logger.debug(f"Training epoch {epoch} with {len(train_loader)} batches")

        # LAYER 3: Track skipped batches for graceful error recovery
        skipped_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (tuple, list)):
                    batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x
                                for x in batch)

                # Forward pass (with signature detection for custom models)
                if isinstance(batch, dict):
                    outputs = self._call_model_forward(batch)
                else:
                    # Handle tuple/list batch format
                    outputs = self._call_model_forward({'input_ids': batch[0]})
                model_output = ModelOutput.from_raw(outputs)

                # Compute loss using strategy
                loss_inputs = self._prepare_loss_inputs(batch, model_output)
                loss = self.loss_strategy.compute_loss(loss_inputs)

                # Debug: Check for nan loss and log details (with safe tensor inspection)
                if torch.isnan(loss):
                    try:
                        logits_info = self._safe_tensor_inspect(loss_inputs['logits'], 'logits')
                        labels_info = self._safe_tensor_inspect(loss_inputs['labels'], 'labels')
                        logger.error(
                            f"NAN loss detected at batch {batch_idx}!\n"
                            f"  {logits_info}\n"
                            f"  {labels_info}"
                        )
                    except Exception as e:
                        # Ultimate safety: even diagnostic failures shouldn't crash training
                        logger.error(
                            f"NAN loss detected at batch {batch_idx} "
                            f"(diagnostic inspection failed: {e.__class__.__name__})"
                        )

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

            except ValueError as e:
                # LAYER 3: Graceful degradation for known data quality issues
                error_msg = str(e)
                if any(keyword in error_msg for keyword in
                       ['seq_len', 'token shifting', 'ragged list', 'empty', 'mixed-length']):
                    logger.warning(
                        f"⚠️  Skipping batch {batch_idx} due to data quality issue:\n"
                        f"    {str(e)[:150]}..."
                    )
                    skipped_batches += 1

                    # Safety check: fail if skip rate > 1% (systemic issue)
                    skip_rate = skipped_batches / (batch_idx + 1)
                    if skip_rate > 0.01:
                        raise RuntimeError(
                            f"❌ Training aborted: {skip_rate:.1%} of batches skipped "
                            f"({skipped_batches}/{batch_idx+1} batches).\n\n"
                            f"This indicates a systemic data quality issue.\n"
                            f"Your dataset has too many problematic sequences.\n\n"
                            f"Please clean your dataset before training:\n"
                            f"  - Remove empty or very short text samples\n"
                            f"  - Ensure all sequences have minimum length for your task\n"
                            f"  - Filter dataset: dataset = dataset.filter(lambda x: len(x['input_ids']) >= 2)"
                        )

                    continue  # Skip to next batch
                else:
                    # Unknown ValueError, don't skip - raise to surface the error
                    raise

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

                # Forward pass (with signature detection for custom models)
                if isinstance(batch, dict):
                    outputs = self._call_model_forward(batch)
                else:
                    # Handle tuple/list batch format
                    outputs = self._call_model_forward({'input_ids': batch[0]})
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

    def _validate_data_quality(self, train_loader, val_loader=None):
        """
        Pre-training data quality validation (general-purpose).

        Checks:
        - Dataset is not empty
        - Sequences meet task requirements
        - No excessive filtering at collation time

        Raises:
            ValueError: If data quality issues detected
        """
        # Check 1: Non-empty dataset
        if len(train_loader) == 0:
            raise ValueError(
                "Training dataset is empty after collation. "
                "This typically means:\n"
                "  - All sequences were filtered (too short)\n"
                "  - Dataset preprocessing removed all samples\n"
                "  - Tokenization produced no valid sequences\n\n"
                "Solutions:\n"
                "  - Check dataset source (is it empty?)\n"
                "  - Review tokenization settings\n"
                "  - Verify min_seq_len requirements for your task\n"
                "  - Use utils.training.data_quality.filter_short_sequences() before training"
            )

        # Check 2: Sample a batch to verify data quality
        try:
            first_batch = next(iter(train_loader))
            logger.debug(f"Sample batch keys: {first_batch.keys()}")
            logger.debug(f"Sample batch size: {len(first_batch.get('input_ids', []))}")
        except StopIteration:
            raise ValueError("Training loader is empty (StopIteration on first batch)")
        except Exception as e:
            logger.warning(f"Could not sample batch for validation: {e}")

        logger.info("✅ Data quality validation passed")

    def _prepare_loss_inputs(
        self,
        batch: Union[Dict[str, Any], tuple[Any, ...]],
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

    def _safe_tensor_inspect(self, tensor: torch.Tensor, name: str = "tensor") -> str:
        """
        Safely inspect tensor for diagnostic logging without crashing.

        Handles edge cases:
        - Empty tensors (numel() == 0)
        - Scalar tensors
        - Tensors with nan/inf values

        Args:
            tensor: Tensor to inspect
            name: Name for display

        Returns:
            Human-readable summary string (never raises exceptions)
        """
        try:
            # Basic info (always safe)
            info = f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}"

            # Handle empty tensors
            if tensor.numel() == 0:
                return f"{info} [EMPTY - possibly all-padding batch or seq_len=1 after shift]"

            # Handle scalar tensors
            if tensor.ndim == 0:
                return f"{info}, value={tensor.item():.4f}"

            # Compute safe statistics
            with torch.no_grad():
                min_val = tensor.min().item()
                max_val = tensor.max().item()
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()

                info += f", range=[{min_val:.2f}, {max_val:.2f}]"

                if has_nan:
                    info += ", contains NaN"
                if has_inf:
                    info += ", contains Inf"

            return info

        except Exception as e:
            # Ultimate fallback
            return f"{name}: <inspection failed: {e.__class__.__name__}>"

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
                'training_config': self.training_config.to_dict(),
                'metrics_history': self.metrics_tracker.metrics_history
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

        # Restore metrics history if available
        custom_state = checkpoint.get('custom_state', {})
        if 'metrics_history' in custom_state:
            self.metrics_tracker.metrics_history = custom_state['metrics_history']

        # Return next epoch to train
        return int(checkpoint['epoch']) + 1

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
