"""
Training and Validation Loop Execution

Provides modular, production-ready training loops with comprehensive error handling,
metrics tracking, and integration with Phase 0 components (LossStrategy, GradientAccumulator,
GradientMonitor).

Features:
- TrainingLoop: Single epoch training with all optimizations
- ValidationLoop: Validation epoch without gradient computation
- EpochResult: Structured metrics for epoch results
- Mixed precision training (torch.amp)
- Exception handling: OOM, NaN loss, keyboard interrupt
- Progress bar integration (tqdm)
- Type-safe implementation (mypy --strict compliant)

Architecture:
- Separates concerns: loop execution vs loss computation vs gradient handling
- Uses dependency injection for Phase 0 components
- Returns structured results for easy integration with metrics trackers
- Handles both text and vision tasks via LossStrategy

Example:
    >>> from utils.training.engine import (
    ...     TrainingLoop,
    ...     ValidationLoop,
    ...     LanguageModelingLoss,
    ...     GradientAccumulator,
    ...     GradientMonitor
    ... )
    >>>
    >>> # Setup components
    >>> loss_strategy = LanguageModelingLoss()
    >>> gradient_accumulator = GradientAccumulator(
    ...     optimizer=optimizer,
    ...     accumulation_steps=4,
    ...     max_grad_norm=1.0
    ... )
    >>> gradient_monitor = GradientMonitor(
    ...     vanishing_threshold=1e-7,
    ...     explosion_threshold=10.0
    ... )
    >>>
    >>> # Create loops
    >>> train_loop = TrainingLoop(
    ...     loss_strategy=loss_strategy,
    ...     gradient_accumulator=gradient_accumulator,
    ...     gradient_monitor=gradient_monitor,
    ...     use_amp=True,
    ...     device='cuda'
    ... )
    >>>
    >>> val_loop = ValidationLoop(
    ...     loss_strategy=loss_strategy,
    ...     device='cuda'
    ... )
    >>>
    >>> # Execute training
    >>> for epoch in range(10):
    ...     train_result = train_loop.train_epoch(
    ...         model=model,
    ...         dataloader=train_loader,
    ...         optimizer=optimizer,
    ...         scheduler=scheduler,
    ...         epoch=epoch
    ...     )
    ...
    ...     val_result = val_loop.validate_epoch(
    ...         model=model,
    ...         dataloader=val_loader,
    ...         epoch=epoch
    ...     )
    ...
    ...     print(f"Epoch {epoch}: train_loss={train_result.loss:.4f}, "
    ...           f"val_loss={val_result.loss:.4f}")
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import autocast, GradScaler

from utils.training.engine.loss import LossStrategy, LossInputs, ModelOutput
from utils.training.engine.gradient_monitor import GradientMonitor, GradientHealth
from utils.training.engine.gradient_accumulator import GradientAccumulator

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """
    Results from training or validation epoch.

    Comprehensive metrics structure for epoch results, suitable for
    integration with MetricsTracker or custom logging.

    Attributes:
        loss: Average loss across all batches
        accuracy: Average accuracy across all batches (0.0 to 1.0)
        metrics: Dictionary of additional metrics (e.g., 'train/loss', 'val/perplexity')
        duration: Total epoch duration in seconds
        batch_count: Number of batches processed
        gradient_norms: List of gradient norms per optimizer step (training only)
        loss_history: List of loss values per batch (for plotting)
        learning_rate: Final learning rate (training only)
        throughput: Samples per second (samples_processed / duration)
    """
    loss: float
    accuracy: float
    metrics: Dict[str, float]
    duration: float
    batch_count: int = 0
    gradient_norms: Optional[List[float]] = None
    loss_history: Optional[List[float]] = None
    learning_rate: Optional[float] = None
    throughput: Optional[float] = None

    def __str__(self) -> str:
        """Human-readable result string."""
        result = f"Loss: {self.loss:.4f} | Acc: {self.accuracy:.4f} | Duration: {self.duration:.1f}s"
        if self.throughput is not None:
            result += f" | Throughput: {self.throughput:.1f} samples/s"
        if self.learning_rate is not None:
            result += f" | LR: {self.learning_rate:.2e}"
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'duration': self.duration,
            'batch_count': self.batch_count,
            'learning_rate': self.learning_rate,
            'throughput': self.throughput,
            **self.metrics
        }


class TrainingLoop:
    """
    Execute one epoch of training with all optimizations.

    Handles batch iteration, loss computation, gradient accumulation,
    gradient monitoring, and mixed precision training. Integrates seamlessly
    with Phase 0 components via dependency injection.

    Attributes:
        loss_strategy: LossStrategy for task-specific loss computation
        gradient_accumulator: GradientAccumulator for gradient accumulation
        gradient_monitor: Optional GradientMonitor for gradient health checks
        use_amp: Enable mixed precision training (torch.amp)
        device: Device to run training on ('cuda' or 'cpu')
        progress_bar: Enable tqdm progress bar (default: True)
        scaler: GradScaler for AMP (created internally)

    Example:
        >>> loop = TrainingLoop(
        ...     loss_strategy=LanguageModelingLoss(),
        ...     gradient_accumulator=GradientAccumulator(optimizer, accumulation_steps=4),
        ...     gradient_monitor=GradientMonitor(),
        ...     use_amp=True,
        ...     device='cuda'
        ... )
        >>>
        >>> result = loop.train_epoch(
        ...     model=model,
        ...     dataloader=train_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=0
        ... )
        >>>
        >>> print(f"Train loss: {result.loss:.4f}")
        >>> print(f"Gradient norms: {result.gradient_norms[:5]}")
    """

    def __init__(
        self,
        loss_strategy: LossStrategy,
        gradient_accumulator: GradientAccumulator,
        gradient_monitor: Optional[GradientMonitor] = None,
        use_amp: bool = False,
        device: str = 'cuda',
        progress_bar: bool = True
    ):
        """
        Initialize training loop.

        Args:
            loss_strategy: Task-specific loss computation strategy
            gradient_accumulator: Gradient accumulation manager
            gradient_monitor: Optional gradient health monitor
            use_amp: Enable automatic mixed precision (default: False)
            device: Device for training ('cuda' or 'cpu')
            progress_bar: Enable tqdm progress bar (default: True)
        """
        self.loss_strategy = loss_strategy
        self.gradient_accumulator = gradient_accumulator
        self.gradient_monitor = gradient_monitor
        self.use_amp = use_amp
        self.device = device
        self.progress_bar = progress_bar

        # Setup AMP scaler
        self.scaler = GradScaler() if use_amp and device == 'cuda' else None

        # Configure accumulator with scaler
        if self.scaler is not None:
            self.gradient_accumulator.scaler = self.scaler

        logger.debug(
            f"TrainingLoop initialized: device={device}, amp={use_amp}, "
            f"accumulation_steps={gradient_accumulator.accumulation_steps}"
        )

    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        epoch: int = 0,
        metrics_tracker: Optional[Any] = None
    ) -> EpochResult:
        """
        Execute one training epoch.

        Performs complete training epoch with:
        - Batch iteration with progress bar
        - Forward pass with optional AMP
        - Loss computation via LossStrategy
        - Gradient accumulation via GradientAccumulator
        - Gradient health checks via GradientMonitor
        - Learning rate scheduling
        - Comprehensive metrics tracking

        Args:
            model: PyTorch model to train
            dataloader: Training data loader
            optimizer: Optimizer instance
            scheduler: Optional learning rate scheduler
            epoch: Current epoch number (for logging)
            metrics_tracker: Optional MetricsTracker for per-batch logging

        Returns:
            EpochResult with loss, accuracy, metrics, duration

        Raises:
            RuntimeError: If training becomes unstable (NaN loss, gradient explosion)
            KeyboardInterrupt: User interruption (re-raised for graceful shutdown)

        Example:
            >>> result = train_loop.train_epoch(
            ...     model=model,
            ...     dataloader=train_loader,
            ...     optimizer=optimizer,
            ...     scheduler=scheduler,
            ...     epoch=5
            ... )
            >>> print(f"Epoch 5 complete: {result}")
        """
        model.train()
        start_time = time.time()

        # Epoch metrics
        epoch_loss_sum = 0.0
        epoch_accuracy_sum = 0.0
        batch_count = 0
        samples_processed = 0
        gradient_norms: List[float] = []
        loss_history: List[float] = []

        # Progress bar setup
        pbar = None
        if self.progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    dataloader,
                    desc=f"Epoch {epoch}",
                    unit="batch",
                    leave=True
                )
            except ImportError:
                logger.debug("tqdm not available, progress bar disabled")

        iterator = pbar if pbar else dataloader

        try:
            for batch_idx, batch_data in enumerate(iterator):
                # Prepare batch (move to device)
                batch = self._prepare_batch(batch_data)
                current_batch_size = self._get_batch_size(batch)
                samples_processed += current_batch_size

                # Forward pass with loss computation
                try:
                    loss, accuracy = self._forward_pass(model, batch)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        raise RuntimeError(
                            f"Out of memory during forward pass at batch {batch_idx}.\n"
                            f"Batch size: {current_batch_size}\n"
                            f"Suggestions:\n"
                            f"  1. Reduce batch size (current: {current_batch_size})\n"
                            f"  2. Enable gradient accumulation\n"
                            f"  3. Use mixed precision (AMP)\n"
                            f"  4. Reduce model size or sequence length"
                        ) from e
                    raise

                # Check for NaN loss
                if torch.isnan(loss):
                    raise RuntimeError(
                        f"NaN loss detected at epoch {epoch}, batch {batch_idx}.\n"
                        f"Remediation steps:\n"
                        f"  1. Lower learning rate (try 0.1x current LR)\n"
                        f"  2. Enable gradient clipping\n"
                        f"  3. Check data for NaN values\n"
                        f"  4. Verify loss computation\n"
                        f"  5. Try mixed precision training"
                    )

                # Accumulate gradients (handles backward pass internally)
                is_final_batch = (batch_idx == len(dataloader) - 1)
                should_step = self.gradient_accumulator.accumulate(
                    loss=loss,
                    model=model,
                    is_final_batch=is_final_batch
                )

                # Optimizer step occurred
                if should_step:
                    # Gradient health check
                    if self.gradient_monitor is not None:
                        try:
                            health = self.gradient_monitor.check_gradients(model)
                            if not health.is_healthy:
                                self._handle_unhealthy_gradients(health, epoch, batch_idx)
                        except RuntimeError as e:
                            # Gradient monitor raises RuntimeError for consecutive failures
                            raise RuntimeError(
                                f"Training halted at epoch {epoch}, batch {batch_idx}.\n{str(e)}"
                            ) from e

                    # Get gradient norm from accumulator
                    grad_norm = self.gradient_accumulator.stats.last_grad_norm
                    gradient_norms.append(grad_norm)

                    # Step scheduler (after optimizer step)
                    if scheduler is not None:
                        scheduler.step()

                    # Log to metrics tracker (if provided)
                    if metrics_tracker is not None:
                        effective_step = self.gradient_accumulator.effective_step
                        try:
                            metrics_tracker.log_scalar('train/batch_loss', loss.item(), step=effective_step)
                            metrics_tracker.log_scalar('train/batch_accuracy', accuracy, step=effective_step)
                            metrics_tracker.log_scalar('train/gradient_norm', grad_norm, step=effective_step)
                            if scheduler is not None:
                                current_lr = scheduler.get_last_lr()[0]
                                metrics_tracker.log_scalar('train/learning_rate', current_lr, step=effective_step)
                        except Exception as e:
                            logger.debug(f"Failed to log metrics: {e}")

                # Track epoch metrics
                epoch_loss_sum += loss.item()
                epoch_accuracy_sum += accuracy
                batch_count += 1
                loss_history.append(loss.item())

                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{accuracy:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else "N/A"
                    })

        except KeyboardInterrupt:
            logger.warning(f"\nTraining interrupted by user at epoch {epoch}, batch {batch_idx}")
            raise

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"\n{str(e)}")
            raise

        finally:
            if pbar is not None:
                pbar.close()

        # Calculate epoch summary metrics
        duration = time.time() - start_time
        avg_loss = epoch_loss_sum / batch_count if batch_count > 0 else float('inf')
        avg_accuracy = epoch_accuracy_sum / batch_count if batch_count > 0 else 0.0
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
        final_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
        throughput = samples_processed / duration if duration > 0 else 0.0

        # Compute perplexity for language modeling
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float('inf')

        metrics = {
            'train/loss': avg_loss,
            'train/accuracy': avg_accuracy,
            'train/perplexity': perplexity,
            'train/grad_norm_avg': avg_grad_norm,
            'train/grad_norm_max': max(gradient_norms) if gradient_norms else 0.0,
            'train/grad_norm_min': min(gradient_norms) if gradient_norms else 0.0,
        }

        logger.info(
            f"Epoch {epoch} training complete: "
            f"loss={avg_loss:.4f}, acc={avg_accuracy:.4f}, "
            f"perplexity={perplexity:.2f}, duration={duration:.1f}s"
        )

        return EpochResult(
            loss=avg_loss,
            accuracy=avg_accuracy,
            metrics=metrics,
            duration=duration,
            batch_count=batch_count,
            gradient_norms=gradient_norms,
            loss_history=loss_history,
            learning_rate=final_lr,
            throughput=throughput
        )

    def _prepare_batch(self, batch_data: Any) -> Dict[str, Any]:
        """
        Prepare batch for training (move to device).

        Handles various batch formats:
        - Tuple from DataLoader: (input_tensor,) or (input, labels)
        - Dictionary: {'input_ids': tensor, 'labels': tensor, ...}
        - Raw tensor

        Args:
            batch_data: Raw batch from DataLoader

        Returns:
            Dictionary with standardized keys: 'input_ids', 'labels', etc.
        """
        # Tuple format (most common for TensorDataset)
        if isinstance(batch_data, (tuple, list)):
            if len(batch_data) == 1:
                # Single tensor: (input,)
                input_tensor = batch_data[0].to(self.device, non_blocking=True)
                return {
                    'input_ids': input_tensor,
                    'labels': input_tensor,  # For LM, input == labels
                    'pad_token_id': 0
                }
            elif len(batch_data) >= 2:
                # Multiple tensors: (input, labels, ...)
                input_tensor = batch_data[0].to(self.device, non_blocking=True)
                labels = batch_data[1].to(self.device, non_blocking=True)
                return {
                    'input_ids': input_tensor,
                    'labels': labels,
                    'pad_token_id': 0
                }

        # Dictionary format (HuggingFace style)
        if isinstance(batch_data, dict):
            result: Dict[str, Any] = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device, non_blocking=True)
                else:
                    result[key] = value
            return result

        # Raw tensor
        if isinstance(batch_data, torch.Tensor):
            input_tensor = batch_data.to(self.device, non_blocking=True)
            return {
                'input_ids': input_tensor,
                'labels': input_tensor,
                'pad_token_id': 0
            }

        raise TypeError(
            f"Unsupported batch type: {type(batch_data)}. "
            f"Expected tuple, list, dict, or tensor."
        )

    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Extract batch size from batch dictionary."""
        if 'input_ids' in batch:
            return int(batch['input_ids'].size(0))
        elif 'pixel_values' in batch:
            return int(batch['pixel_values'].size(0))
        # Fallback: find first tensor and get batch size
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
        return 1

    def _forward_pass(self, model: nn.Module, batch: Dict[str, Any]) -> tuple[torch.Tensor, float]:
        """
        Forward pass with loss computation and accuracy.

        Args:
            model: PyTorch model
            batch: Prepared batch dictionary

        Returns:
            Tuple of (loss_tensor, accuracy_float)
        """
        # Extract pad_token_id with proper type
        pad_token_id: int = 0
        if 'pad_token_id' in batch:
            pad_val = batch['pad_token_id']
            if isinstance(pad_val, int):
                pad_token_id = pad_val
            elif isinstance(pad_val, torch.Tensor):
                pad_token_id = int(pad_val.item())

        # Forward pass with optional AMP
        if self.use_amp and self.device == 'cuda':
            with autocast():
                logits = self._get_model_output(model, batch['input_ids'])
                loss_inputs_amp: LossInputs = {
                    'logits': logits,
                    'labels': batch['labels'],
                    'attention_mask': batch.get('attention_mask'),
                    'pad_token_id': pad_token_id
                }
                loss = self.loss_strategy.compute_loss(loss_inputs_amp)
        else:
            logits = self._get_model_output(model, batch['input_ids'])
            loss_inputs_std: LossInputs = {
                'logits': logits,
                'labels': batch['labels'],
                'attention_mask': batch.get('attention_mask'),
                'pad_token_id': pad_token_id
            }
            loss = self.loss_strategy.compute_loss(loss_inputs_std)

        # Compute accuracy (outside autocast for numerical stability)
        with torch.no_grad():
            accuracy = self._compute_accuracy(logits, batch['labels'], pad_token_id)

        return loss, accuracy

    def _get_model_output(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get model output (logits) handling various output formats.

        Supports:
        - Raw tensor output
        - Tuple output (logits, loss, ...)
        - Dict output {'logits': tensor}
        - HuggingFace ModelOutput objects

        Args:
            model: PyTorch model
            input_ids: Input tensor

        Returns:
            Logits tensor
        """
        output = model(input_ids)

        # Parse using ModelOutput.from_raw
        try:
            parsed = ModelOutput.from_raw(output)
            return parsed.logits
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to parse model output: {e}. Using fallback.")
            # Fallback: assume raw tensor
            if isinstance(output, torch.Tensor):
                return output
            raise

    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int = 0
    ) -> float:
        """
        Compute accuracy excluding padding tokens.

        Args:
            logits: Model logits [batch, seq_len, vocab_size] or [batch, num_classes]
            labels: Target labels [batch, seq_len] or [batch]
            pad_token_id: Token ID to exclude from accuracy calculation

        Returns:
            Accuracy as float (0.0 to 1.0)
        """
        # Get predictions
        if logits.ndim == 3:
            # Sequence modeling: [batch, seq, vocab]
            preds = logits.argmax(dim=-1)  # [batch, seq]
        elif logits.ndim == 2:
            # Classification: [batch, classes]
            preds = logits.argmax(dim=-1)  # [batch]
        else:
            logger.warning(f"Unexpected logits shape: {logits.shape}")
            return 0.0

        # Create mask for non-padding tokens
        if labels.ndim == preds.ndim:
            # Same shape, create mask
            mask = labels != pad_token_id
            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item()
        else:
            # Shape mismatch, compute without mask
            correct = (preds == labels).sum().item()
            total = labels.numel()

        return correct / total if total > 0 else 0.0

    def _handle_unhealthy_gradients(
        self,
        health: GradientHealth,
        epoch: int,
        batch_idx: int
    ) -> None:
        """
        Handle unhealthy gradient detection.

        Logs warnings for vanishing/exploding gradients.
        Raises RuntimeError for NaN/Inf (handled by gradient_monitor).

        Args:
            health: GradientHealth status
            epoch: Current epoch
            batch_idx: Current batch index
        """
        if health.vanishing_layers:
            logger.warning(
                f"Epoch {epoch}, batch {batch_idx}: "
                f"Vanishing gradients in {len(health.vanishing_layers)} layers. "
                f"Examples: {health.vanishing_layers[:3]}"
            )

        if health.exploding_layers:
            logger.warning(
                f"Epoch {epoch}, batch {batch_idx}: "
                f"Exploding gradients in {len(health.exploding_layers)} layers. "
                f"Examples: {health.exploding_layers[:3]}"
            )

        if health.has_nan:
            logger.error(
                f"Epoch {epoch}, batch {batch_idx}: "
                f"NaN gradients in {len([l for l in health.affected_layers if 'NaN' in l])} layers. "
                f"Affected: {health.affected_layers[:5]}"
            )

        if health.has_inf:
            logger.error(
                f"Epoch {epoch}, batch {batch_idx}: "
                f"Inf gradients in {len([l for l in health.affected_layers if 'Inf' in l])} layers. "
                f"Affected: {health.affected_layers[:5]}"
            )


class ValidationLoop:
    """
    Execute validation epoch without gradient computation.

    Performs evaluation with no_grad context for efficiency.
    Uses same LossStrategy as training for consistency.

    Attributes:
        loss_strategy: Task-specific loss computation strategy
        device: Device for validation ('cuda' or 'cpu')
        progress_bar: Enable tqdm progress bar (default: True)

    Example:
        >>> val_loop = ValidationLoop(
        ...     loss_strategy=LanguageModelingLoss(),
        ...     device='cuda'
        ... )
        >>>
        >>> result = val_loop.validate_epoch(
        ...     model=model,
        ...     dataloader=val_loader,
        ...     epoch=5
        ... )
        >>>
        >>> print(f"Val loss: {result.loss:.4f}, perplexity: {result.metrics['val/perplexity']:.2f}")
    """

    def __init__(
        self,
        loss_strategy: LossStrategy,
        device: str = 'cuda',
        progress_bar: bool = True
    ):
        """
        Initialize validation loop.

        Args:
            loss_strategy: Task-specific loss computation strategy
            device: Device for validation ('cuda' or 'cpu')
            progress_bar: Enable tqdm progress bar (default: True)
        """
        self.loss_strategy = loss_strategy
        self.device = device
        self.progress_bar = progress_bar

        logger.debug(f"ValidationLoop initialized: device={device}")

    def validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        epoch: int = 0,
        metrics_tracker: Optional[Any] = None
    ) -> EpochResult:
        """
        Execute validation epoch.

        Performs complete validation with:
        - No gradient computation (eval mode + torch.no_grad)
        - Batch iteration with progress bar
        - Loss and accuracy computation
        - Comprehensive metrics tracking

        Args:
            model: PyTorch model to validate
            dataloader: Validation data loader
            epoch: Current epoch number (for logging)
            metrics_tracker: Optional MetricsTracker for logging

        Returns:
            EpochResult with loss, accuracy, metrics, duration

        Example:
            >>> result = val_loop.validate_epoch(
            ...     model=model,
            ...     dataloader=val_loader,
            ...     epoch=5
            ... )
            >>> print(f"Validation perplexity: {result.metrics['val/perplexity']:.2f}")
        """
        model.eval()
        start_time = time.time()

        # Epoch metrics
        epoch_loss_sum = 0.0
        epoch_accuracy_sum = 0.0
        batch_count = 0
        samples_processed = 0
        loss_history: List[float] = []

        # Progress bar setup
        pbar = None
        if self.progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    dataloader,
                    desc=f"Validation (Epoch {epoch})",
                    unit="batch",
                    leave=False
                )
            except ImportError:
                logger.debug("tqdm not available, progress bar disabled")

        iterator = pbar if pbar else dataloader

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(iterator):
                # Prepare batch
                batch = self._prepare_batch(batch_data)
                current_batch_size = self._get_batch_size(batch)
                samples_processed += current_batch_size

                # Extract pad_token_id with proper type
                pad_token_id: int = 0
                if 'pad_token_id' in batch:
                    pad_val = batch['pad_token_id']
                    if isinstance(pad_val, int):
                        pad_token_id = pad_val
                    elif isinstance(pad_val, torch.Tensor):
                        pad_token_id = int(pad_val.item())

                # Forward pass
                logits = self._get_model_output(model, batch['input_ids'])
                loss_inputs: LossInputs = {
                    'logits': logits,
                    'labels': batch['labels'],
                    'attention_mask': batch.get('attention_mask'),
                    'pad_token_id': pad_token_id
                }
                loss = self.loss_strategy.compute_loss(loss_inputs)

                # Compute accuracy
                accuracy = self._compute_accuracy(logits, batch['labels'], pad_token_id)

                # Track metrics
                epoch_loss_sum += loss.item()
                epoch_accuracy_sum += accuracy
                batch_count += 1
                loss_history.append(loss.item())

                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{accuracy:.4f}"
                    })

        if pbar is not None:
            pbar.close()

        # Calculate summary metrics
        duration = time.time() - start_time
        avg_loss = epoch_loss_sum / batch_count if batch_count > 0 else float('inf')
        avg_accuracy = epoch_accuracy_sum / batch_count if batch_count > 0 else 0.0
        throughput = samples_processed / duration if duration > 0 else 0.0

        # Compute perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float('inf')

        metrics = {
            'val/loss': avg_loss,
            'val/accuracy': avg_accuracy,
            'val/perplexity': perplexity
        }

        logger.info(
            f"Epoch {epoch} validation complete: "
            f"loss={avg_loss:.4f}, acc={avg_accuracy:.4f}, "
            f"perplexity={perplexity:.2f}, duration={duration:.1f}s"
        )

        return EpochResult(
            loss=avg_loss,
            accuracy=avg_accuracy,
            metrics=metrics,
            duration=duration,
            batch_count=batch_count,
            gradient_norms=None,  # No gradients in validation
            loss_history=loss_history,
            learning_rate=None,
            throughput=throughput
        )

    def _prepare_batch(self, batch_data: Any) -> Dict[str, Any]:
        """Prepare batch for validation (same as training)."""
        # Reuse training logic
        if isinstance(batch_data, (tuple, list)):
            if len(batch_data) == 1:
                input_tensor = batch_data[0].to(self.device, non_blocking=True)
                return {
                    'input_ids': input_tensor,
                    'labels': input_tensor,
                    'pad_token_id': 0
                }
            elif len(batch_data) >= 2:
                input_tensor = batch_data[0].to(self.device, non_blocking=True)
                labels = batch_data[1].to(self.device, non_blocking=True)
                return {
                    'input_ids': input_tensor,
                    'labels': labels,
                    'pad_token_id': 0
                }

        if isinstance(batch_data, dict):
            result: Dict[str, Any] = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device, non_blocking=True)
                else:
                    result[key] = value
            return result

        if isinstance(batch_data, torch.Tensor):
            input_tensor = batch_data.to(self.device, non_blocking=True)
            return {
                'input_ids': input_tensor,
                'labels': input_tensor,
                'pad_token_id': 0
            }

        raise TypeError(
            f"Unsupported batch type: {type(batch_data)}. "
            f"Expected tuple, list, dict, or tensor."
        )

    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Extract batch size from batch dictionary."""
        if 'input_ids' in batch:
            return int(batch['input_ids'].size(0))
        elif 'pixel_values' in batch:
            return int(batch['pixel_values'].size(0))
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return int(value.size(0))
        return 1

    def _get_model_output(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Get model output (logits) handling various formats."""
        output = model(input_ids)

        try:
            parsed = ModelOutput.from_raw(output)
            return parsed.logits
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to parse model output: {e}. Using fallback.")
            if isinstance(output, torch.Tensor):
                return output
            raise

    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int = 0
    ) -> float:
        """Compute accuracy excluding padding tokens."""
        # Get predictions
        if logits.ndim == 3:
            preds = logits.argmax(dim=-1)
        elif logits.ndim == 2:
            preds = logits.argmax(dim=-1)
        else:
            logger.warning(f"Unexpected logits shape: {logits.shape}")
            return 0.0

        # Create mask
        if labels.ndim == preds.ndim:
            mask = labels != pad_token_id
            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item()
        else:
            correct = (preds == labels).sum().item()
            total = labels.numel()

        return correct / total if total > 0 else 0.0
