"""
Gradient Accumulation Management

Provides unified gradient accumulation handling that works with both manual
accumulation and PyTorch Lightning's accumulate_grad_batches. Prevents
double accumulation and ensures correct step counting for metrics logging.

Features:
- Automatic Lightning detection and delegation
- Manual accumulation with proper loss scaling
- Step counting for MetricsTracker integration
- Conflict detection with clear error messages
- Zero overhead when accumulation_steps=1
- Type-safe implementation

Example:
    >>> # Manual accumulation
    >>> accumulator = GradientAccumulator(
    ...     optimizer=optimizer,
    ...     accumulation_steps=4,
    ...     max_grad_norm=1.0
    ... )
    >>>
    >>> for batch_idx, batch in enumerate(dataloader):
    ...     loss = compute_loss(model, batch)
    ...     # Returns True when optimizer should step
    ...     should_step = accumulator.accumulate(loss)
    ...
    ...     if should_step:
    ...         # GradientAccumulator handles:
    ...         # - Loss scaling (loss / accumulation_steps)
    ...         # - Gradient clipping
    ...         # - Optimizer step
    ...         # - Zero grad
    ...         pass
    ...
    ...     # Get effective step for logging
    ...     effective_step = accumulator.effective_step
    >>>
    >>> # With PyTorch Lightning (automatic delegation)
    >>> accumulator = GradientAccumulator(
    ...     optimizer=optimizer,
    ...     accumulation_steps=1,  # Disabled
    ...     trainer=lightning_trainer  # Has accumulate_grad_batches=4
    ... )
    >>> # accumulator.is_lightning_managed == True
    >>> # Manual accumulation is bypassed
"""

from dataclasses import dataclass
from typing import Optional, Any
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AccumulationStats:
    """
    Statistics for gradient accumulation cycle.

    Attributes:
        total_steps: Total number of backward() calls
        optimizer_steps: Number of optimizer.step() calls
        current_accumulation: Current position in accumulation window (0 to steps-1)
        effective_batch_size: Physical batch size * accumulation steps
        is_accumulating: True if currently accumulating gradients
        last_grad_norm: Last computed gradient norm (pre-clip)
    """
    total_steps: int
    optimizer_steps: int
    current_accumulation: int
    effective_batch_size: int
    is_accumulating: bool
    last_grad_norm: float

    def __str__(self) -> str:
        """Human-readable stats string."""
        return (
            f"Steps: {self.total_steps} | Optimizer updates: {self.optimizer_steps} | "
            f"Current: {self.current_accumulation} | Effective batch: {self.effective_batch_size}"
        )


class GradientAccumulator:
    """
    Unified gradient accumulation manager.

    Handles gradient accumulation for both manual and Lightning-managed training.
    Prevents double accumulation conflicts and ensures correct step counting.

    Attributes:
        optimizer: PyTorch optimizer instance
        accumulation_steps: Number of batches to accumulate before optimizer step
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
        scaler: Optional GradScaler for AMP training
        batch_size: Physical batch size (for effective_batch_size calculation)
        is_lightning_managed: True if Lightning trainer controls accumulation
        stats: Current accumulation statistics

    Example:
        >>> accumulator = GradientAccumulator(
        ...     optimizer=optimizer,
        ...     accumulation_steps=4,
        ...     max_grad_norm=1.0,
        ...     batch_size=8
        ... )
        >>>
        >>> for epoch in range(epochs):
        ...     for batch_idx, batch in enumerate(dataloader):
        ...         loss = model(batch)
        ...
        ...         # Accumulate gradients and check if optimizer should step
        ...         should_step = accumulator.accumulate(
        ...             loss=loss,
        ...             model=model,
        ...             is_final_batch=(batch_idx == len(dataloader) - 1)
        ...         )
        ...
        ...         if should_step:
        ...             # Get gradient norm for logging
        ...             grad_norm = accumulator.stats.last_grad_norm
        ...
        ...         # Log metrics at effective steps
        ...         metrics_tracker.log_scalar(
        ...             'train/loss',
        ...             loss.item(),
        ...             step=accumulator.effective_step
        ...         )
        ...
        ...     # Reset for next epoch
        ...     accumulator.reset_epoch()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        scaler: Optional[Any] = None,
        batch_size: int = 1,
        trainer: Optional[Any] = None
    ):
        """
        Initialize gradient accumulator.

        Args:
            optimizer: PyTorch optimizer for parameter updates
            accumulation_steps: Number of batches to accumulate (default: 1, no accumulation)
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0, None to disable)
            scaler: Optional GradScaler for AMP training (default: None)
            batch_size: Physical batch size for effective batch size calculation (default: 1)
            trainer: Optional PyTorch Lightning Trainer instance (default: None)

        Raises:
            ValueError: If both accumulation_steps > 1 and trainer.accumulate_grad_batches > 1
        """
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler
        self.batch_size = batch_size
        self._trainer = trainer

        # Step counters
        self._total_steps = 0
        self._optimizer_steps = 0
        self._accumulation_counter = 0
        self._last_grad_norm = 0.0

        # Lightning detection
        self._is_lightning_managed = self._detect_lightning_accumulation()

        # Validate configuration
        self._validate_configuration()

        # Log initialization
        if self._is_lightning_managed:
            logger.info(
                f"GradientAccumulator: Lightning-managed accumulation detected "
                f"(accumulate_grad_batches={self._get_lightning_accumulation()}). "
                f"Manual accumulation disabled."
            )
        elif accumulation_steps > 1:
            logger.info(
                f"GradientAccumulator: Manual accumulation enabled "
                f"(steps={accumulation_steps}, effective_batch={self.effective_batch_size})"
            )
        else:
            logger.debug("GradientAccumulator: No accumulation (steps=1)")

    def _detect_lightning_accumulation(self) -> bool:
        """
        Detect if PyTorch Lightning is managing gradient accumulation.

        Returns:
            True if Lightning trainer with accumulate_grad_batches > 1 is detected
        """
        if self._trainer is None:
            return False

        # Check if trainer is a Lightning Trainer instance
        # We use string comparison to avoid hard dependency on pytorch_lightning
        trainer_class_name = type(self._trainer).__name__
        if 'Trainer' not in trainer_class_name:
            return False

        # Check if trainer has accumulate_grad_batches attribute
        if not hasattr(self._trainer, 'accumulate_grad_batches'):
            return False

        # Lightning manages accumulation if accumulate_grad_batches > 1
        lightning_accum = self._get_lightning_accumulation()
        return lightning_accum > 1

    def _get_lightning_accumulation(self) -> int:
        """
        Get Lightning's accumulate_grad_batches value.

        Returns:
            accumulate_grad_batches value, or 1 if not available
        """
        if self._trainer is None or not hasattr(self._trainer, 'accumulate_grad_batches'):
            return 1

        accum = self._trainer.accumulate_grad_batches
        # Handle both int and dict configurations
        if isinstance(accum, dict):
            # Dict maps epoch to accumulation steps
            # Use the value for epoch 0 as default
            value = accum.get(0, 1)
            return int(value) if value is not None else 1
        return int(accum)

    def _validate_configuration(self) -> None:
        """
        Validate accumulation configuration.

        Raises:
            ValueError: If conflicting accumulation settings detected
        """
        # Check for double accumulation conflict
        if self._is_lightning_managed and self.accumulation_steps > 1:
            lightning_accum = self._get_lightning_accumulation()
            raise ValueError(
                f"Gradient accumulation conflict detected!\n"
                f"  Manual accumulation_steps: {self.accumulation_steps}\n"
                f"  Lightning accumulate_grad_batches: {lightning_accum}\n\n"
                f"Resolution options:\n"
                f"  1. Set accumulation_steps=1 (let Lightning manage accumulation)\n"
                f"  2. Set trainer.accumulate_grad_batches=1 (use manual accumulation)\n"
                f"  3. Remove trainer parameter (disable Lightning integration)\n\n"
                f"Recommended: Use Lightning's accumulate_grad_batches for cleaner integration."
            )

        # Validate accumulation_steps
        if self.accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {self.accumulation_steps}"
            )

        # Validate max_grad_norm
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be > 0 or None, got {self.max_grad_norm}"
            )

    @property
    def is_lightning_managed(self) -> bool:
        """True if Lightning trainer is managing gradient accumulation."""
        return self._is_lightning_managed

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = physical batch size * accumulation steps."""
        if self._is_lightning_managed:
            return self.batch_size * self._get_lightning_accumulation()
        return self.batch_size * self.accumulation_steps

    @property
    def effective_step(self) -> int:
        """
        Effective step for metrics logging.

        Returns the optimizer step count, which corresponds to the number
        of actual parameter updates. This is used by MetricsTracker to
        align metrics with optimizer updates.

        Returns:
            Number of optimizer.step() calls
        """
        return self._optimizer_steps

    @property
    def stats(self) -> AccumulationStats:
        """Current accumulation statistics."""
        return AccumulationStats(
            total_steps=self._total_steps,
            optimizer_steps=self._optimizer_steps,
            current_accumulation=self._accumulation_counter,
            effective_batch_size=self.effective_batch_size,
            is_accumulating=self._accumulation_counter > 0,
            last_grad_norm=self._last_grad_norm
        )

    def accumulate(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        is_final_batch: bool = False
    ) -> bool:
        """
        Accumulate gradients and optionally step optimizer.

        This method handles the complete gradient accumulation cycle:
        1. Scale loss by 1/accumulation_steps
        2. Compute gradients via loss.backward()
        3. Check if accumulation window is complete
        4. If complete: clip gradients, step optimizer, zero gradients
        5. Update step counters

        Args:
            loss: Loss tensor from forward pass (not scaled)
            model: Model to compute gradients for
            is_final_batch: True if this is the final batch in epoch

        Returns:
            True if optimizer.step() was called, False if still accumulating

        Example:
            >>> for batch_idx, batch in enumerate(dataloader):
            ...     loss = model(batch)
            ...     should_step = accumulator.accumulate(
            ...         loss=loss,
            ...         model=model,
            ...         is_final_batch=(batch_idx == len(dataloader) - 1)
            ...     )
            ...
            ...     if should_step:
            ...         print(f"Optimizer stepped at batch {batch_idx}")
        """
        # If Lightning-managed, bypass manual accumulation
        if self._is_lightning_managed:
            # Lightning handles everything, just track steps
            self._total_steps += 1
            # Lightning global_step is the effective step
            if self._trainer is not None and hasattr(self._trainer, 'global_step'):
                self._optimizer_steps = int(self._trainer.global_step)
            return True  # Lightning controls when optimizer steps

        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps

        # Compute gradients (AMP-aware)
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Update counters
        self._total_steps += 1
        self._accumulation_counter += 1

        # Check if we should step optimizer
        should_step = (
            self._accumulation_counter >= self.accumulation_steps or
            is_final_batch
        )

        if should_step:
            # Unscale gradients for clipping (AMP)
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Compute pre-clip gradient norm
            pre_clip_norm = self._compute_gradient_norm(model)
            self._last_grad_norm = pre_clip_norm

            # Clip gradients
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.max_grad_norm
                )

            # Optimizer step (AMP-aware with overflow check)
            if self.scaler is not None:
                # Check for gradient overflow
                if torch.isfinite(torch.tensor(pre_clip_norm)):
                    self.scaler.step(self.optimizer)
                else:
                    logger.warning(
                        f"Gradient overflow detected (norm={pre_clip_norm:.2e}). "
                        f"Skipping optimizer step."
                    )
                self.scaler.update()
            else:
                self.optimizer.step()

            # Zero gradients for next accumulation cycle
            self.optimizer.zero_grad()

            # Update optimizer step counter
            self._optimizer_steps += 1

            # Reset accumulation counter
            self._accumulation_counter = 0

            return True

        return False

    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """
        Compute total gradient norm across all parameters.

        Args:
            model: Model to compute gradient norm for

        Returns:
            Total gradient norm (L2 norm across all parameters)
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm: float = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        return float(total_norm ** 0.5)

    def reset_epoch(self) -> None:
        """
        Reset accumulation state for new epoch.

        Call this at the end of each epoch to ensure clean state.
        Clears accumulation counter but preserves total step counts.
        """
        if self._accumulation_counter > 0:
            logger.warning(
                f"Resetting accumulator with {self._accumulation_counter} "
                f"accumulated gradients. These gradients will be lost. "
                f"Consider calling accumulate() with is_final_batch=True."
            )
        self._accumulation_counter = 0

    def state_dict(self) -> dict[str, int | float | None]:
        """
        Get accumulator state for checkpointing.

        Returns:
            Dictionary with accumulator state
        """
        return {
            'total_steps': self._total_steps,
            'optimizer_steps': self._optimizer_steps,
            'accumulation_counter': self._accumulation_counter,
            'last_grad_norm': self._last_grad_norm
        }

    def load_state_dict(self, state_dict: dict[str, int | float | None]) -> None:
        """
        Load accumulator state from checkpoint.

        Args:
            state_dict: Dictionary with accumulator state
        """
        total_steps_val = state_dict.get('total_steps', 0)
        self._total_steps = int(total_steps_val) if isinstance(total_steps_val, (int, float)) else 0
        optimizer_steps_val = state_dict.get('optimizer_steps', 0)
        self._optimizer_steps = int(optimizer_steps_val) if isinstance(optimizer_steps_val, (int, float)) else 0
        accum_counter_val = state_dict.get('accumulation_counter', 0)
        self._accumulation_counter = int(accum_counter_val) if isinstance(accum_counter_val, (int, float)) else 0
        grad_norm_val = state_dict.get('last_grad_norm', 0.0)
        self._last_grad_norm = float(grad_norm_val) if isinstance(grad_norm_val, (int, float)) else 0.0

        logger.info(
            f"Loaded GradientAccumulator state: "
            f"total_steps={self._total_steps}, "
            f"optimizer_steps={self._optimizer_steps}"
        )

    def __str__(self) -> str:
        """Human-readable accumulator status."""
        if self._is_lightning_managed:
            return (
                f"GradientAccumulator(Lightning-managed, "
                f"accumulate_grad_batches={self._get_lightning_accumulation()})"
            )
        return (
            f"GradientAccumulator(steps={self.accumulation_steps}, "
            f"effective_batch={self.effective_batch_size}, "
            f"optimizer_steps={self._optimizer_steps})"
        )
