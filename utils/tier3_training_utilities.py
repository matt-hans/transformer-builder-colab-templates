"""
Tier 3: Training Utilities

This module contains training-focused utilities for transformer models:
- Fine-tuning loop with loss tracking and gradient monitoring
- Hyperparameter optimization using Optuna
- Benchmark comparison against baseline models
- AMP (Automatic Mixed Precision) training support

These utilities are useful for training workflows and model optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import time
import numpy as np

# Import AMP utilities for mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Import DataLoader utilities for efficient data loading
from torch.utils.data import TensorDataset, DataLoader

# Import AMP benchmark from dedicated module
from utils.training.amp_benchmark import test_amp_speedup_benchmark

# Import benchmark utilities from dedicated module
from utils.training.benchmark_utils import (
    load_baseline_model,
    benchmark_inference_speed,
    compute_model_perplexity,
    create_benchmark_visualization
)

# Import environment snapshot utilities for reproducibility
from utils.training.environment_snapshot import (
    capture_environment,
    save_environment_snapshot,
    log_environment_to_wandb
)

# Re-export for backward compatibility
__all__ = ['test_fine_tuning', 'test_hyperparameter_search', 'test_benchmark_comparison', 'test_amp_speedup_benchmark']


def _detect_vocab_size(model: nn.Module, config: Any) -> int:
    """
    Detect vocabulary size from model or config.

    Priority:
    1. config.vocab_size (explicit)
    2. model embedding layer vocab size (introspection)
    3. Default fallback (50257 for GPT-2 compatibility)
    """
    # Try config first
    if hasattr(config, 'vocab_size') and config.vocab_size is not None:
        return config.vocab_size

    # Try to detect from model embedding layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module.num_embeddings

    # Fallback with warning
    print("⚠️ Could not detect vocab_size, using default 50257 (GPT-2)")
    return 50257


def _extract_output_tensor(output: Any) -> torch.Tensor:
    """
    Extract tensor from various model output formats.

    Handles:
    - Direct tensor: return as-is
    - Tuple: return first element
    - Dict: return output['logits'] or output['last_hidden_state']
    - ModelOutput object: return .logits attribute
    """
    # Direct tensor
    if isinstance(output, torch.Tensor):
        return output

    # Tuple (common for models that return multiple outputs)
    if isinstance(output, tuple):
        return output[0]

    # Dict
    if isinstance(output, dict):
        if 'logits' in output:
            return output['logits']
        if 'last_hidden_state' in output:
            return output['last_hidden_state']
        # Return first tensor value found
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value

    # HuggingFace ModelOutput object
    if hasattr(output, 'logits'):
        return output.logits
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state

    # Fallback - assume it's tensor-like
    return output


def _safe_get_model_output(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Safely extract logits tensor from model output.

    Wraps model() call and handles diverse output formats.
    """
    output = model(input_ids)
    return _extract_output_tensor(output)


def _compute_loss_and_backward(
    model: nn.Module,
    batch: torch.Tensor,
    scaler: Optional[Any],
    use_amp: bool,
    vocab_size: int,
    metrics_tracker: Any,
    gradient_accumulation_steps: int
) -> tuple:
    """
    Compute loss, backward pass with gradient accumulation scaling.

    This function only computes loss and accumulates gradients. It does NOT
    call optimizer.step() or scheduler.step() - that's handled by the caller
    based on accumulation logic.

    Args:
        model: The model to train
        batch: Input batch tensor
        scaler: GradScaler for AMP (None if use_amp=False)
        use_amp: Whether to use automatic mixed precision
        vocab_size: Vocabulary size for loss computation
        metrics_tracker: Metrics tracking instance
        gradient_accumulation_steps: Number of steps to accumulate gradients over

    Returns:
        Tuple of (loss_value, accuracy) where loss_value is the unscaled loss
    """
    # Forward pass with optional autocast
    if use_amp:
        with autocast():
            logits = _safe_get_model_output(model, batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Scale loss by accumulation steps to get correct gradient magnitude
            scaled_loss = loss / gradient_accumulation_steps

        # Compute accuracy outside autocast (FP32)
        with torch.no_grad():
            accuracy = metrics_tracker.compute_accuracy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
    else:
        # Standard FP32 forward pass
        logits = _safe_get_model_output(model, batch)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )

        # Scale loss by accumulation steps
        scaled_loss = loss / gradient_accumulation_steps

        accuracy = metrics_tracker.compute_accuracy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )

    # Backward pass with optional gradient scaling
    if use_amp:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    # Return unscaled loss for logging
    return loss.item(), accuracy


def _setup_training_environment(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]],
    val_data: Optional[List[torch.Tensor]],
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    use_amp: bool,
    use_wandb: bool
) -> Dict[str, Any]:
    """
    Setup training environment: data, optimizer, scheduler, scaler, metrics tracker.

    Returns:
        Dictionary with all training components
    """
    from utils.training.metrics_tracker import MetricsTracker

    device = next(model.parameters()).device
    vocab_size = _detect_vocab_size(model, config)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() if (use_amp and torch.cuda.is_available()) else None
    if use_amp and not torch.cuda.is_available():
        print("⚠️ AMP requested but CUDA not available, falling back to FP32")
        use_amp = False

    # Generate synthetic training data if not provided
    if train_data is None:
        print("Generating synthetic training data...")
        train_data = [torch.randint(0, vocab_size, (32,)) for _ in range(50)]

    # Create validation split if not provided
    if val_data is None:
        split_idx = int(0.8 * len(train_data))
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]

    # Create DataLoaders for efficient async data loading
    # Note: Use num_workers=0 for test environments to avoid multiprocessing issues
    use_workers = torch.cuda.is_available()  # Only use workers on GPU
    num_workers = 2 if use_workers else 0

    train_dataset = TensorDataset(torch.stack(train_data))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster CPU->GPU transfer
        prefetch_factor=2 if use_workers else None,  # Pre-load 2 batches
        persistent_workers=use_workers
    )

    val_dataset = TensorDataset(torch.stack(val_data))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if use_workers else None,
        persistent_workers=use_workers
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(use_wandb=use_wandb)

    return {
        'device': device,
        'vocab_size': vocab_size,
        'scaler': scaler,
        'use_amp': use_amp,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'metrics_tracker': metrics_tracker
    }


def _run_training_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    use_amp: bool,
    vocab_size: int,
    metrics_tracker: Any,
    device: torch.device,
    gradient_accumulation_steps: int = 1
) -> Dict[str, Any]:
    """
    Execute one training epoch with gradient accumulation support.

    Args:
        model: Model to train
        train_loader: DataLoader with training batches
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        scaler: GradScaler for AMP
        use_amp: Whether to use automatic mixed precision
        vocab_size: Vocabulary size
        metrics_tracker: Metrics tracking instance
        device: Device to run on
        gradient_accumulation_steps: Number of batches to accumulate before optimizer step

    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    train_loss_sum = 0.0
    train_acc_sum = 0.0
    train_steps = 0
    max_grad_norm = 0.0
    loss_history = []
    grad_norm_history = []

    # Initialize gradient accumulation
    optimizer.zero_grad()
    accumulation_counter = 0

    # Iterate through DataLoader (shuffling handled by DataLoader)
    for batch_idx, batch_tuple in enumerate(train_loader):
        # Extract batch from DataLoader tuple
        batch = batch_tuple[0].to(device, non_blocking=True)

        # Compute loss and accumulate gradients
        loss_value, accuracy = _compute_loss_and_backward(
            model=model,
            batch=batch,
            scaler=scaler,
            use_amp=use_amp,
            vocab_size=vocab_size,
            metrics_tracker=metrics_tracker,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        accumulation_counter += 1
        train_loss_sum += loss_value
        train_acc_sum += accuracy
        train_steps += 1
        loss_history.append(loss_value)

        # Check if we should update weights
        should_update = (accumulation_counter == gradient_accumulation_steps) or \
                       (batch_idx + 1 == len(train_loader))

        if should_update:
            # Clip gradients and compute norm
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check for inf/nan gradients before optimizer step
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Skip optimizer step on non-finite gradients
                    metrics_tracker.log_scalar('train/gradient_overflow', 1.0)
                    scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Step scheduler (once per optimizer step, not per batch)
            scheduler.step()

            # Zero gradients for next accumulation
            optimizer.zero_grad()
            accumulation_counter = 0

            max_grad_norm = max(max_grad_norm, grad_norm.item())
            grad_norm_history.append(grad_norm.item())

    return {
        'train_loss': train_loss_sum / train_steps,
        'train_accuracy': train_acc_sum / train_steps,
        'max_grad_norm': max_grad_norm,
        'loss_history': loss_history,
        'grad_norm_history': grad_norm_history
    }


def _run_validation_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    vocab_size: int,
    metrics_tracker: Any,
    device: torch.device
) -> Dict[str, float]:
    """
    Execute validation epoch using DataLoader for efficient async data loading.

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    val_loss_sum = 0.0
    val_acc_sum = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch_tuple in val_loader:
            # Extract batch from DataLoader tuple
            val_batch = batch_tuple[0].to(device, non_blocking=True)

            logits = _safe_get_model_output(model, val_batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = val_batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            accuracy = metrics_tracker.compute_accuracy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            val_loss_sum += loss.item()
            val_acc_sum += accuracy
            val_steps += 1

    return {
        'val_loss': val_loss_sum / val_steps,
        'val_accuracy': val_acc_sum / val_steps
    }


def _create_training_visualization(
    loss_history: List[float],
    grad_norm_history: List[float],
    metrics_summary: Any,
    n_epochs: int,
    batch_size: int,
    train_data_size: int
):
    """Create training visualization plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curve (step-level)
    axes[0, 0].plot(loss_history, linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Curve (Step-Level)')
    axes[0, 0].grid(True, alpha=0.3)

    # Add epoch markers
    steps_per_epoch = train_data_size // batch_size
    for e in range(1, n_epochs):
        axes[0, 0].axvline(
            x=e * steps_per_epoch, color='r',
            linestyle='--', alpha=0.5, linewidth=1
        )

    # Epoch-level metrics (train vs val loss)
    axes[0, 1].plot(
        metrics_summary['epoch'], metrics_summary['train/loss'],
        marker='o', label='Train Loss', linewidth=2
    )
    axes[0, 1].plot(
        metrics_summary['epoch'], metrics_summary['val/loss'],
        marker='s', label='Val Loss', linewidth=2
    )
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Train vs Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gradient norm
    axes[1, 0].plot(
        grad_norm_history, linewidth=2, alpha=0.7, color='orange'
    )
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norm (after clipping)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(
        y=1.0, color='r', linestyle='--',
        linewidth=1, label='Clip threshold'
    )
    axes[1, 0].legend()

    # Perplexity
    axes[1, 1].plot(
        metrics_summary['epoch'], metrics_summary['train/perplexity'],
        marker='o', label='Train PPL', linewidth=2
    )
    axes[1, 1].plot(
        metrics_summary['epoch'], metrics_summary['val/perplexity'],
        marker='s', label='Val PPL', linewidth=2
    )
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Perplexity')
    axes[1, 1].set_title('Train vs Validation Perplexity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_fine_tuning(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    val_data: Optional[List[torch.Tensor]] = None,
    n_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    use_wandb: bool = False,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1
) -> Dict[str, Any]:
    """
    Run a basic fine-tuning loop with comprehensive metrics tracking.

    Demonstrates:
    - Training loop setup with train/validation splits
    - Gradient clipping and monitoring
    - Learning rate scheduling
    - W&B metrics logging (loss, perplexity, accuracy, LR, gradient norms)
    - System metrics (GPU memory/utilization)
    - Loss convergence tracking
    - Mixed precision training with PyTorch AMP (optional)
    - Gradient accumulation for simulating larger batch sizes

    Args:
        model: The transformer model to fine-tune
        config: Model configuration
        train_data: List of input_ids tensors (if None, generates synthetic data)
        val_data: List of validation input_ids tensors (if None, uses 20% of train)
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Physical batch size (loaded into GPU memory)
        use_wandb: Whether to log metrics to W&B (default: False)
        use_amp: Whether to use Automatic Mixed Precision (FP16) for faster training (default: False)
        gradient_accumulation_steps: Number of batches to accumulate gradients over before
            updating weights. Effective batch size = batch_size * gradient_accumulation_steps.
            Default: 1 (no accumulation, update every batch)

    Returns:
        Dictionary with training metrics, loss curves, and MetricsTracker summary
    """
    # Setup training environment
    env = _setup_training_environment(
        model, config, train_data, val_data, n_epochs, learning_rate, batch_size, use_amp, use_wandb
    )

    # Compute effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Print training configuration
    print("=" * 60)
    print("FINE-TUNING TEST")
    print("=" * 60)
    print(f"Training samples: {len(env['train_loader'].dataset)}")
    print(f"Validation samples: {len(env['val_loader'].dataset)}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"W&B logging: {use_wandb}")
    print(f"Mixed precision (AMP): {env['use_amp']}")
    print(f"Device: {env['device']}")
    print("-" * 60)

    # Capture environment snapshot for reproducibility
    print("📸 Capturing environment snapshot...")
    try:
        env_info = capture_environment()
        req_path, env_path, repro_path = save_environment_snapshot(env_info, "./environment")

        # Log to W&B if enabled
        if use_wandb:
            try:
                log_environment_to_wandb(req_path, env_path, repro_path, env_info)
            except Exception as e:
                print(f"⚠️ Failed to log environment to W&B: {e}")
    except Exception as e:
        print(f"⚠️ Failed to capture environment snapshot: {e}")
        print("   Training will continue without environment snapshot")

    print("-" * 60)

    all_loss_history = []
    all_grad_norm_history = []
    start_time = time.time()

    # Training loop
    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        # Run training epoch
        train_results = _run_training_epoch(
            model, env['train_loader'], env['optimizer'], env['scheduler'],
            env['scaler'], env['use_amp'], env['vocab_size'], env['metrics_tracker'], env['device'],
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Run validation epoch
        val_results = _run_validation_epoch(
            model, env['val_loader'], env['vocab_size'], env['metrics_tracker'], env['device']
        )

        # Log metrics
        epoch_duration = time.time() - epoch_start_time
        current_lr = env['scheduler'].get_last_lr()[0]

        env['metrics_tracker'].log_epoch(
            epoch=epoch,
            train_metrics={'loss': train_results['train_loss'], 'accuracy': train_results['train_accuracy']},
            val_metrics={'loss': val_results['val_loss'], 'accuracy': val_results['val_accuracy']},
            learning_rate=current_lr,
            gradient_norm=train_results['max_grad_norm'],
            epoch_duration=epoch_duration
        )

        # Log training configuration metrics (effective batch size, AMP)
        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    config_metrics = {
                        'config/effective_batch_size': effective_batch_size,
                        'config/gradient_accumulation_steps': gradient_accumulation_steps,
                        'config/physical_batch_size': batch_size
                    }

                    # Add AMP metrics if enabled
                    if env['use_amp'] and env['scaler'] is not None:
                        config_metrics['amp/loss_scale'] = env['scaler'].get_scale()
                        config_metrics['amp/enabled'] = 1

                    wandb.log(config_metrics, step=epoch)
            except Exception as e:
                print(f"⚠️ Failed to log configuration metrics: {e}")

        all_loss_history.extend(train_results['loss_history'])
        all_grad_norm_history.extend(train_results['grad_norm_history'])

    training_time = time.time() - start_time

    print("-" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final loss: {all_loss_history[-1]:.4f}")
    print(f"Loss reduction: {((all_loss_history[0] - all_loss_history[-1]) / all_loss_history[0] * 100):.1f}%")
    print("=" * 60)

    # Create visualization
    metrics_summary = env['metrics_tracker'].get_summary()
    train_dataset_size = len(env['train_loader'].dataset)
    _create_training_visualization(
        all_loss_history, all_grad_norm_history, metrics_summary, n_epochs, batch_size, train_dataset_size
    )

    return {
        "loss_history": all_loss_history,
        "grad_norm_history": all_grad_norm_history,
        "final_loss": all_loss_history[-1],
        "initial_loss": all_loss_history[0],
        "training_time_seconds": training_time,
        "samples_per_second": train_dataset_size * n_epochs / training_time,
        "metrics_summary": metrics_summary,
        "best_epoch": env['metrics_tracker'].get_best_epoch('val/loss', 'min'),
        "amp_enabled": env['use_amp'],
        "final_loss_scale": env['scaler'].get_scale() if (env['use_amp'] and env['scaler'] is not None) else None
    }


def test_hyperparameter_search(
    model_factory: Any,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_trials: int = 10,
    search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization using Optuna.

    Searches over:
    - Learning rate
    - Batch size
    - Warmup steps
    - Weight decay

    Args:
        model_factory: Function that creates a fresh model instance
        config: Model configuration
        train_data: Training data (if None, generates synthetic)
        n_trials: Number of Optuna trials
        search_space: Custom search space (if None, uses defaults)

    Returns:
        Dictionary with best parameters and optimization history
    """
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. Install with: pip install optuna")
        return {"error": "optuna not installed"}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    # Instantiate a temporary model to detect vocab_size
    temp_model = model_factory()
    vocab_size = _detect_vocab_size(temp_model, config)
    del temp_model  # Free memory

    # Generate synthetic data if needed
    if train_data is None:
        train_data = [
            torch.randint(0, vocab_size, (32,))
            for _ in range(30)
        ]

    print("=" * 60)
    print("HYPERPARAMETER SEARCH (Optuna)")
    print("=" * 60)
    print(f"Trials: {n_trials}")
    print(f"Training samples: {len(train_data)}")
    print("-" * 60)

    def objective(trial):
        """Optuna objective function."""
        # Sample hyperparameters
        if search_space is None:
            lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
            warmup_steps = trial.suggest_int('warmup_steps', 0, 10)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        else:
            lr = trial.suggest_loguniform('learning_rate', *search_space.get('lr', (1e-5, 1e-3)))
            batch_size = trial.suggest_categorical('batch_size', search_space.get('batch_size', [2, 4, 8]))
            warmup_steps = trial.suggest_int('warmup_steps', *search_space.get('warmup', (0, 10)))
            weight_decay = trial.suggest_loguniform('weight_decay', *search_space.get('wd', (1e-6, 1e-2)))

        # Create fresh model
        model = model_factory()
        device = next(model.parameters()).device
        model.train()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Quick training (2 epochs)
        n_epochs = 2
        losses = []

        for epoch in range(n_epochs):
            for i in range(0, len(train_data), batch_size):
                batch = torch.stack(train_data[i:i+batch_size]).to(device)

                logits = _safe_get_model_output(model, batch)

                # Next-token prediction loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())

        # Return average loss
        return np.mean(losses)

    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("-" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params
            }
            for t in study.trials
        ]
    }

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Optimization history
        trial_numbers = [t.number for t in study.trials]
        trial_values = [t.value for t in study.trials]

        axes[0].plot(trial_numbers, trial_values, marker='o', linewidth=2, alpha=0.7)
        axes[0].axhline(y=study.best_value, color='r', linestyle='--',
                       linewidth=2, label=f'Best: {study.best_value:.4f}')
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())

            axes[1].barh(params, importances, edgecolor='black')
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, alpha=0.3, axis='x')
        except:
            axes[1].text(0.5, 0.5, 'Importance analysis\nnot available',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return results


def test_benchmark_comparison(
    model: nn.Module,
    config: Any,
    baseline_model_name: str = "distilgpt2",
    test_data: Optional[List[torch.Tensor]] = None,
    n_samples: int = 20
) -> Dict[str, Any]:
    """
    Compare model against a baseline transformer.

    Compares:
    - Inference speed
    - Parameter count
    - Memory footprint
    - Loss/perplexity on test data

    Args:
        model: Custom model to benchmark
        config: Model configuration
        baseline_model_name: HuggingFace model name to compare against
        test_data: Test samples (if None, generates synthetic)
        n_samples: Number of samples to test

    Returns:
        Dictionary with comparative metrics
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return {"error": "transformers not installed"}

    device = next(model.parameters()).device
    vocab_size = _detect_vocab_size(model, config)

    print("=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"Custom model vs. {baseline_model_name}")
    print(f"Test samples: {n_samples}")
    print("-" * 60)

    # Load baseline model
    print(f"Loading baseline model: {baseline_model_name}...")
    baseline = load_baseline_model(baseline_model_name, device)
    if baseline is None:
        return {"error": f"Failed to load baseline: {baseline_model_name}"}

    # Generate test data
    if test_data is None:
        test_data = [torch.randint(0, vocab_size, (32,)).to(device) for _ in range(n_samples)]
    else:
        test_data = [t.to(device) for t in test_data[:n_samples]]

    # Compare parameter counts
    custom_params = sum(p.numel() for p in model.parameters())
    baseline_params = sum(p.numel() for p in baseline.parameters())

    print(f"\nParameter Count:")
    print(f"  Custom model:   {custom_params:,}")
    print(f"  Baseline model: {baseline_params:,}")
    print(f"  Ratio: {custom_params / baseline_params:.2f}x")

    # Compare inference speed
    print(f"\nBenchmarking inference speed...")
    custom_times = benchmark_inference_speed(model, test_data, device)
    baseline_times = benchmark_inference_speed(baseline, test_data, device)

    custom_avg_ms = np.mean(custom_times) * 1000
    baseline_avg_ms = np.mean(baseline_times) * 1000

    print(f"\nInference Speed (avg):")
    print(f"  Custom model:   {custom_avg_ms:.2f} ms")
    print(f"  Baseline model: {baseline_avg_ms:.2f} ms")
    print(f"  Speedup: {baseline_avg_ms / custom_avg_ms:.2f}x")

    # Compare loss/perplexity
    print(f"\nComputing perplexity...")
    custom_ppl = compute_model_perplexity(model, test_data, vocab_size, is_baseline=False, safe_get_model_output=_safe_get_model_output)
    baseline_ppl = compute_model_perplexity(baseline, test_data, vocab_size, is_baseline=True)

    print(f"\nPerplexity:")
    print(f"  Custom model:   {custom_ppl:.2f}")
    print(f"  Baseline model: {baseline_ppl:.2f}")
    print(f"  Ratio: {custom_ppl / baseline_ppl:.2f}x")

    print("=" * 60)

    # Create visualization
    create_benchmark_visualization(
        custom_params, baseline_params, custom_avg_ms, baseline_avg_ms, custom_ppl, baseline_ppl
    )

    return {
        "parameter_count": {
            "custom": custom_params,
            "baseline": baseline_params,
            "ratio": custom_params / baseline_params
        },
        "inference_speed_ms": {
            "custom": custom_avg_ms,
            "baseline": baseline_avg_ms,
            "speedup": baseline_avg_ms / custom_avg_ms
        },
        "perplexity": {
            "custom": custom_ppl,
            "baseline": baseline_ppl,
            "ratio": custom_ppl / baseline_ppl
        },
        "baseline_model": baseline_model_name,
    }
