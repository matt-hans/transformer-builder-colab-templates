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
import math
from typing import Any, Dict, List, Optional
import time
import numpy as np

# Import AMP utilities for mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Import DataLoader utilities for efficient data loading
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

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
__all__ = ['test_fine_tuning', 'test_hyperparameter_search', 'test_benchmark_comparison', 'test_amp_speedup_benchmark', 'get_cosine_schedule_with_warmup']


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


def _detect_pad_token_id(config: Any) -> int:
    """
    Detect padding token ID from config or tokenizer.

    Priority:
    1. config.pad_token_id (explicit attribute)
    2. config.tokenizer.pad_token_id (tokenizer attribute)
    3. Default fallback (0)

    Args:
        config: Model configuration object

    Returns:
        Padding token ID (int)
    """
    if hasattr(config, 'pad_token_id') and config.pad_token_id is not None:
        return config.pad_token_id
    elif hasattr(config, 'tokenizer') and hasattr(config.tokenizer, 'pad_token_id'):
        return config.tokenizer.pad_token_id
    else:
        print("⚠️  No pad_token_id found in config/tokenizer, defaulting to 0")
        return 0


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


def _safe_get_model_output(
    model: nn.Module,
    input_ids: torch.Tensor,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
) -> torch.Tensor:
    """
    Safely extract logits tensor from model output.

    Wraps model() call and handles diverse output formats.
    """
    if adapter is not None and task_spec is not None:
        try:
            batch = {'input_ids': input_ids}
            prepared = adapter.prepare_inputs(batch, task_spec)
            _loss, outputs = adapter.forward_for_loss(model, prepared, task_spec)
            if isinstance(outputs, dict) and 'logits' in outputs:
                return outputs['logits']
            # Fallback extraction if adapter returns raw output
            output_tmp = outputs
            try:
                from utils.adapters.model_adapter import _extract_logits_generic as _extract
                return _extract(output_tmp)
            except Exception:
                pass
        except Exception:
            pass
    output = model(input_ids)
    return _extract_output_tensor(output)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup followed by cosine decay.

    LR schedule:
      - Steps [0, num_warmup_steps): Linear increase from 0 to initial LR
      - Steps [num_warmup_steps, num_training_steps]: Cosine decay to 0

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps (typically 10% of total)
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default 0.5 → decay to 0)
        last_epoch: Last epoch for resuming (default -1)

    Returns:
        LambdaLR scheduler to step after each optimizer step
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # progress ∈ [0, 1]
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Build optimizer parameter groups applying weight decay only to appropriate weights.

    Excludes biases and LayerNorm weights from weight decay, as standard in
    transformer training (BERT/GPT). Uses parameter names for classification.

    Args:
        model: Model with named parameters
        weight_decay: Weight decay value for decayed parameters

    Returns:
        Two parameter groups: [{'params': decay_params, 'weight_decay': wd}, {'params': no_decay_params, 'weight_decay': 0.0}]
    """
    no_decay_keys = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keys):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from an average cross-entropy loss value.

    Perplexity is defined as exp(loss). Returns inf for infinite loss and
    caps very large values to 1e6 to avoid overflow in downstream consumers.

    Args:
        loss: Average cross-entropy loss (natural log base)

    Returns:
        Perplexity as a float (>= 1.0), or inf if loss is inf.
    """
    if loss == float('inf'):
        return float('inf')
    try:
        return min(float(torch.exp(torch.tensor(loss)).item()), 1e6)
    except Exception:
        return float('inf')


def _compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute L2 norm of gradients across all trainable parameters.

    Calculates sqrt(sum(||grad_i||_2^2)) for all parameters that currently
    have gradients. Returns 0.0 when no gradients are present (e.g., before
    the first backward pass).

    Args:
        model: PyTorch model with gradients computed (after loss.backward())

    Returns:
        Float L2 norm of gradients (0.0 if no gradients exist).

    Example:
        >>> loss.backward()
        >>> gnorm = _compute_gradient_norm(model)
        >>> print(f"Gradient norm: {gnorm:.4f}")
    """
    total_sq = 0.0
    any_grad = False

    for p in model.parameters():
        if p.grad is None:
            continue
        any_grad = True
        g = p.grad.detach()
        # Convert to dense norm for sparse tensors
        if g.is_sparse:
            g = g.coalesce()
            param_norm = g.values().float().norm(2)
        else:
            param_norm = g.float().norm(2)
        total_sq += float(param_norm.item() ** 2)

    if not any_grad:
        return 0.0
    return float(total_sq ** 0.5)


def _log_gpu_metrics(tracker: Any, step: int) -> None:
    """
    Log GPU metrics (memory allocated/reserved, utilization, temperature) if available.

    Uses torch.cuda for memory and optionally pynvml or nvidia-smi for
    utilization/temperature. Swallows all exceptions to avoid interfering
    with training.
    """
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            return
        # Memory metrics (MB)
        mem_alloc_mb = float(_torch.cuda.memory_allocated() / (1024 ** 2))
        mem_res_mb = float(_torch.cuda.memory_reserved() / (1024 ** 2))
        try:
            tracker.log_scalar('gpu/memory_allocated_mb', mem_alloc_mb, step=step)
            tracker.log_scalar('gpu/memory_reserved_mb', mem_res_mb, step=step)
        except Exception:
            pass

        # Utilization/temperature via pynvml
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            try:
                tracker.log_scalar('gpu/utilization_percent', util, step=step)
                tracker.log_scalar('gpu/temperature_celsius', temp, step=step)
            except Exception:
                pass
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        except Exception:
            # Fallback to nvidia-smi
            try:
                import subprocess as _sp
                out = _sp.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=False
                )
                parts = out.stdout.strip().split(',')
                if parts and parts[0].strip():
                    try:
                        tracker.log_scalar('gpu/utilization_percent', float(parts[0].strip()), step=step)
                    except Exception:
                        pass
                if len(parts) > 1 and parts[1].strip():
                    try:
                        tracker.log_scalar('gpu/temperature_celsius', float(parts[1].strip()), step=step)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass


def _log_gradient_distribution(
    model: nn.Module,
    tracker: Any,
    step: int,
    *,
    log_histogram: bool = False,
    max_samples: int = 200000
) -> None:
    """
    Log per-parameter gradient norms and optional histogram to the tracker.

    Args:
        model: The model with gradients populated (after backward, before zero_grad)
        tracker: MetricsTracker instance (must support log_scalar)
        step: Training step/epoch for logging
        log_histogram: When True, also logs a global gradient value histogram
        max_samples: Max number of gradient values to sample for histogram to limit overhead
    """
    grads_sampled = []
    sample_every = 1
    collected = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        try:
            gnorm = float(param.grad.data.float().norm(2).item())
            tracker.log_scalar(f'gradients/{name}/norm', gnorm, step=step)
        except Exception:
            # Continue if any param causes an issue
            continue

        if log_histogram and max_samples > 0:
            try:
                g = param.grad.detach().flatten()
                if g.numel() == 0:
                    continue
                # Downsample if too large
                if collected < max_samples:
                    remaining = max_samples - collected
                    if g.numel() > remaining:
                        # Uniform sampling without replacement
                        idx = torch.randperm(g.numel())[:remaining]
                        grads_sampled.append(g[idx].cpu())
                        collected += int(remaining)
                    else:
                        grads_sampled.append(g.cpu())
                        collected += int(g.numel())
            except Exception:
                pass

    if log_histogram and grads_sampled and tracker.use_wandb:
        try:
            import wandb
            all_vals = torch.cat(grads_sampled).numpy().tolist()
            wandb.log({'gradients/histogram': wandb.Histogram(all_vals)}, step=step)
        except Exception:
            pass
def _run_training_epoch_simple(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_token_id: int = 0,
    max_grad_norm: float = 1.0,
    *,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
) -> float:
    """
    Run a single training epoch over a provided DataLoader.

    This simplified helper focuses on correctness and reuse for compact
    training routines (e.g., Optuna objectives). It computes next-token
    prediction loss with padding masked out, applies a standard backward
    pass and gradient clipping, and updates model parameters.

    Args:
        model: Model to train (set to train mode by caller or inside)
        dataloader: Iterable of input batches (TensorDataset-style)
        optimizer: Optimizer instance for parameter updates
        device: Target device for tensors and model execution
        pad_token_id: Token to ignore in loss calculation (default: 0)
        max_grad_norm: Gradient clipping norm (default: 1.0)

    Returns:
        Average loss (float) across all batches in the epoch. If the
        dataloader is empty, returns float('inf').

    Raises:
        RuntimeError: If model outputs cannot be coerced into logits tensor
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch_tuple in dataloader:
        batch = batch_tuple[0].to(device, non_blocking=True)

        logits = _safe_get_model_output(model, batch, adapter, task_spec)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    return total_loss / total_steps if total_steps > 0 else float('inf')


def _run_validation_epoch_simple(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_token_id: int = 0,
    *,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Run a single validation epoch over a provided DataLoader.

    Computes average masked loss and corresponding perplexity. Designed for
    quick reuse in light-weight validation flows.

    Args:
        model: Model to evaluate (set to eval mode inside)
        dataloader: Iterable of validation input batches
        device: Target device
        pad_token_id: Token to ignore in loss calculation

    Returns:
        Dict with keys:
            - 'loss': float
            - 'perplexity': float
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch_tuple in dataloader:
            batch = batch_tuple[0].to(device, non_blocking=True)
            logits = _safe_get_model_output(model, batch, adapter, task_spec)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
            )

            total_loss += float(loss.item())
            total_steps += 1

    avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
    return {"loss": avg_loss, "perplexity": _calculate_perplexity(avg_loss)}


def _setup_training(
    model: nn.Module,
    config: Any,
    n_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    use_amp: bool,
    use_wandb: bool,
    random_seed: int,
    use_lr_schedule: bool,
    train_data: Optional[List[torch.Tensor]] = None,
    val_data: Optional[List[torch.Tensor]] = None,
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    """
    Setup optimizer, scheduler, dataloaders, scaler, and metrics tracker.

    Thin wrapper around _setup_training_environment to provide a stable
    orchestration surface for test_fine_tuning().
    """
    env = _setup_training_environment(
        model, config, train_data, val_data, n_epochs,
        learning_rate, weight_decay, batch_size,
        use_amp, use_wandb,
        random_seed=random_seed,
        use_lr_schedule=use_lr_schedule,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    # Attach model for downstream helpers that expect it
    env['model'] = model
    return env


def _train_model(
    model: nn.Module,
    env: Dict[str, Any],
    n_epochs: int,
    pad_token_id: int,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    batch_size: int,
    effective_batch_size: int,
    use_wandb: bool,
    *,
    log_grad_dist_every: int = 5,
    log_grad_histogram: bool = False,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the training/validation loop and return histories, metrics, and timing.
    """
    all_loss_history: List[float] = []
    all_grad_norm_history: List[float] = []
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        # Log GPU metrics once per epoch (before training step)
        try:
            _log_gpu_metrics(env['metrics_tracker'], step=epoch)
        except Exception:
            pass

        # Determine whether to log gradient distribution this epoch
        log_this_epoch = (log_grad_dist_every > 0 and (epoch % log_grad_dist_every == 0))

        train_results = _run_training_epoch(
            model,
            env['train_loader'], env['optimizer'], env['scheduler'],
            env['scaler'], env['use_amp'], env['vocab_size'], env['metrics_tracker'], env['device'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_norm=gradient_clip_norm,
            pad_token_id=pad_token_id,
            log_grad_dist=log_this_epoch,
            grad_log_step=epoch,
            log_grad_histogram=log_grad_histogram,
            adapter=adapter,
            task_spec=task_spec,
        )

        val_results = _run_validation_epoch(
            model,
            env['val_loader'], env['vocab_size'], env['metrics_tracker'], env['device'],
            pad_token_id=pad_token_id,
            adapter=adapter,
            task_spec=task_spec,
        )

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

        _log_training_config_to_wandb(
            use_wandb=use_wandb,
            effective_batch_size=effective_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_size=batch_size,
            use_amp=env['use_amp'],
            scaler=env['scaler'],
            epoch=epoch
        )

        try:
            current_lr = env['optimizer'].param_groups[0]['lr']
            env['metrics_tracker'].log_scalar('train/learning_rate', current_lr, step=epoch)
        except Exception:
            pass

        all_loss_history.extend(train_results['loss_history'])
        all_grad_norm_history.extend(train_results['grad_norm_history'])

    training_time = time.time() - start_time

    metrics_summary = env['metrics_tracker'].get_summary()
    return {
        'training_time': training_time,
        'loss_history': all_loss_history,
        'grad_norm_history': all_grad_norm_history,
        'metrics_summary': metrics_summary
    }


def _format_results(
    loss_history: List[float],
    training_time: float,
    metrics_summary: Any,
    n_epochs: int,
    batch_size: int,
    train_dataset_size: int,
) -> Dict[str, Any]:
    """
    Format final results dictionary from histories and summary metrics.
    """
    # Best epoch from metrics_summary if available
    best_epoch = None
    try:
        if hasattr(metrics_summary, 'columns') and 'val/loss' in metrics_summary.columns:
            best_epoch = metrics_summary['val/loss'].idxmin()
    except Exception:
        best_epoch = None

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else float('inf'),
        "initial_loss": loss_history[0] if loss_history else float('inf'),
        "training_time_seconds": training_time,
        "samples_per_second": (train_dataset_size * n_epochs / training_time) if training_time > 0 else 0.0,
        "metrics_summary": metrics_summary,
        "best_epoch": best_epoch
    }


def _compute_loss_and_backward(
    model: nn.Module,
    batch: torch.Tensor,
    scaler: Optional[Any],
    use_amp: bool,
    vocab_size: int,
    metrics_tracker: Any,
    gradient_accumulation_steps: int,
    pad_token_id: int = 0,
    *,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
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
        pad_token_id: Token ID to exclude from loss calculation (default: 0)

    Returns:
        Tuple of (loss_value, accuracy) where loss_value is the unscaled loss
    """
    # Forward pass with optional autocast
    if use_amp:
        with autocast():
            logits = _safe_get_model_output(model, batch, adapter, task_spec)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id  # CRITICAL FIX: Exclude padding from loss
            )

            # Scale loss by accumulation steps to get correct gradient magnitude
            scaled_loss = loss / gradient_accumulation_steps

        # Compute accuracy outside autocast (FP32)
        # CRITICAL FIX: Exclude padding tokens from accuracy to match loss calculation
        with torch.no_grad():
            accuracy = metrics_tracker.compute_accuracy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id
            )
    else:
        # Standard FP32 forward pass
        logits = _safe_get_model_output(model, batch, adapter, task_spec)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id  # CRITICAL FIX: Exclude padding from loss
        )

        # Scale loss by accumulation steps
        scaled_loss = loss / gradient_accumulation_steps

        # CRITICAL FIX: Exclude padding tokens from accuracy to match loss calculation
        accuracy = metrics_tracker.compute_accuracy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=pad_token_id
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
    weight_decay: float,
    batch_size: int,
    use_amp: bool,
    use_wandb: bool,
    random_seed: int = 42,
    use_lr_schedule: bool = True,
    gradient_accumulation_steps: int = 1
) -> Dict[str, Any]:
    """
    Setup training environment: data, optimizer, scheduler, scaler, metrics tracker.

    Args:
        random_seed: Random seed for DataLoader generator (ensures reproducible shuffling)
        gradient_accumulation_steps: Number of gradient accumulation steps for effective step tracking

    Returns:
        Dictionary with all training components
    """
    from utils.training.metrics_tracker import MetricsTracker
    from utils.training.seed_manager import seed_worker, create_seeded_generator

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

    # CRITICAL: Create seeded generator for reproducible DataLoader shuffling
    # Without this, batch order will be non-deterministic even with set_random_seed()
    generator = create_seeded_generator(random_seed)

    train_dataset = TensorDataset(torch.stack(train_data))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster CPU->GPU transfer
        prefetch_factor=2 if use_workers else None,  # Pre-load 2 batches
        persistent_workers=use_workers,
        worker_init_fn=seed_worker,  # CRITICAL: Seed each worker process
        generator=generator  # CRITICAL: Reproducible shuffling
    )

    val_dataset = TensorDataset(torch.stack(val_data))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if use_workers else None,
        persistent_workers=use_workers,
        worker_init_fn=seed_worker  # Also seed validation workers for consistency
    )

    # Setup optimizer (with weight decay exclusion) and scheduler
    param_groups = _get_optimizer_grouped_parameters(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    total_steps = max(1, n_epochs * len(train_loader))
    if use_lr_schedule:
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        # Constant LR scheduler (no change) for backward compatibility
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # Initialize metrics tracker with gradient accumulation awareness
    metrics_tracker = MetricsTracker(
        use_wandb=use_wandb,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

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
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float = 1.0,
    pad_token_id: int = 0,
    *,
    log_grad_dist: bool = False,
    grad_log_step: Optional[int] = None,
    log_grad_histogram: bool = False,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
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
        gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
        pad_token_id: Token ID to exclude from loss calculation (default: 0)

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
    logged_clip_metrics = False
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
            gradient_accumulation_steps=gradient_accumulation_steps,
            pad_token_id=pad_token_id,
            adapter=adapter,
            task_spec=task_spec,
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
            # Pre/post-clip gradient norms and clipping
            if use_amp:
                scaler.unscale_(optimizer)

            # Compute pre-clip norm once per optimizer update
            pre_clip_norm = _compute_gradient_norm(model)

            if gradient_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip_norm
                )
                post_clip_value = grad_norm.item()
            else:
                # Clipping disabled; use current norm as post value
                grad_norm = torch.tensor(pre_clip_norm)
                post_clip_value = pre_clip_norm

            # Log pre/post once per epoch (first update)
            if not logged_clip_metrics:
                try:
                    metrics_tracker.log_scalar('gradients/pre_clip_norm', float(pre_clip_norm))
                    metrics_tracker.log_scalar('gradients/post_clip_norm', float(post_clip_value))
                except Exception:
                    pass
                logged_clip_metrics = True

            # Optimizer step with overflow check for AMP
            if use_amp:
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    metrics_tracker.log_scalar('train/gradient_overflow', 1.0)
                    scaler.update()
            else:
                optimizer.step()

            # Optionally log per-layer gradient distribution before zeroing grads
            if log_grad_dist and grad_log_step is not None:
                try:
                    _log_gradient_distribution(model, metrics_tracker, grad_log_step, log_histogram=log_grad_histogram)
                except Exception:
                    pass

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
    device: torch.device,
    pad_token_id: int = 0,
    *,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Execute validation epoch using DataLoader for efficient async data loading.

    Args:
        model: Model to validate
        val_loader: DataLoader with validation batches
        vocab_size: Vocabulary size
        metrics_tracker: Metrics tracking instance
        device: Device to run on
        pad_token_id: Token ID to exclude from loss calculation (default: 0)

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

            logits = _safe_get_model_output(model, val_batch, adapter, task_spec)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = val_batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id  # CRITICAL FIX: Exclude padding from loss
            )

            # CRITICAL FIX: Exclude padding tokens from accuracy to match loss calculation
            accuracy = metrics_tracker.compute_accuracy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id
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


def _capture_and_save_environment_snapshot() -> None:
    """
    Capture environment snapshot for reproducibility.

    Captures system information, package versions, and generates reproduction script.
    Prints status messages about success/failure.
    """
    print("📸 Capturing environment snapshot...")
    try:
        env_info = capture_environment()
        req_path, env_path, repro_path = save_environment_snapshot(env_info, "./environment")
        return {'env_info': env_info, 'req_path': req_path, 'env_path': env_path, 'repro_path': repro_path}
    except Exception as e:
        print(f"⚠️ Failed to capture environment snapshot: {e}")
        print("   Training will continue without environment snapshot")
        return None


def _log_training_config_to_wandb(
    use_wandb: bool,
    effective_batch_size: int,
    gradient_accumulation_steps: int,
    batch_size: int,
    use_amp: bool,
    scaler: Optional[Any],
    epoch: int
) -> None:
    """
    Log training configuration metrics to W&B.

    Args:
        use_wandb: Whether W&B logging is enabled
        effective_batch_size: Physical batch size * accumulation steps
        gradient_accumulation_steps: Number of accumulation steps
        batch_size: Physical batch size
        use_amp: Whether AMP is enabled
        scaler: GradScaler instance (if AMP enabled)
        epoch: Current epoch number
    """
    if not use_wandb:
        return

    try:
        import wandb
        if wandb.run is None:
            return

        config_metrics = {
            'config/effective_batch_size': effective_batch_size,
            'config/gradient_accumulation_steps': gradient_accumulation_steps,
            'config/physical_batch_size': batch_size
        }

        # Add AMP metrics if enabled
        if use_amp and scaler is not None:
            config_metrics['amp/loss_scale'] = scaler.get_scale()
            config_metrics['amp/enabled'] = 1

        wandb.log(config_metrics, step=epoch)
    except Exception as e:
        print(f"⚠️ Failed to log configuration metrics: {e}")


def test_fine_tuning(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    val_data: Optional[List[torch.Tensor]] = None,
    n_epochs: int = 3,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    batch_size: int = 4,
    use_wandb: bool = False,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_clip_norm: float = 1.0,
    random_seed: int = 42,
    deterministic: bool = False,
    use_lr_schedule: bool = True,
    log_grad_dist_every: int = 5,
    log_grad_histogram: bool = False,
    adapter: Optional[Any] = None,
    task_spec: Optional[Any] = None,
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
    - Padding token exclusion from loss calculation (ignore_index)
    - Reproducible training with DataLoader worker seeding

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
        gradient_clip_norm: Maximum gradient norm for clipping (default: 1.0)
        random_seed: Random seed for reproducibility (default: 42)
        deterministic: If True, enables fully deterministic mode (slower, ~5-10% performance impact).
            If False, uses fast mode with cuDNN optimizations (default: False)

    Returns:
        Dictionary with training metrics, loss curves, and MetricsTracker summary
    """
    from utils.training.seed_manager import set_random_seed

    # Set random seed with determinism option
    set_random_seed(random_seed, deterministic=deterministic)

    # Detect pad_token_id from config or tokenizer using helper function
    pad_token_id = _detect_pad_token_id(config)

    # Setup training environment with seeded DataLoaders
    env = _setup_training(
        model=model,
        config=config,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        use_amp=use_amp,
        use_wandb=use_wandb,
        random_seed=random_seed,
        use_lr_schedule=use_lr_schedule,
        train_data=train_data,
        val_data=val_data,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # Compute effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Log training configuration
    try:
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _logger.info("FINE-TUNING TEST")
        _logger.info(
            f"Train samples: {len(env['train_loader'].dataset)} | Val samples: {len(env['val_loader'].dataset)}"
        )
        _logger.info(
            f"Epochs: {n_epochs} | LR: {learning_rate} | Batch size: {batch_size} | Weight decay: {weight_decay}"
        )
        _logger.info(
            f"Grad accum: {gradient_accumulation_steps} | Effective batch: {effective_batch_size} | AMP: {env['use_amp']}"
        )
        _logger.info(f"W&B logging: {use_wandb} | Device: {env['device']}")
    except Exception:
        pass
    if use_lr_schedule:
        total_steps = n_epochs * len(env['train_loader'])
        warmup_steps = int(0.1 * total_steps)
        try:
            _logger.info(f"LR schedule: warmup_steps={warmup_steps}, total_steps={total_steps}")
        except Exception:
            pass
    

    # Capture environment snapshot for reproducibility
    env_snapshot = _capture_and_save_environment_snapshot()
    if env_snapshot and use_wandb:
        try:
            log_environment_to_wandb(
                env_snapshot['req_path'],
                env_snapshot['env_path'],
                env_snapshot['repro_path'],
                env_snapshot['env_info']
            )
        except Exception as e:
            try:
                _logger.warning(f"Failed to log environment to W&B: {e}")
            except Exception:
                pass

    

    # Delegate training to orchestrator
    train_out = _train_model(
        model=model,
        env=env,
        n_epochs=n_epochs,
        pad_token_id=pad_token_id,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        use_wandb=use_wandb,
        log_grad_dist_every=log_grad_dist_every,
        log_grad_histogram=log_grad_histogram,
        adapter=adapter,
        task_spec=task_spec,
    )

    # Visualization
    metrics_summary = train_out['metrics_summary']
    train_dataset_size = len(env['train_loader'].dataset)
    _create_training_visualization(
        train_out['loss_history'], train_out['grad_norm_history'], metrics_summary, n_epochs, batch_size, train_dataset_size
    )

    # Results
    results = _format_results(
        loss_history=train_out['loss_history'],
        training_time=train_out['training_time'],
        metrics_summary=metrics_summary,
        n_epochs=n_epochs,
        batch_size=batch_size,
        train_dataset_size=train_dataset_size,
    )
    results.update({
        "grad_norm_history": train_out['grad_norm_history'],
        "amp_enabled": env['use_amp'],
        "final_loss_scale": env['scaler'].get_scale() if (env['use_amp'] and env['scaler'] is not None) else None
    })
    return results


def test_hyperparameter_search(
    model_factory: Any,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_trials: int = 10,
    search_space: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    deterministic: bool = False
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization using Optuna.

    Searches over:
    - Learning rate
    - Batch size
    - Warmup steps
    - Weight decay

    Loss calculation excludes padding tokens using ignore_index parameter.

    Args:
        model_factory: Function that creates a fresh model instance
        config: Model configuration
        train_data: Training data (if None, generates synthetic)
        n_trials: Number of Optuna trials
        search_space: Custom search space (if None, uses defaults)
        random_seed: Random seed for reproducibility (default: 42)
        deterministic: If True, enables fully deterministic mode (default: False)

    Returns:
        Dictionary with best parameters and optimization history
    """
    from utils.training.seed_manager import set_random_seed, seed_worker, create_seeded_generator

    # Set random seed with determinism option
    set_random_seed(random_seed, deterministic=deterministic)

    # Detect pad_token_id from config or tokenizer using helper function
    pad_token_id = _detect_pad_token_id(config)

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

        # Setup optimizer with weight decay exclusion
        param_groups = _get_optimizer_grouped_parameters(model, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

        # Quick training (2 epochs) using shared helpers
        n_epochs = 2
        epoch_losses: List[float] = []

        # Build DataLoader for shared epoch helper
        from utils.training.seed_manager import create_seeded_generator, seed_worker
        dl = DataLoader(
            TensorDataset(torch.stack(train_data)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=create_seeded_generator(random_seed),
        )

        for _ in range(n_epochs):
            avg_loss = _run_training_epoch_simple(
                model=model,
                dataloader=dl,
                optimizer=optimizer,
                device=device,
                pad_token_id=pad_token_id,
                max_grad_norm=1.0,
            )
            epoch_losses.append(avg_loss)

        # Return mean of epoch averages (equivalent when epoch lengths equal)
        return float(np.mean(epoch_losses))

    # Create study and optimize with reproducible sampler
    # Use TPESampler with fixed seed for reproducible hyperparameter selection
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
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
        except Exception:  # CRITICAL FIX: Don't catch KeyboardInterrupt/SystemExit
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

    try:
        _logger.info("BENCHMARK COMPARISON")
        _logger.info(f"Custom model vs. {baseline_model_name} | Test samples: {n_samples}")
    except Exception:
        pass

    # Load baseline model
    try:
        _logger.info(f"Loading baseline model: {baseline_model_name}...")
    except Exception:
        pass
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

    try:
        _logger.info("Parameter Count:")
        _logger.info(f"  Custom model:   {custom_params:,}")
        _logger.info(f"  Baseline model: {baseline_params:,}")
        _logger.info(f"  Ratio: {custom_params / baseline_params:.2f}x")
    except Exception:
        pass

    # Compare inference speed
    try:
        _logger.info("Benchmarking inference speed...")
    except Exception:
        pass
    custom_times = benchmark_inference_speed(model, test_data, device)
    baseline_times = benchmark_inference_speed(baseline, test_data, device)

    custom_avg_ms = np.mean(custom_times) * 1000
    baseline_avg_ms = np.mean(baseline_times) * 1000

    try:
        _logger.info("Inference Speed (avg):")
        _logger.info(f"  Custom model:   {custom_avg_ms:.2f} ms")
        _logger.info(f"  Baseline model: {baseline_avg_ms:.2f} ms")
        _logger.info(f"  Speedup: {baseline_avg_ms / custom_avg_ms:.2f}x")
    except Exception:
        pass

    # Compare loss/perplexity
    try:
        _logger.info("Computing perplexity...")
    except Exception:
        pass
    custom_ppl = compute_model_perplexity(model, test_data, vocab_size, is_baseline=False, safe_get_model_output=_safe_get_model_output)
    baseline_ppl = compute_model_perplexity(baseline, test_data, vocab_size, is_baseline=True)

    try:
        _logger.info("Perplexity:")
        _logger.info(f"  Custom model:   {custom_ppl:.2f}")
        _logger.info(f"  Baseline model: {baseline_ppl:.2f}")
        _logger.info(f"  Ratio: {custom_ppl / baseline_ppl:.2f}x")
    except Exception:
        pass

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

# Prevent pytest from collecting these API-style functions as tests when imported
for _name in [
    'test_fine_tuning',
    'test_hyperparameter_search',
    'test_benchmark_comparison',
]:
    try:
        globals()[_name].__test__ = False  # type: ignore[attr-defined]
    except Exception:
        pass
