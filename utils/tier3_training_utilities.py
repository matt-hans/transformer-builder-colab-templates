"""
Tier 3: Training Utilities

This module contains training-focused utilities for transformer models:
- Fine-tuning loop with loss tracking and gradient monitoring
- Hyperparameter optimization using Optuna
- Benchmark comparison against baseline models

These utilities are useful for training workflows and model optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import time
import numpy as np


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


def test_fine_tuning(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    val_data: Optional[List[torch.Tensor]] = None,
    n_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    use_wandb: bool = False
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

    Args:
        model: The transformer model to fine-tune
        config: Model configuration
        train_data: List of input_ids tensors (if None, generates synthetic data)
        val_data: List of validation input_ids tensors (if None, uses 20% of train)
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size for training
        use_wandb: Whether to log metrics to W&B (default: False)

    Returns:
        Dictionary with training metrics, loss curves, and MetricsTracker summary
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    # Import MetricsTracker
    from utils.training.metrics_tracker import MetricsTracker

    device = next(model.parameters()).device
    vocab_size = _detect_vocab_size(model, config)

    # Generate synthetic training data if not provided
    if train_data is None:
        print("Generating synthetic training data...")
        train_data = [
            torch.randint(0, vocab_size, (32,))
            for _ in range(50)  # 50 samples
        ]

    # Create validation split if not provided
    if val_data is None:
        split_idx = int(0.8 * len(train_data))
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]

    print("=" * 60)
    print("FINE-TUNING TEST")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"W&B logging: {use_wandb}")
    print("-" * 60)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(use_wandb=use_wandb)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = n_epochs * (len(train_data) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps
    )

    model.train()

    loss_history = []
    grad_norm_history = []

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_steps = 0
        max_grad_norm = 0.0

        # Shuffle data
        indices = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = torch.stack([train_data[idx] for idx in batch_indices]).to(device)

            # Forward pass
            logits = _safe_get_model_output(model, batch)

            # Compute loss (language modeling: predict next token)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Compute accuracy
            accuracy = metrics_tracker.compute_accuracy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            max_grad_norm = max(max_grad_norm, grad_norm.item())

            optimizer.step()
            scheduler.step()

            # Track metrics
            train_loss_sum += loss.item()
            train_acc_sum += accuracy
            train_steps += 1
            grad_norm_history.append(grad_norm.item())
            loss_history.append(loss.item())

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_sample in val_data:
                val_sample = val_sample.to(device).unsqueeze(0)  # Add batch dim

                logits = _safe_get_model_output(model, val_sample)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = val_sample[:, 1:].contiguous()

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

        # Log epoch metrics
        epoch_duration = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        metrics_tracker.log_epoch(
            epoch=epoch,
            train_metrics={
                'loss': train_loss_sum / train_steps,
                'accuracy': train_acc_sum / train_steps
            },
            val_metrics={
                'loss': val_loss_sum / val_steps,
                'accuracy': val_acc_sum / val_steps
            },
            learning_rate=current_lr,
            gradient_norm=max_grad_norm,
            epoch_duration=epoch_duration
        )

    training_time = time.time() - start_time

    print("-" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Loss reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%")
    print("=" * 60)

    results = {
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "initial_loss": loss_history[0],
        "training_time_seconds": training_time,
        "samples_per_second": len(train_data) * n_epochs / training_time,
        "metrics_summary": metrics_tracker.get_summary(),
        "best_epoch": metrics_tracker.get_best_epoch('val/loss', 'min')
    }

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curve (step-level)
        axes[0, 0].plot(loss_history, linewidth=2, alpha=0.7)
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Curve (Step-Level)')
        axes[0, 0].grid(True, alpha=0.3)

        # Add epoch markers
        steps_per_epoch = len(train_data) // batch_size
        for e in range(1, n_epochs):
            axes[0, 0].axvline(
                x=e * steps_per_epoch, color='r',
                linestyle='--', alpha=0.5, linewidth=1
            )

        # Epoch-level metrics (train vs val loss)
        df = metrics_tracker.get_summary()
        axes[0, 1].plot(
            df['epoch'], df['train/loss'], marker='o',
            label='Train Loss', linewidth=2
        )
        axes[0, 1].plot(
            df['epoch'], df['val/loss'], marker='s',
            label='Val Loss', linewidth=2
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
            df['epoch'], df['train/perplexity'], marker='o',
            label='Train PPL', linewidth=2
        )
        axes[1, 1].plot(
            df['epoch'], df['val/perplexity'], marker='s',
            label='Val PPL', linewidth=2
        )
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].set_title('Train vs Validation Perplexity')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return results


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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return {"error": "transformers not installed"}

    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

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
    try:
        baseline = AutoModelForCausalLM.from_pretrained(baseline_model_name).to(device)
        baseline.eval()
    except Exception as e:
        print(f"❌ Failed to load baseline: {str(e)}")
        return {"error": f"Failed to load baseline: {str(e)}"}

    # Generate test data
    if test_data is None:
        test_data = [
            torch.randint(0, vocab_size, (32,)).to(device)
            for _ in range(n_samples)
        ]
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
    model.eval()

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(test_data[0].unsqueeze(0))
            _ = baseline(test_data[0].unsqueeze(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark custom model
    custom_times = []
    for sample in test_data:
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample.unsqueeze(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        custom_times.append(time.perf_counter() - start)

    # Benchmark baseline
    baseline_times = []
    for sample in test_data:
        start = time.perf_counter()
        with torch.no_grad():
            _ = baseline(sample.unsqueeze(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - start)

    custom_avg_ms = np.mean(custom_times) * 1000
    baseline_avg_ms = np.mean(baseline_times) * 1000

    print(f"\nInference Speed (avg):")
    print(f"  Custom model:   {custom_avg_ms:.2f} ms")
    print(f"  Baseline model: {baseline_avg_ms:.2f} ms")
    print(f"  Speedup: {baseline_avg_ms / custom_avg_ms:.2f}x")

    # Compare loss/perplexity
    print(f"\nComputing perplexity...")
    custom_losses = []
    baseline_losses = []

    for sample in test_data:
        input_ids = sample.unsqueeze(0)

        # Custom model
        with torch.no_grad():
            custom_logits = _safe_get_model_output(model, input_ids)
            custom_loss = F.cross_entropy(
                custom_logits[:, :-1, :].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            custom_losses.append(custom_loss.item())

        # Baseline model
        with torch.no_grad():
            baseline_logits = baseline(input_ids).logits
            baseline_loss = F.cross_entropy(
                baseline_logits[:, :-1, :].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            baseline_losses.append(baseline_loss.item())

    custom_ppl = np.exp(np.mean(custom_losses))
    baseline_ppl = np.exp(np.mean(baseline_losses))

    print(f"\nPerplexity:")
    print(f"  Custom model:   {custom_ppl:.2f}")
    print(f"  Baseline model: {baseline_ppl:.2f}")
    print(f"  Ratio: {custom_ppl / baseline_ppl:.2f}x")

    print("=" * 60)

    results = {
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

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Parameter comparison
        axes[0].bar(['Custom', 'Baseline'],
                   [custom_params, baseline_params],
                   edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Parameter Count')
        axes[0].set_title('Model Size')
        axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Speed comparison
        axes[1].bar(['Custom', 'Baseline'],
                   [custom_avg_ms, baseline_avg_ms],
                   edgecolor='black', linewidth=2, color=['green', 'blue'])
        axes[1].set_ylabel('Latency (ms)')
        axes[1].set_title('Inference Speed')

        # Perplexity comparison
        axes[2].bar(['Custom', 'Baseline'],
                   [custom_ppl, baseline_ppl],
                   edgecolor='black', linewidth=2, color=['orange', 'red'])
        axes[2].set_ylabel('Perplexity (lower is better)')
        axes[2].set_title('Language Modeling Quality')

        plt.tight_layout()
        plt.show()

    return results
