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
import numpy as np
from typing import Any, Dict, List, Optional
import time


def test_fine_tuning(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Run a basic fine-tuning loop with loss tracking.

    Demonstrates:
    - Training loop setup
    - Gradient clipping
    - Learning rate scheduling
    - Loss convergence tracking

    Args:
        model: The transformer model to fine-tune
        config: Model configuration
        train_data: List of input_ids tensors (if None, generates synthetic data)
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size for training

    Returns:
        Dictionary with training metrics and loss curves
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    device = next(model.parameters()).device
    vocab_size = getattr(config, 'vocab_size', 50257)

    # Generate synthetic training data if not provided
    if train_data is None:
        print("Generating synthetic training data...")
        train_data = [
            torch.randint(0, vocab_size, (32,))
            for _ in range(50)  # 50 samples
        ]

    print("=" * 60)
    print("FINE-TUNING TEST")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs * (len(train_data) // batch_size)
    )

    model.train()

    loss_history = []
    grad_norm_history = []

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_losses = []

        # Shuffle data
        indices = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = torch.stack([train_data[idx] for idx in batch_indices]).to(device)

            # Forward pass
            logits = model(batch)

            # Compute loss (language modeling: predict next token)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_losses.append(loss.item())
            grad_norm_history.append(grad_norm.item())
            loss_history.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")

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
    }

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Loss curve
        axes[0].plot(loss_history, linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].grid(True, alpha=0.3)

        # Add epoch markers
        steps_per_epoch = len(train_data) // batch_size
        for e in range(1, n_epochs):
            axes[0].axvline(x=e * steps_per_epoch, color='r',
                           linestyle='--', alpha=0.5, linewidth=1)

        # Gradient norm
        axes[1].plot(grad_norm_history, linewidth=2, alpha=0.7, color='orange')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Gradient Norm')
        axes[1].set_title('Gradient Norm (after clipping)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1.0, color='r', linestyle='--',
                       linewidth=1, label='Clip threshold')
        axes[1].legend()

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

    vocab_size = getattr(config, 'vocab_size', 50257)

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

                logits = model(batch)

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
    vocab_size = getattr(config, 'vocab_size', 50257)

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
            custom_logits = model(input_ids)
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
