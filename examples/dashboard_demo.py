"""
Dashboard visualization demo with simulated training metrics.

This example demonstrates TrainingDashboard usage with different metric scenarios:
1. Full metrics (all 6 panels)
2. Minimal metrics (loss only)
3. Integration with TrainingConfig

Run this script to generate sample dashboards:
    python examples/dashboard_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from types import SimpleNamespace
from utils.training.dashboard import TrainingDashboard


def generate_full_metrics():
    """Generate realistic training metrics for 20 epochs."""
    np.random.seed(42)
    epochs = 20

    # Simulate realistic training curves
    train_loss = 2.5 * np.exp(-0.15 * np.arange(epochs)) + np.random.normal(0, 0.05, epochs)
    val_loss = 2.6 * np.exp(-0.14 * np.arange(epochs)) + np.random.normal(0, 0.08, epochs)

    # Accuracy improves with training
    train_acc = 0.3 + 0.5 * (1 - np.exp(-0.2 * np.arange(epochs))) + np.random.normal(0, 0.02, epochs)
    val_acc = 0.28 + 0.48 * (1 - np.exp(-0.19 * np.arange(epochs))) + np.random.normal(0, 0.03, epochs)

    # Learning rate: 10% warmup + cosine decay
    warmup_steps = int(0.1 * epochs)
    lr = np.concatenate([
        np.linspace(1e-6, 5e-5, warmup_steps),
        5e-5 * (1 + np.cos(np.pi * np.arange(epochs - warmup_steps) / (epochs - warmup_steps))) / 2
    ])

    # Gradient norms with occasional spikes
    grad_norms = np.random.lognormal(0.5, 0.3, epochs)
    post_clip = np.minimum(grad_norms, 1.0)  # Clipped at 1.0

    # Epoch duration with slight variance
    durations = 45 + np.random.normal(0, 2, epochs)

    return pd.DataFrame({
        'epoch': np.arange(1, epochs + 1),
        'train/loss': train_loss,
        'val/loss': val_loss,
        'val/perplexity': np.exp(val_loss),
        'train/accuracy': np.clip(train_acc, 0, 1),
        'val/accuracy': np.clip(val_acc, 0, 1),
        'learning_rate': lr,
        'gradients/pre_clip_norm': grad_norms,
        'gradients/post_clip_norm': post_clip,
        'epoch_duration': durations
    })


def main():
    """Generate sample dashboards."""
    print("=" * 80)
    print("TrainingDashboard Demo")
    print("=" * 80)

    # Example 1: Full metrics dashboard
    print("\n1. Creating dashboard with full metrics (20 epochs)...")
    full_metrics = generate_full_metrics()

    config = SimpleNamespace(
        learning_rate=5e-5,
        batch_size=4,
        epochs=20,
        gradient_clip_norm=1.0
    )

    dashboard = TrainingDashboard(figsize=(18, 12))
    fig = dashboard.plot(full_metrics, config=config, title='GPT-2 Fine-Tuning Dashboard')

    output_dir = 'examples/outputs'
    os.makedirs(output_dir, exist_ok=True)

    dashboard.save(f'{output_dir}/full_dashboard.png', dpi=150)
    print(f"   ✅ Saved to {output_dir}/full_dashboard.png")

    # Example 2: Minimal metrics (loss only)
    print("\n2. Creating dashboard with minimal metrics (5 epochs)...")
    minimal_metrics = pd.DataFrame({
        'epoch': [1, 2, 3, 4, 5],
        'train/loss': [2.5, 2.0, 1.8, 1.6, 1.5],
        'val/loss': [2.6, 2.1, 1.9, 1.7, 1.6]
    })

    dashboard_min = TrainingDashboard(figsize=(18, 12))
    fig_min = dashboard_min.plot(minimal_metrics, title='Minimal Metrics Dashboard')
    dashboard_min.save(f'{output_dir}/minimal_dashboard.png', dpi=150)
    print(f"   ✅ Saved to {output_dir}/minimal_dashboard.png")

    # Example 3: Export formats
    print("\n3. Exporting dashboard in multiple formats...")
    dashboard.save(f'{output_dir}/full_dashboard.pdf', dpi=150)
    print(f"   ✅ Saved PDF to {output_dir}/full_dashboard.pdf")

    dashboard.save(f'{output_dir}/full_dashboard.svg')
    print(f"   ✅ Saved SVG to {output_dir}/full_dashboard.svg")

    # Print metrics summary
    print("\n" + "=" * 80)
    print("Training Summary (Full Metrics)")
    print("=" * 80)
    best_idx = full_metrics['val/loss'].idxmin()
    print(f"Best Epoch: {int(full_metrics.loc[best_idx, 'epoch'])}")
    print(f"Best Val Loss: {full_metrics.loc[best_idx, 'val/loss']:.4f}")
    print(f"Best Perplexity: {full_metrics.loc[best_idx, 'val/perplexity']:.2f}")
    print(f"Best Val Accuracy: {full_metrics.loc[best_idx, 'val/accuracy']:.2%}")
    print(f"Total Training Time: {full_metrics['epoch_duration'].sum():.1f}s")
    print(f"Avg Epoch Duration: {full_metrics['epoch_duration'].mean():.1f}s")

    print("\n✅ All demos completed successfully!")
    print(f"   Check {output_dir}/ for generated dashboards")


if __name__ == '__main__':
    main()
