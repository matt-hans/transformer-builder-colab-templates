"""
Real-time training visualization for Jupyter/Colab notebooks.

Provides live-updating plots during training for immediate feedback on:
- Loss curves (train vs validation)
- Perplexity trends
- Accuracy progression
- Learning rate schedule
- Gradient norms

Usage:
    plotter = LivePlotter(metrics=['loss', 'perplexity', 'accuracy'])

    for epoch in range(n_epochs):
        # ... training ...
        plotter.update(epoch, train_metrics, val_metrics)
"""

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import numpy as np
from typing import List, Dict, Optional


class LivePlotter:
    """
    Real-time training curve plotter with auto-refresh.

    Designed for Jupyter/Colab environments where plots can be dynamically
    updated during training. Automatically clears and redraws plots each epoch.

    Attributes:
        metrics: List of metric names to plot
        figsize: Figure size tuple (width, height)
        history: Dictionary tracking metric values over epochs
        epochs: List of epoch numbers
    """

    def __init__(
        self,
        metrics: List[str] = ['loss', 'perplexity', 'accuracy'],
        figsize: tuple = (18, 5),
        style: str = 'whitegrid'
    ):
        """
        Initialize live plotter.

        Args:
            metrics: List of metrics to plot (e.g., ['loss', 'accuracy'])
            figsize: Figure dimensions (width, height) in inches
            style: Matplotlib/seaborn style ('whitegrid', 'darkgrid', 'white', etc.)
        """
        self.metrics = metrics
        self.figsize = figsize
        self.style = style

        # Initialize history storage
        self.history = {m: {'train': [], 'val': []} for m in metrics}
        self.epochs = []

        # Track best values for annotations
        self.best_values = {m: {'val': float('inf'), 'epoch': 0} for m in metrics}

        # Set plotting style
        try:
            import seaborn as sns
            sns.set_style(style)
        except ImportError:
            # Seaborn not available, use default matplotlib style
            plt.style.use('seaborn-v0_8-whitegrid' if style == 'whitegrid' else 'default')

    def update(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """
        Update plots with new epoch data.

        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics {'loss': 2.5, 'accuracy': 0.85}
            val_metrics: Dictionary of validation metrics {'loss': 2.3, 'accuracy': 0.87}
        """
        self.epochs.append(epoch)

        # Update history for each metric
        for metric in self.metrics:
            # Handle different metric naming conventions
            train_key = metric if metric in train_metrics else f'train/{metric}'
            val_key = metric if metric in val_metrics else f'val/{metric}'

            if train_key in train_metrics:
                self.history[metric]['train'].append(train_metrics[train_key])
            if val_key in val_metrics:
                self.history[metric]['val'].append(val_metrics[val_key])

                # Track best validation value
                val_value = val_metrics[val_key]
                if val_value < self.best_values[metric]['val']:
                    self.best_values[metric]['val'] = val_value
                    self.best_values[metric]['epoch'] = epoch

        # Render updated plots
        self._render()

    def _render(self):
        """Redraw all plots with current history."""
        clear_output(wait=True)

        n_metrics = len(self.metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=self.figsize)

        # Handle single metric case (axes is not a list)
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]

            # Plot train curve
            if self.history[metric]['train']:
                ax.plot(
                    self.epochs,
                    self.history[metric]['train'],
                    marker='o',
                    label='Train',
                    linewidth=2,
                    markersize=6,
                    alpha=0.8
                )

            # Plot validation curve
            if self.history[metric]['val']:
                ax.plot(
                    self.epochs,
                    self.history[metric]['val'],
                    marker='s',
                    label='Validation',
                    linewidth=2,
                    markersize=6,
                    alpha=0.8
                )

                # Annotate best validation point
                best_epoch = self.best_values[metric]['epoch']
                best_value = self.best_values[metric]['val']

                if best_epoch in self.epochs:
                    best_idx = self.epochs.index(best_epoch)
                    ax.annotate(
                        f'Best\n(Epoch {best_epoch})',
                        xy=(best_epoch, best_value),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2),
                        fontsize=9
                    )

            # Styling
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.capitalize()} Curve', fontsize=12, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add subtle background color
            ax.set_facecolor('#f9f9f9')

        plt.tight_layout()
        display(fig)
        plt.close()  # Prevent duplicate displays

    def save(self, filepath: str = 'training_curves.png', dpi: int = 150):
        """
        Save current plots to file.

        Args:
            filepath: Output file path
            dpi: Resolution in dots per inch
        """
        n_metrics = len(self.metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=self.figsize)

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]

            if self.history[metric]['train']:
                ax.plot(self.epochs, self.history[metric]['train'],
                       marker='o', label='Train', linewidth=2)
            if self.history[metric]['val']:
                ax.plot(self.epochs, self.history[metric]['val'],
                       marker='s', label='Validation', linewidth=2)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {filepath}")


class CompactLivePlotter:
    """
    Compact single-plot version for space-constrained environments.

    Plots only loss curve in a smaller figure, useful for quick monitoring
    without taking up too much notebook space.
    """

    def __init__(self, figsize: tuple = (10, 4)):
        """
        Initialize compact plotter.

        Args:
            figsize: Figure dimensions (width, height)
        """
        self.figsize = figsize
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def update(self, epoch: int, train_loss: float, val_loss: float):
        """
        Update plot with new epoch data.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Track best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        # Render
        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(self.epochs, self.train_losses, 'o-', label='Train Loss', linewidth=2)
        ax.plot(self.epochs, self.val_losses, 's-', label='Val Loss', linewidth=2)

        # Annotate best
        if self.best_epoch in self.epochs:
            ax.annotate(
                f'Best: {self.best_val_loss:.4f}',
                xy=(self.best_epoch, self.best_val_loss),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
                fontsize=9
            )

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Progress', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        display(fig)
        plt.close()
