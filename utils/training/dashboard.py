"""
Comprehensive 6-panel training visualization dashboard.

Provides professional-grade post-training analysis visualizations with:
- Loss curves (train vs validation)
- Perplexity trends
- Accuracy metrics (if available)
- Learning rate schedule
- Gradient norm monitoring
- Training time analysis

Example:
    >>> from utils.training.metrics_tracker import MetricsTracker
    >>> from utils.training.dashboard import TrainingDashboard
    >>>
    >>> # After training
    >>> tracker = MetricsTracker(use_wandb=False)
    >>> # ... training loop with tracker.log_epoch() ...
    >>>
    >>> # Create dashboard
    >>> metrics_df = tracker.get_summary()
    >>> dashboard = TrainingDashboard(figsize=(18, 12))
    >>> fig = dashboard.plot(metrics_df, config=training_config)
    >>> dashboard.save('training_dashboard.png', dpi=150)
"""

import logging
from typing import Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingDashboard:
    """Comprehensive 6-panel training visualization dashboard."""

    def __init__(self, figsize: Tuple[int, int] = (18, 12)):
        """
        Initialize dashboard with configurable figure size.

        Args:
            figsize: Figure dimensions (width, height) in inches.
                    Default (18, 12) provides good balance for 6 panels.
        """
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError(f"figsize must be tuple of 2 ints, got {figsize}")
        if figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError(f"figsize dimensions must be positive, got {figsize}")

        self.figsize = figsize
        self.fig: Optional[Figure] = None

    def plot(
        self,
        metrics_df: pd.DataFrame,
        config: Optional[Any] = None,
        title: str = 'Training Dashboard'
    ) -> Figure:
        """
        Create comprehensive 6-panel dashboard from metrics DataFrame.

        Args:
            metrics_df: DataFrame from MetricsTracker.get_summary() with columns:
                       - epoch (int, required)
                       - train/loss (float, required)
                       - val/loss (float, required)
                       - val/perplexity (float, optional)
                       - train/accuracy (float, optional)
                       - val/accuracy (float, optional)
                       - learning_rate (float, optional)
                       - gradients/pre_clip_norm (float, optional)
                       - gradients/post_clip_norm (float, optional)
                       - epoch_duration (float, optional)
            config: Optional TrainingConfig for displaying hyperparameters
            title: Dashboard title

        Returns:
            matplotlib Figure object with 6-panel visualization

        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        self._validate_dataframe(metrics_df)

        # Create figure and layout
        self.fig = plt.figure(figsize=self.figsize)
        self.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

        # Create 2x3 grid layout with space for summary card
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                               top=0.92, bottom=0.05, left=0.05, right=0.95)

        # Add summary card
        self._add_summary_card(metrics_df, config, gs[0, :])

        # Panel 1: Loss Curves (top-left)
        ax1 = self.fig.add_subplot(gs[1, 0])
        self._plot_loss_curves(metrics_df, ax1)

        # Panel 2: Perplexity (top-middle)
        ax2 = self.fig.add_subplot(gs[1, 1])
        self._plot_perplexity(metrics_df, ax2)

        # Panel 3: Accuracy (top-right)
        ax3 = self.fig.add_subplot(gs[1, 2])
        self._plot_accuracy(metrics_df, ax3)

        # Panel 4: Learning Rate (bottom-left)
        ax4 = self.fig.add_subplot(gs[2, 0])
        self._plot_learning_rate(metrics_df, ax4)

        # Panel 5: Gradient Norms (bottom-middle)
        ax5 = self.fig.add_subplot(gs[2, 1])
        self._plot_gradient_norms(metrics_df, ax5)

        # Panel 6: Training Time (bottom-right)
        ax6 = self.fig.add_subplot(gs[2, 2])
        self._plot_training_time(metrics_df, ax6)

        return self.fig

    def save(self, filepath: str, dpi: int = 150) -> None:
        """
        Save dashboard to file.

        Args:
            filepath: Output file path (supports PNG, PDF, SVG)
            dpi: Resolution for raster formats (PNG)

        Raises:
            RuntimeError: If plot() has not been called yet
            ValueError: If file format is unsupported
        """
        if self.fig is None:
            raise RuntimeError("Must call plot() before save()")

        path = Path(filepath)
        supported_formats = {'.png', '.pdf', '.svg'}
        if path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format {path.suffix}. "
                f"Use one of: {supported_formats}"
            )

        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Dashboard saved to {filepath} (dpi={dpi})")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame schema and content."""
        if df.empty:
            raise ValueError("DataFrame is empty")

        required_cols = ['epoch', 'train/loss', 'val/loss']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"DataFrame must contain: {required_cols}"
            )

        if len(df) < 1:
            raise ValueError("DataFrame must have at least 1 row")

    def _add_summary_card(
        self, df: pd.DataFrame, config: Optional[Any], gs_slot
    ) -> None:
        """Add summary card with key metrics and config."""
        ax = self.fig.add_subplot(gs_slot)
        ax.axis('off')

        # Find best epoch (min validation loss)
        best_idx = df['val/loss'].idxmin()
        best_epoch = int(df.loc[best_idx, 'epoch'])
        best_val_loss = df.loc[best_idx, 'val/loss']

        # Build summary text
        summary_parts = []

        # Config hyperparameters (if available)
        if config is not None:
            config_str = f"Config: lr={getattr(config, 'learning_rate', 'N/A')}, "
            config_str += f"batch={getattr(config, 'batch_size', 'N/A')}, "
            config_str += f"epochs={len(df)}"
            summary_parts.append(config_str)

        # Best epoch info
        summary_parts.append(f"Best Epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

        # Final metrics
        final_metrics = []
        if 'val/perplexity' in df.columns:
            final_ppl = df.loc[best_idx, 'val/perplexity']
            final_metrics.append(f"ppl={final_ppl:.2f}")
        if 'val/accuracy' in df.columns:
            final_acc = df.loc[best_idx, 'val/accuracy']
            final_metrics.append(f"acc={final_acc:.2%}")
        if final_metrics:
            summary_parts.append(f"Best Metrics: {', '.join(final_metrics)}")

        # Total training time
        if 'epoch_duration' in df.columns:
            total_time = df['epoch_duration'].sum()
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            summary_parts.append(f"Total Time: {hours}h {minutes}m")

        # Display summary
        summary_text = ' | '.join(summary_parts)
        ax.text(
            0.5, 0.5, summary_text,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    def _plot_loss_curves(self, df: pd.DataFrame, ax) -> None:
        """Panel 1: Train vs Validation loss curves."""
        epochs = df['epoch'].values
        train_loss = df['train/loss'].values
        val_loss = df['val/loss'].values

        # Plot curves
        ax.plot(epochs, train_loss, 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
        ax.plot(epochs, val_loss, 's-', label='Val Loss', color='#ff7f0e', linewidth=2)

        # Annotate best validation loss
        best_idx = df['val/loss'].idxmin()
        best_epoch = df.loc[best_idx, 'epoch']
        best_val = df.loc[best_idx, 'val/loss']
        ax.plot(best_epoch, best_val, 'r*', markersize=15, label=f'Best (epoch {int(best_epoch)})')

        # Use log scale if loss varies >10x
        loss_range = max(train_loss.max(), val_loss.max()) / min(train_loss.min(), val_loss.min())
        if loss_range > 10:
            ax.set_yscale('log')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Loss Curves', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_perplexity(self, df: pd.DataFrame, ax) -> None:
        """Panel 2: Validation perplexity."""
        epochs = df['epoch'].values

        # Compute perplexity if not present
        if 'val/perplexity' in df.columns:
            perplexity = df['val/perplexity'].values
        else:
            perplexity = np.exp(df['val/loss'].values)

        # Plot perplexity
        ax.plot(epochs, perplexity, 'o-', color='#2ca02c', linewidth=2)

        # Annotate best perplexity
        best_idx = perplexity.argmin()
        best_epoch = epochs[best_idx]
        best_ppl = perplexity[best_idx]
        ax.plot(best_epoch, best_ppl, 'r*', markersize=15, label=f'Best: {best_ppl:.2f}')

        # Reference line at perplexity=10
        if best_ppl > 10:
            ax.axhline(10, color='gray', linestyle='--', alpha=0.5, label='Baseline (10)')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Perplexity', fontweight='bold')
        ax.set_title('Perplexity (lower is better)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_accuracy(self, df: pd.DataFrame, ax) -> None:
        """Panel 3: Train vs Validation accuracy."""
        # Skip if accuracy not available
        if 'train/accuracy' not in df.columns and 'val/accuracy' not in df.columns:
            ax.text(
                0.5, 0.5, 'Accuracy metrics\nnot available',
                ha='center', va='center', fontsize=12, color='gray'
            )
            ax.set_title('Accuracy (N/A)', fontweight='bold')
            ax.axis('off')
            return

        epochs = df['epoch'].values

        # Plot train accuracy if available
        if 'train/accuracy' in df.columns:
            train_acc = df['train/accuracy'].values * 100  # Convert to percentage
            ax.plot(epochs, train_acc, 'o-', label='Train Acc', color='#1f77b4', linewidth=2)

        # Plot val accuracy if available
        if 'val/accuracy' in df.columns:
            val_acc = df['val/accuracy'].values * 100  # Convert to percentage
            ax.plot(epochs, val_acc, 's-', label='Val Acc', color='#ff7f0e', linewidth=2)

            # Annotate best validation accuracy
            best_idx = df['val/accuracy'].idxmax()
            best_epoch = df.loc[best_idx, 'epoch']
            best_val = df.loc[best_idx, 'val/accuracy'] * 100
            ax.plot(best_epoch, best_val, 'r*', markersize=15, label=f'Best: {best_val:.1f}%')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Accuracy', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_learning_rate(self, df: pd.DataFrame, ax) -> None:
        """Panel 4: Learning rate schedule."""
        if 'learning_rate' not in df.columns:
            ax.text(
                0.5, 0.5, 'Learning rate\nnot tracked',
                ha='center', va='center', fontsize=12, color='gray'
            )
            ax.set_title('Learning Rate (N/A)', fontweight='bold')
            ax.axis('off')
            return

        epochs = df['epoch'].values
        lr = df['learning_rate'].values

        # Plot LR schedule
        ax.plot(epochs, lr, 'o-', color='#d62728', linewidth=2)

        # Highlight warmup phase (first 10% of epochs where LR increases)
        warmup_cutoff = int(len(epochs) * 0.1)
        if warmup_cutoff > 1 and lr[warmup_cutoff] > lr[0]:
            ax.axvspan(epochs[0], epochs[warmup_cutoff], alpha=0.2, color='yellow', label='Warmup')

        # Use log scale if LR varies >10x
        lr_range = lr.max() / lr.min() if lr.min() > 0 else 1
        if lr_range > 10:
            ax.set_yscale('log')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontweight='bold')
        if warmup_cutoff > 1:
            ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norms(self, df: pd.DataFrame, ax) -> None:
        """Panel 5: Gradient norm monitoring."""
        if 'gradients/pre_clip_norm' not in df.columns:
            ax.text(
                0.5, 0.5, 'Gradient norms\nnot tracked',
                ha='center', va='center', fontsize=12, color='gray'
            )
            ax.set_title('Gradient Norms (N/A)', fontweight='bold')
            ax.axis('off')
            return

        epochs = df['epoch'].values
        pre_clip = df['gradients/pre_clip_norm'].values

        # Plot pre-clip norms
        ax.plot(epochs, pre_clip, 'o-', label='Pre-clip', color='#1f77b4', linewidth=2)

        # Plot post-clip norms if available
        if 'gradients/post_clip_norm' in df.columns:
            post_clip = df['gradients/post_clip_norm'].values
            ax.plot(epochs, post_clip, 's-', label='Post-clip', color='#ff7f0e', linewidth=2)

            # Clip threshold (inferred from difference)
            clip_threshold = pre_clip.max()  # Conservative estimate
            ax.axhline(clip_threshold, color='red', linestyle='--', alpha=0.7, label=f'Clip threshold')

        # Warning zone for gradient explosion (norms >5.0)
        if pre_clip.max() > 5.0:
            ax.axhspan(5.0, pre_clip.max() * 1.1, alpha=0.2, color='red', label='Warning zone')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Gradient Norm', fontweight='bold')
        ax.set_title('Gradient Norms', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_training_time(self, df: pd.DataFrame, ax) -> None:
        """Panel 6: Epoch duration analysis."""
        if 'epoch_duration' not in df.columns:
            ax.text(
                0.5, 0.5, 'Training time\nnot tracked',
                ha='center', va='center', fontsize=12, color='gray'
            )
            ax.set_title('Training Time (N/A)', fontweight='bold')
            ax.axis('off')
            return

        epochs = df['epoch'].values
        durations = df['epoch_duration'].values

        # Bar chart of epoch durations
        ax.bar(epochs, durations, color='#9467bd', alpha=0.7)

        # Average time per epoch
        avg_duration = durations.mean()
        ax.axhline(avg_duration, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_duration:.1f}s')

        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Duration (seconds)', fontweight='bold')
        ax.set_title('Training Time per Epoch', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
