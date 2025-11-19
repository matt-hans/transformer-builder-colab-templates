"""
Comprehensive 6-panel training visualization dashboard with drift detection.

Provides professional-grade post-training analysis visualizations with:
- Loss curves (train vs validation)
- Perplexity trends
- Accuracy metrics (if available)
- Learning rate schedule
- Gradient norm monitoring
- Training time analysis
- Drift detection visualization (NEW in v3.6)

Example:
    >>> from utils.training.metrics_tracker import MetricsTracker
    >>> from utils.training.dashboard import TrainingDashboard
    >>> from utils.training.drift_metrics import compute_dataset_profile, compare_profiles
    >>>
    >>> # After training
    >>> tracker = MetricsTracker(use_wandb=False)
    >>> # ... training loop with tracker.log_epoch() ...
    >>>
    >>> # Create standard dashboard
    >>> metrics_df = tracker.get_summary()
    >>> dashboard = TrainingDashboard(figsize=(18, 12))
    >>> fig = dashboard.plot(metrics_df, config=training_config)
    >>> dashboard.save('training_dashboard.png', dpi=150)
    >>>
    >>> # NEW: Create dashboard with drift visualization
    >>> ref_profile = compute_dataset_profile(train_dataset, task_spec)
    >>> new_profile = compute_dataset_profile(val_dataset, task_spec)
    >>> drift_comparison = compare_profiles(ref_profile, new_profile)
    >>> drift_data = {
    ...     'ref_profile': ref_profile,
    ...     'new_profile': new_profile,
    ...     'drift_scores': drift_comparison['drift_scores'],
    ...     'status': drift_comparison['status']
    ... }
    >>> fig = dashboard.plot_with_drift(metrics_df, drift_data, config=training_config)
    >>> dashboard.save('training_dashboard_with_drift.png', dpi=150)
"""

import logging
from typing import Optional, Tuple, Any, Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
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

    # ========== NEW DRIFT VISUALIZATION METHODS (v3.6) ==========

    def _plot_drift_distributions(self, ref_profile: Dict, new_profile: Dict, ax) -> None:
        """
        Plot side-by-side histograms showing reference vs current distributions.

        For text: sequence length distribution
        For vision: brightness histogram

        Args:
            ref_profile: Reference dataset profile from drift_metrics.compute_dataset_profile()
            new_profile: Current dataset profile
            ax: Matplotlib axis to plot on
        """
        # Detect modality
        if 'seq_length_hist' in ref_profile:
            # Text modality
            bins = np.array(ref_profile['seq_length_bins'])
            ref_counts = np.array(ref_profile['seq_length_hist'])
            new_counts = np.array(new_profile['seq_length_hist'])

            width = (bins[1] - bins[0]) * 0.4  # Bar width

            ax.bar(bins[:-1] - width/2, ref_counts, width=width,
                   alpha=0.6, label='Reference', color='#3498db')
            ax.bar(bins[:-1] + width/2, new_counts, width=width,
                   alpha=0.6, label='Current', color='#e74c3c')

            ax.set_xlabel('Sequence Length', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Sequence Length Distribution Shift', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        elif 'brightness_hist' in ref_profile:
            # Vision modality
            bins = np.linspace(0, 1, 6)  # 5 brightness bins
            ref_counts = np.array(ref_profile['brightness_hist'])
            new_counts = np.array(new_profile['brightness_hist'])

            bin_centers = (bins[:-1] + bins[1:]) / 2
            width = (bins[1] - bins[0]) * 0.4

            ax.bar(bin_centers - width/2, ref_counts, width=width,
                   alpha=0.6, label='Reference', color='#3498db')
            ax.bar(bin_centers + width/2, new_counts, width=width,
                   alpha=0.6, label='Current', color='#e74c3c')

            ax.set_xlabel('Brightness', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Brightness Distribution Shift', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            # Unknown modality
            ax.text(0.5, 0.5, 'Distribution data\nnot available',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title('Distribution Shift (N/A)', fontweight='bold')
            ax.axis('off')

    def _plot_drift_timeseries(self, drift_history: List[Dict], ax) -> None:
        """
        Plot drift scores over time/checkpoints.

        Args:
            drift_history: List of drift comparison results over time
                [{'epoch': 0, 'drift_scores': {...}, 'status': 'ok'}, ...]
            ax: Matplotlib axis
        """
        if not drift_history:
            ax.text(0.5, 0.5, 'No drift history available',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title('Drift Score Over Time (N/A)', fontweight='bold')
            ax.axis('off')
            return

        epochs = [entry['epoch'] for entry in drift_history]

        # Extract primary drift metric (seq_length_js or brightness_js)
        if 'seq_length_js' in drift_history[0]['drift_scores']:
            metric_key = 'seq_length_js'
            metric_label = 'Seq Length JS Distance'
        elif 'brightness_js' in drift_history[0]['drift_scores']:
            metric_key = 'brightness_js'
            metric_label = 'Brightness JS Distance'
        else:
            # Fallback to max drift
            metric_key = None
            metric_label = 'Max JS Distance'

        if metric_key:
            scores = [entry['drift_scores'].get(metric_key, 0) for entry in drift_history]
        else:
            # Use max drift from all available metrics
            scores = []
            for entry in drift_history:
                drift_vals = [v for k, v in entry['drift_scores'].items()
                             if k.endswith('_js') or k.endswith('_distance')]
                scores.append(max(drift_vals) if drift_vals else 0)

        # Plot drift scores
        ax.plot(epochs, scores, 'o-', color='#9b59b6', linewidth=2,
                markersize=6, label=metric_label)

        # Add threshold lines
        ax.axhline(y=0.1, color='#f39c12', linestyle='--', linewidth=1.5,
                   label='Warn Threshold (0.1)')
        ax.axhline(y=0.2, color='#e74c3c', linestyle='--', linewidth=1.5,
                   label='Alert Threshold (0.2)')

        # Color background regions
        ax.axhspan(0, 0.1, alpha=0.1, color='green')  # OK zone
        ax.axhspan(0.1, 0.2, alpha=0.1, color='yellow')  # Warn zone
        ax.axhspan(0.2, 1.0, alpha=0.1, color='red')  # Alert zone

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('JS Distance', fontsize=10)
        ax.set_title('Drift Score Over Time', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, min(max(scores) * 1.2, 1.0) if scores else 1.0)

    def _plot_drift_heatmap(self, drift_scores: Dict, ax) -> None:
        """
        Color-coded heatmap showing drift status for each metric.

        Args:
            drift_scores: Dict from compare_profiles()
                {'seq_length_js': 0.05, 'token_overlap': 0.95, 'output_js': 0.12, ...}
            ax: Matplotlib axis
        """
        # Define metrics to show
        metric_names = []
        statuses = []

        for metric_key in ['seq_length_js', 'brightness_js', 'token_overlap',
                           'output_js', 'output_kl', 'channel_mean_distance']:
            if metric_key not in drift_scores:
                continue

            score = drift_scores[metric_key]
            metric_names.append(metric_key.replace('_', ' ').title())

            # Classify status: ok (0), warn (1), alert (2)
            if metric_key == 'token_overlap':
                # Higher is better
                if score > 0.9:
                    statuses.append(0)  # ok
                elif score > 0.7:
                    statuses.append(1)  # warn
                else:
                    statuses.append(2)  # alert
            else:
                # Lower is better
                if score < 0.1:
                    statuses.append(0)  # ok
                elif score < 0.2:
                    statuses.append(1)  # warn
                else:
                    statuses.append(2)  # alert

        if not metric_names:
            ax.text(0.5, 0.5, 'No drift metrics available',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title('Drift Status Heatmap (N/A)', fontweight='bold')
            ax.axis('off')
            return

        # Create heatmap
        cmap = ListedColormap(['#2ecc71', '#f39c12', '#e74c3c'])  # green, yellow, red
        data = np.array(statuses).reshape(1, -1)

        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=2)

        # Labels
        ax.set_yticks([0])
        ax.set_yticklabels(['Status'])
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
        ax.set_title('Drift Status Heatmap', fontsize=11, fontweight='bold')

        # Add text annotations
        for i, (name, status) in enumerate(zip(metric_names, statuses)):
            status_text = ['✓ OK', '⚠ Warn', '✗ Alert'][status]
            color = 'white' if status == 2 else 'black'
            ax.text(i, 0, status_text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    def _plot_drift_summary(self, drift_scores: Dict, status: str, ax) -> None:
        """
        Text table showing key drift metrics.

        Args:
            drift_scores: Drift scores dict
            status: Overall status ("ok", "warn", "alert")
            ax: Matplotlib axis
        """
        ax.axis('off')

        # Build table data
        table_data = [['Metric', 'Value', 'Status']]

        # Overall status
        status_emoji = {'ok': '✅', 'warn': '⚠️', 'alert': '🚨'}[status]
        table_data.append(['Overall Status', status.upper(), status_emoji])
        table_data.append(['', '', ''])  # Separator

        # Individual metrics
        for key in ['seq_length_js', 'brightness_js', 'token_overlap',
                    'output_js', 'output_kl', 'channel_mean_distance']:
            if key not in drift_scores:
                continue

            value = drift_scores[key]
            name = key.replace('_', ' ').title()

            # Format value
            if 'overlap' in key:
                value_str = f"{value:.1%}"
                status_str = '✅' if value > 0.9 else '⚠️' if value > 0.7 else '🚨'
            else:
                value_str = f"{value:.3f}"
                status_str = '✅' if value < 0.1 else '⚠️' if value < 0.2 else '🚨'

            table_data.append([name, value_str, status_str])

        # Create table
        table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                         colWidths=[0.5, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(3):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')

        ax.set_title('Drift Metrics Summary', fontsize=11, fontweight='bold', pad=20)

    def plot_with_drift(
        self,
        metrics_df: pd.DataFrame,
        drift_data: Optional[Dict] = None,
        config: Optional[Any] = None,
        title: str = 'Training Dashboard with Drift Analysis'
    ) -> Figure:
        """
        Extended dashboard with drift visualization panels.

        Args:
            metrics_df: Training metrics DataFrame from MetricsTracker.get_summary()
            drift_data: Optional dict with:
                {
                    'ref_profile': {...},  # Reference dataset profile
                    'new_profile': {...},  # Current dataset profile
                    'drift_scores': {...}, # From compare_profiles()
                    'status': 'ok'|'warn'|'alert',
                    'drift_history': [...]  # Optional timeseries
                }
            config: TrainingConfig (optional)
            title: Dashboard title

        Returns:
            matplotlib.figure.Figure with 10-panel visualization (if drift_data provided)
            or 6-panel standard dashboard (if drift_data=None)
        """
        if drift_data is None:
            # Fall back to standard dashboard
            return self.plot(metrics_df, config, title.replace(' with Drift Analysis', ''))

        self._validate_dataframe(metrics_df)

        # Extended 10-panel layout (6 training + 4 drift)
        self.fig = plt.figure(figsize=(24, 18))
        gs = gridspec.GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3,
                               top=0.92, bottom=0.05, left=0.05, right=0.95)

        # Row 1: Loss, Perplexity, Accuracy, Learning Rate (no summary card in drift mode)
        ax_loss = self.fig.add_subplot(gs[0, 0])
        ax_perplexity = self.fig.add_subplot(gs[0, 1])
        ax_accuracy = self.fig.add_subplot(gs[0, 2])
        ax_lr = self.fig.add_subplot(gs[0, 3])

        # Row 2: Gradient/time + drift panels
        ax_gradients = self.fig.add_subplot(gs[1, 0])
        ax_time = self.fig.add_subplot(gs[1, 1])
        ax_drift_hist = self.fig.add_subplot(gs[1, 2])  # NEW: Drift histograms
        ax_drift_ts = self.fig.add_subplot(gs[1, 3])    # NEW: Drift timeseries

        # Row 3: Drift panels
        ax_drift_heatmap = self.fig.add_subplot(gs[2, 0])  # NEW: Drift heatmap
        ax_drift_summary = self.fig.add_subplot(gs[2, 1:])  # NEW: Drift summary (spans 3 columns)

        # Plot existing panels (methods from base TrainingDashboard)
        self._plot_loss_curves(metrics_df, ax_loss)
        self._plot_perplexity(metrics_df, ax_perplexity)
        self._plot_accuracy(metrics_df, ax_accuracy)
        self._plot_learning_rate(metrics_df, ax_lr)
        self._plot_gradient_norms(metrics_df, ax_gradients)
        self._plot_training_time(metrics_df, ax_time)

        # Plot NEW drift panels
        self._plot_drift_distributions(
            drift_data['ref_profile'],
            drift_data['new_profile'],
            ax_drift_hist
        )
        self._plot_drift_timeseries(
            drift_data.get('drift_history', []),
            ax_drift_ts
        )
        self._plot_drift_heatmap(drift_data['drift_scores'], ax_drift_heatmap)
        self._plot_drift_summary(
            drift_data['drift_scores'],
            drift_data['status'],
            ax_drift_summary
        )

        self.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        return self.fig
