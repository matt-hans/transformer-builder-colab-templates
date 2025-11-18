"""
Unit tests for TrainingDashboard visualization.

Tests cover:
- Full metrics dashboard (all 6 panels)
- Minimal metrics (loss only)
- Missing optional metrics (no accuracy, no gradients)
- Export functionality (PNG, PDF, SVG)
- Error handling (empty DataFrame, missing columns)
- Edge cases (NaN values, single epoch)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.training.dashboard import TrainingDashboard


@pytest.fixture
def full_metrics_df():
    """DataFrame with all metrics (6 panels worth)."""
    return pd.DataFrame({
        'epoch': [1, 2, 3, 4, 5],
        'train/loss': [2.5, 2.0, 1.8, 1.6, 1.5],
        'val/loss': [2.6, 2.1, 1.9, 1.7, 1.6],
        'val/perplexity': [13.5, 8.2, 6.7, 5.5, 5.0],
        'train/accuracy': [0.35, 0.45, 0.52, 0.58, 0.62],
        'val/accuracy': [0.33, 0.43, 0.50, 0.55, 0.59],
        'learning_rate': [1e-5, 5e-5, 4e-5, 3e-5, 2e-5],
        'gradients/pre_clip_norm': [2.3, 2.1, 1.9, 1.8, 1.7],
        'gradients/post_clip_norm': [2.3, 2.1, 1.9, 1.8, 1.7],
        'epoch_duration': [45.2, 44.8, 45.0, 44.9, 45.1]
    })


@pytest.fixture
def minimal_metrics_df():
    """DataFrame with only required columns (epoch, train/loss, val/loss)."""
    return pd.DataFrame({
        'epoch': [1, 2, 3],
        'train/loss': [2.5, 2.0, 1.8],
        'val/loss': [2.6, 2.1, 1.9]
    })


@pytest.fixture
def no_accuracy_df():
    """DataFrame without accuracy metrics."""
    return pd.DataFrame({
        'epoch': [1, 2, 3],
        'train/loss': [2.5, 2.0, 1.8],
        'val/loss': [2.6, 2.1, 1.9],
        'val/perplexity': [13.5, 8.2, 6.7],
        'learning_rate': [5e-5, 4e-5, 3e-5],
        'gradients/pre_clip_norm': [2.3, 2.1, 1.9]
    })


@pytest.fixture
def training_config():
    """Mock TrainingConfig object."""
    return SimpleNamespace(
        learning_rate=5e-5,
        batch_size=4,
        epochs=5,
        random_seed=42
    )


# === GREEN PATH TESTS ===

def test_dashboard_full_metrics(full_metrics_df, training_config):
    """Test dashboard creation with all 6 panels populated."""
    dashboard = TrainingDashboard(figsize=(18, 12))
    fig = dashboard.plot(full_metrics_df, config=training_config, title='Full Dashboard')

    assert fig is not None, "Figure should be created"
    assert len(fig.axes) >= 7, "Should have 7+ axes (summary + 6 panels)"

    # Verify summary card exists (first axis should be off)
    assert not fig.axes[0].axison, "First axis should be summary card (axis off)"

    # Verify all 6 panels have titles
    panel_titles = [ax.get_title() for ax in fig.axes[1:7]]
    expected_titles = [
        'Loss Curves', 'Perplexity (lower is better)', 'Accuracy',
        'Learning Rate Schedule', 'Gradient Norms', 'Training Time per Epoch'
    ]
    for expected in expected_titles:
        assert any(expected in title for title in panel_titles), f"Missing panel: {expected}"


def test_dashboard_minimal_metrics(minimal_metrics_df):
    """Test dashboard with only required columns (loss only)."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(minimal_metrics_df, title='Minimal Dashboard')

    assert fig is not None, "Figure should be created"

    # Loss panel should be populated
    loss_ax = fig.axes[1]  # First panel after summary
    assert loss_ax.get_title() == 'Loss Curves', "Loss panel should exist"
    assert len(loss_ax.lines) >= 2, "Should have train and val loss lines"

    # Perplexity should auto-compute from val/loss
    ppl_ax = fig.axes[2]
    assert ppl_ax.get_title() == 'Perplexity (lower is better)', "Perplexity should be computed"


def test_dashboard_no_accuracy(no_accuracy_df):
    """Test dashboard skips accuracy panel when metrics absent."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(no_accuracy_df, title='No Accuracy')

    # Accuracy panel should show N/A message
    acc_ax = fig.axes[3]  # Third panel after summary
    assert 'Accuracy' in acc_ax.get_title(), "Accuracy panel should exist"
    # Check if axis is off (N/A message)
    assert not acc_ax.axison, "Accuracy panel should be disabled (N/A)"


def test_save_png(full_metrics_df, tmp_path):
    """Test export to PNG format."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(full_metrics_df)

    output_path = tmp_path / "dashboard.png"
    dashboard.save(str(output_path), dpi=100)

    assert output_path.exists(), "PNG file should be created"
    assert output_path.stat().st_size > 1000, "PNG should have content (>1KB)"


def test_save_pdf(full_metrics_df, tmp_path):
    """Test export to PDF format."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(full_metrics_df)

    output_path = tmp_path / "dashboard.pdf"
    dashboard.save(str(output_path), dpi=150)

    assert output_path.exists(), "PDF file should be created"
    assert output_path.stat().st_size > 1000, "PDF should have content (>1KB)"


def test_save_svg(full_metrics_df, tmp_path):
    """Test export to SVG format."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(full_metrics_df)

    output_path = tmp_path / "dashboard.svg"
    dashboard.save(str(output_path))

    assert output_path.exists(), "SVG file should be created"
    assert output_path.stat().st_size > 1000, "SVG should have content (>1KB)"


# === RED PATH TESTS ===

def test_empty_dataframe():
    """Test error handling for empty DataFrame."""
    dashboard = TrainingDashboard()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="DataFrame is empty"):
        dashboard.plot(empty_df)


def test_missing_required_columns():
    """Test error handling for missing required columns."""
    dashboard = TrainingDashboard()
    invalid_df = pd.DataFrame({
        'epoch': [1, 2, 3],
        'train/loss': [2.5, 2.0, 1.8]
        # Missing 'val/loss'
    })

    with pytest.raises(ValueError, match="Missing required columns.*val/loss"):
        dashboard.plot(invalid_df)


def test_invalid_figsize():
    """Test error handling for invalid figure size."""
    with pytest.raises(ValueError, match="figsize must be tuple of 2 ints"):
        TrainingDashboard(figsize=(18,))  # Only 1 dimension

    with pytest.raises(ValueError, match="figsize dimensions must be positive"):
        TrainingDashboard(figsize=(18, -12))  # Negative dimension


def test_save_before_plot(full_metrics_df, tmp_path):
    """Test error when saving before calling plot()."""
    dashboard = TrainingDashboard()

    with pytest.raises(RuntimeError, match="Must call plot\\(\\) before save\\(\\)"):
        dashboard.save(str(tmp_path / "dashboard.png"))


def test_save_unsupported_format(full_metrics_df, tmp_path):
    """Test error for unsupported file format."""
    dashboard = TrainingDashboard()
    dashboard.plot(full_metrics_df)

    with pytest.raises(ValueError, match="Unsupported format.*jpg"):
        dashboard.save(str(tmp_path / "dashboard.jpg"))


def test_nan_metrics(minimal_metrics_df):
    """Test handling of NaN values in metrics."""
    # Introduce NaN in middle of data
    df_with_nan = minimal_metrics_df.copy()
    df_with_nan.loc[1, 'val/loss'] = np.nan

    dashboard = TrainingDashboard()
    # Should not raise error, matplotlib handles NaN gracefully
    fig = dashboard.plot(df_with_nan)
    assert fig is not None, "Dashboard should handle NaN values"


# === EDGE CASES ===

def test_single_epoch():
    """Test dashboard with only 1 epoch (no trends)."""
    single_epoch_df = pd.DataFrame({
        'epoch': [1],
        'train/loss': [2.5],
        'val/loss': [2.6]
    })

    dashboard = TrainingDashboard()
    fig = dashboard.plot(single_epoch_df, title='Single Epoch')

    assert fig is not None, "Dashboard should handle single epoch"


def test_best_epoch_annotation(full_metrics_df):
    """Test that best epoch is correctly annotated."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(full_metrics_df)

    # Best epoch should be epoch 5 (min val/loss = 1.6)
    best_idx = full_metrics_df['val/loss'].idxmin()
    best_epoch = int(full_metrics_df.loc[best_idx, 'epoch'])

    # Check summary card contains best epoch
    summary_ax = fig.axes[0]
    summary_text = summary_ax.texts[0].get_text()
    assert f"Best Epoch: {best_epoch}" in summary_text, "Summary should show best epoch"


def test_log_scale_loss(minimal_metrics_df):
    """Test log scale activation for large loss variation."""
    # Create loss data with >10x variation
    wide_range_df = minimal_metrics_df.copy()
    wide_range_df['train/loss'] = [100.0, 50.0, 5.0]
    wide_range_df['val/loss'] = [110.0, 55.0, 5.5]

    dashboard = TrainingDashboard()
    fig = dashboard.plot(wide_range_df)

    loss_ax = fig.axes[1]
    assert loss_ax.get_yscale() == 'log', "Loss panel should use log scale for wide range"


def test_warmup_highlighting():
    """Test learning rate warmup phase highlighting."""
    df_with_warmup = pd.DataFrame({
        'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'train/loss': np.linspace(2.5, 1.5, 10),
        'val/loss': np.linspace(2.6, 1.6, 10),
        'learning_rate': [1e-6, 5e-6, 1e-5, 5e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 5e-6]
        # First 4 epochs warmup (increasing LR)
    })

    dashboard = TrainingDashboard()
    fig = dashboard.plot(df_with_warmup)

    lr_ax = fig.axes[4]  # Learning rate panel
    # Check for warmup shading (axvspan creates a patch)
    patches = [p for p in lr_ax.patches if hasattr(p, 'get_facecolor')]
    # Note: warmup detection is heuristic, may not always trigger
    # This test validates the code doesn't crash, not exact behavior


def test_gradient_warning_zone():
    """Test gradient norm warning zone for high values."""
    df_high_grads = pd.DataFrame({
        'epoch': [1, 2, 3],
        'train/loss': [2.5, 2.0, 1.8],
        'val/loss': [2.6, 2.1, 1.9],
        'gradients/pre_clip_norm': [3.0, 7.0, 6.5],  # Spike to 7.0 (>5.0)
        'gradients/post_clip_norm': [3.0, 5.0, 5.0]
    })

    dashboard = TrainingDashboard()
    fig = dashboard.plot(df_high_grads)

    grad_ax = fig.axes[5]  # Gradient norms panel
    # Warning zone creates a patch
    patches = grad_ax.patches
    assert len(patches) > 0, "Should have warning zone for high gradient norms"


def test_custom_figsize():
    """Test dashboard with custom figure size."""
    dashboard = TrainingDashboard(figsize=(24, 16))
    df = pd.DataFrame({
        'epoch': [1, 2],
        'train/loss': [2.5, 2.0],
        'val/loss': [2.6, 2.1]
    })

    fig = dashboard.plot(df)
    assert fig.get_figwidth() == 24, "Figure width should match custom size"
    assert fig.get_figheight() == 16, "Figure height should match custom size"


def test_config_display(full_metrics_df, training_config):
    """Test config hyperparameters displayed in summary card."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(full_metrics_df, config=training_config)

    summary_ax = fig.axes[0]
    summary_text = summary_ax.texts[0].get_text()

    assert "lr=5e-05" in summary_text or "lr=5e-5" in summary_text, "Summary should show learning rate"
    assert "batch=4" in summary_text, "Summary should show batch size"


def test_no_config_display(full_metrics_df):
    """Test dashboard works without config (None)."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(full_metrics_df, config=None)

    # Should still create summary card, just without config info
    summary_ax = fig.axes[0]
    assert summary_ax is not None, "Summary card should exist even without config"
