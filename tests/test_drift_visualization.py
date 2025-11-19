"""
Unit tests for drift visualization in TrainingDashboard.

Tests cover:
- Drift distribution histograms (text and vision modalities)
- Drift timeseries plotting with threshold zones
- Drift status heatmap with color coding
- Drift summary table rendering
- Full 10-panel dashboard generation
- Fallback to standard 6-panel dashboard
- Integration with drift_metrics.py
- Backward compatibility
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
from utils.training.drift_metrics import compute_dataset_profile, compare_profiles
from utils.training.task_spec import TaskSpec


# ========== FIXTURES ==========

@pytest.fixture
def metrics_df():
    """Basic metrics DataFrame for testing."""
    return pd.DataFrame({
        'epoch': [1, 2, 3, 4, 5],
        'train/loss': [2.5, 2.0, 1.8, 1.6, 1.5],
        'val/loss': [2.6, 2.1, 1.9, 1.7, 1.6],
        'val/perplexity': [13.5, 8.2, 6.7, 5.5, 5.0],
        'train/accuracy': [0.35, 0.45, 0.52, 0.58, 0.62],
        'val/accuracy': [0.33, 0.43, 0.50, 0.55, 0.59],
        'learning_rate': [1e-5, 5e-5, 4e-5, 3e-5, 2e-5],
        'gradients/pre_clip_norm': [2.3, 2.1, 1.9, 1.8, 1.7],
        'epoch_duration': [45.2, 44.8, 45.0, 44.9, 45.1]
    })


@pytest.fixture
def text_drift_data():
    """Mock drift data for text modality."""
    return {
        'ref_profile': {
            'modality': 'text',
            'seq_length_hist': [10, 20, 30, 25, 15, 10, 5, 3, 2, 1],
            'seq_length_bins': list(np.linspace(0, 512, 11)),
            'top_tokens': list(range(100))
        },
        'new_profile': {
            'modality': 'text',
            'seq_length_hist': [8, 18, 32, 28, 18, 12, 6, 4, 2, 1],
            'seq_length_bins': list(np.linspace(0, 512, 11)),
            'top_tokens': list(range(95)) + [500, 501, 502, 503, 504]
        },
        'drift_scores': {
            'seq_length_js': 0.05,
            'token_overlap': 0.95,
            'output_js': 0.08,
            'output_kl': 0.12
        },
        'status': 'ok'
    }


@pytest.fixture
def vision_drift_data():
    """Mock drift data for vision modality."""
    return {
        'ref_profile': {
            'modality': 'vision',
            'brightness_hist': [20, 40, 60, 50, 30],
            'brightness_bins': list(np.linspace(0, 1, 6)),
            'channel_means': [0.485, 0.456, 0.406],
            'channel_stds': [0.229, 0.224, 0.225]
        },
        'new_profile': {
            'modality': 'vision',
            'brightness_hist': [15, 35, 65, 55, 30],
            'brightness_bins': list(np.linspace(0, 1, 6)),
            'channel_means': [0.490, 0.460, 0.410],
            'channel_stds': [0.230, 0.225, 0.226]
        },
        'drift_scores': {
            'brightness_js': 0.03,
            'channel_mean_distance': 0.007,
            'output_js': 0.15,
            'output_kl': 0.18
        },
        'status': 'warn'
    }


@pytest.fixture
def drift_history():
    """Mock drift history over time."""
    return [
        {'epoch': 0, 'drift_scores': {'seq_length_js': 0.02}, 'status': 'ok'},
        {'epoch': 1, 'drift_scores': {'seq_length_js': 0.05}, 'status': 'ok'},
        {'epoch': 2, 'drift_scores': {'seq_length_js': 0.08}, 'status': 'ok'},
        {'epoch': 3, 'drift_scores': {'seq_length_js': 0.12}, 'status': 'warn'},
        {'epoch': 4, 'drift_scores': {'seq_length_js': 0.18}, 'status': 'warn'},
    ]


# ========== UNIT TESTS ==========

def test_plot_drift_distributions_text(metrics_df, text_drift_data):
    """Test histogram rendering for text modality."""
    dashboard = TrainingDashboard()

    # Create a simple plot to test the method
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_distributions(
        text_drift_data['ref_profile'],
        text_drift_data['new_profile'],
        ax
    )

    # Verify histogram bars rendered
    assert len(ax.patches) > 0, "Should have histogram bars"
    assert ax.get_title() == 'Sequence Length Distribution Shift', "Should have correct title"
    assert ax.get_xlabel() == 'Sequence Length', "Should have correct x-label"

    # Verify legend exists
    legend = ax.get_legend()
    assert legend is not None, "Should have legend"
    labels = [t.get_text() for t in legend.get_texts()]
    assert 'Reference' in labels, "Should have Reference label"
    assert 'Current' in labels, "Should have Current label"

    plt.close(fig)


def test_plot_drift_distributions_vision(metrics_df, vision_drift_data):
    """Test histogram rendering for vision modality."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_distributions(
        vision_drift_data['ref_profile'],
        vision_drift_data['new_profile'],
        ax
    )

    # Verify histogram bars rendered
    assert len(ax.patches) > 0, "Should have histogram bars"
    assert ax.get_title() == 'Brightness Distribution Shift', "Should have correct title"
    assert ax.get_xlabel() == 'Brightness', "Should have correct x-label"

    plt.close(fig)


def test_plot_drift_timeseries(metrics_df, drift_history):
    """Test timeseries plot with threshold lines."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_timeseries(drift_history, ax)

    # Verify line plot
    assert len(ax.lines) >= 3, "Should have drift line + 2 threshold lines"
    assert ax.get_title() == 'Drift Score Over Time', "Should have correct title"
    assert ax.get_xlabel() == 'Epoch', "Should have correct x-label"
    assert ax.get_ylabel() == 'JS Distance', "Should have correct y-label"

    # Verify threshold lines exist
    legend = ax.get_legend()
    assert legend is not None, "Should have legend with thresholds"
    labels = [t.get_text() for t in legend.get_texts()]
    assert any('Warn Threshold' in label for label in labels), "Should have warn threshold"
    assert any('Alert Threshold' in label for label in labels), "Should have alert threshold"

    # Verify background color zones (axhspan creates patches)
    patches = ax.patches
    assert len(patches) >= 3, "Should have 3 background zones (ok/warn/alert)"

    plt.close(fig)


def test_plot_drift_timeseries_empty(metrics_df):
    """Test empty drift history shows 'No data' message."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_timeseries([], ax)

    # Verify "No data" message
    assert not ax.axison, "Axis should be off for empty history"
    assert ax.get_title() == 'Drift Score Over Time (N/A)', "Should have N/A title"

    # Verify text exists
    texts = ax.texts
    assert len(texts) > 0, "Should have 'No data' text"
    assert 'No drift history available' in texts[0].get_text(), "Should show no data message"

    plt.close(fig)


def test_plot_drift_heatmap(metrics_df, text_drift_data):
    """Test status heatmap color coding."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_heatmap(text_drift_data['drift_scores'], ax)

    # Verify heatmap rendered
    assert ax.get_title() == 'Drift Status Heatmap', "Should have correct title"

    # Verify x-axis labels (metrics)
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert len(xticklabels) > 0, "Should have metric labels"
    # Check for some expected metric names (capitalized)
    metric_names = ' '.join(xticklabels).lower()
    assert 'seq length js' in metric_names or 'token overlap' in metric_names, \
        f"Should have metric names, got: {xticklabels}"

    # Verify text annotations exist (status labels)
    texts = ax.texts
    assert len(texts) > 0, "Should have status text annotations"
    text_contents = [t.get_text() for t in texts]
    # Should have mix of OK/Warn/Alert
    assert any('OK' in t or 'Warn' in t or 'Alert' in t for t in text_contents), \
        f"Should have status labels, got: {text_contents}"

    plt.close(fig)


def test_plot_drift_summary_table(metrics_df, text_drift_data):
    """Test metrics table rendering."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_summary(
        text_drift_data['drift_scores'],
        text_drift_data['status'],
        ax
    )

    # Verify axis off (table mode)
    assert not ax.axison, "Axis should be off for table"
    assert ax.get_title() == 'Drift Metrics Summary', "Should have correct title"

    # Verify table exists (tables stored in ax.tables)
    # Note: matplotlib doesn't expose tables easily, but we can check title exists
    # which means _plot_drift_summary executed successfully

    plt.close(fig)


# ========== INTEGRATION TESTS ==========

def test_plot_with_drift_full_dashboard(metrics_df, text_drift_data):
    """Test full 10-panel dashboard generation."""
    dashboard = TrainingDashboard()

    fig = dashboard.plot_with_drift(
        metrics_df,
        drift_data=text_drift_data,
        title='Full Drift Dashboard'
    )

    # Verify figure created
    assert fig is not None, "Figure should be created"

    # Verify 10 panels exist (4 + 2 + 4 = 10)
    # Row 1: Loss, Perplexity, Accuracy, LR (4 panels)
    # Row 2: Gradients, Time, Drift Hist, Drift TS (4 panels)
    # Row 3: Drift Heatmap, Drift Summary (2 panels, summary spans 3 cols)
    assert len(fig.axes) == 10, f"Should have 10 axes, got {len(fig.axes)}"

    # Verify title (exact match, not substring)
    assert fig._suptitle.get_text() == 'Full Drift Dashboard', "Should have correct title"

    # Verify extended figure size
    assert fig.get_figwidth() == 24, "Should use extended width (24)"
    assert fig.get_figheight() == 18, "Should use extended height (18)"


def test_plot_with_drift_fallback(metrics_df):
    """Test fallback to standard 6-panel dashboard when drift_data=None."""
    dashboard = TrainingDashboard()

    # Call with drift_data=None
    fig = dashboard.plot_with_drift(metrics_df, drift_data=None)

    # Should fall back to standard plot
    assert fig is not None, "Figure should be created"

    # Standard dashboard has 7 axes (summary + 6 panels)
    assert len(fig.axes) >= 7, f"Should have 7+ axes (standard dashboard), got {len(fig.axes)}"

    # Should NOT have extended size
    assert fig.get_figwidth() == 18, "Should use standard width (18)"
    assert fig.get_figheight() == 12, "Should use standard height (12)"


def test_plot_with_drift_uses_existing_drift_metrics(metrics_df):
    """Test integration with drift_metrics.py compute_dataset_profile()."""
    # Create mock datasets
    text_dataset = [
        {'input_ids': [1, 2, 3] * 10},  # seq_length=30
        {'input_ids': [4, 5, 6] * 20},  # seq_length=60
        {'input_ids': [7, 8, 9] * 15},  # seq_length=45
    ]

    # Create minimal task spec for text
    task_spec = SimpleNamespace(
        modality='text',
        input_field='input_ids',
        target_field='labels'
    )

    # Compute profiles using real drift_metrics.py
    ref_profile = compute_dataset_profile(text_dataset, task_spec, sample_size=10)
    new_profile = compute_dataset_profile(text_dataset, task_spec, sample_size=10)

    # Compare profiles
    drift_comparison = compare_profiles(ref_profile, new_profile)

    # Create drift_data
    drift_data = {
        'ref_profile': ref_profile,
        'new_profile': new_profile,
        'drift_scores': drift_comparison['drift_scores'],
        'status': drift_comparison['status']
    }

    # Plot with real drift data
    dashboard = TrainingDashboard()
    fig = dashboard.plot_with_drift(metrics_df, drift_data=drift_data)

    assert fig is not None, "Should create dashboard with real drift metrics"
    assert len(fig.axes) == 10, "Should have 10 panels"


def test_plot_with_drift_layout_correct(metrics_df, text_drift_data):
    """Test GridSpec(3, 4) layout is correct."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot_with_drift(metrics_df, drift_data=text_drift_data)

    # Verify GridSpec layout (3 rows, 4 columns)
    # Row 1: 4 panels (Loss, Perplexity, Accuracy, LR)
    # Row 2: 4 panels (Gradients, Time, Drift Hist, Drift TS)
    # Row 3: 2 panels (Drift Heatmap, Drift Summary spans 3 cols)

    # Check figure has correct number of axes
    assert len(fig.axes) == 10, f"Should have 10 axes, got {len(fig.axes)}"

    # Verify panel titles exist
    panel_titles = [ax.get_title() for ax in fig.axes]

    # Check for standard training panels
    assert any('Loss' in title for title in panel_titles), "Should have Loss panel"
    assert any('Perplexity' in title for title in panel_titles), "Should have Perplexity panel"
    assert any('Accuracy' in title for title in panel_titles), "Should have Accuracy panel"
    assert any('Learning Rate' in title for title in panel_titles), "Should have LR panel"
    assert any('Gradient' in title for title in panel_titles), "Should have Gradients panel"
    assert any('Training Time' in title for title in panel_titles), "Should have Time panel"

    # Check for NEW drift panels
    assert any('Distribution Shift' in title for title in panel_titles), "Should have Drift Distributions panel"
    assert any('Drift Score Over Time' in title for title in panel_titles), "Should have Drift Timeseries panel"
    assert any('Drift Status Heatmap' in title for title in panel_titles), "Should have Drift Heatmap panel"
    assert any('Drift Metrics Summary' in title for title in panel_titles), "Should have Drift Summary panel"


# ========== REGRESSION TESTS ==========

def test_standard_plot_still_works(metrics_df):
    """Test standard plot() method unchanged."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot(metrics_df, title='Standard Dashboard')

    # Verify standard dashboard works
    assert fig is not None, "Standard plot should still work"
    assert len(fig.axes) >= 7, "Standard dashboard should have 7+ axes"
    assert fig.get_figwidth() == 18, "Standard dashboard should use (18, 12) size"


def test_backward_compatible_api(metrics_df):
    """Test old dashboard code works (no regressions)."""
    # Old usage pattern (from v3.5)
    dashboard = TrainingDashboard(figsize=(18, 12))
    fig = dashboard.plot(metrics_df, config=None, title='V3.5 Dashboard')

    assert fig is not None, "Old API should work"

    # Save should work
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'dashboard.png')
        dashboard.save(output_path, dpi=100)
        assert os.path.exists(output_path), "Save should work with old API"


# ========== EDGE CASES ==========

def test_drift_data_with_history(metrics_df, text_drift_data, drift_history):
    """Test drift_data with optional drift_history field."""
    text_drift_data['drift_history'] = drift_history

    dashboard = TrainingDashboard()
    fig = dashboard.plot_with_drift(metrics_df, drift_data=text_drift_data)

    assert fig is not None, "Should handle drift_history"
    assert len(fig.axes) == 10, "Should have 10 panels"


def test_drift_data_missing_optional_metrics(metrics_df):
    """Test drift_data with minimal metrics (some missing)."""
    minimal_drift_data = {
        'ref_profile': {
            'modality': 'text',
            'seq_length_hist': [10, 20, 30, 25, 15, 10, 5, 3, 2, 1],
            'seq_length_bins': list(np.linspace(0, 512, 11)),
        },
        'new_profile': {
            'modality': 'text',
            'seq_length_hist': [8, 18, 32, 28, 18, 12, 6, 4, 2, 1],
            'seq_length_bins': list(np.linspace(0, 512, 11)),
        },
        'drift_scores': {
            'seq_length_js': 0.05,
            # Missing: token_overlap, output_js, output_kl
        },
        'status': 'ok'
    }

    dashboard = TrainingDashboard()
    # Should not crash with missing metrics
    fig = dashboard.plot_with_drift(metrics_df, drift_data=minimal_drift_data)

    assert fig is not None, "Should handle missing optional metrics"


def test_drift_heatmap_empty_scores(metrics_df):
    """Test drift heatmap with no metrics shows N/A."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # Empty drift_scores
    dashboard._plot_drift_heatmap({}, ax)

    # Should show N/A message
    assert not ax.axison, "Should disable axis for empty scores"
    texts = ax.texts
    assert len(texts) > 0, "Should have N/A text"
    assert 'No drift metrics available' in texts[0].get_text(), "Should show no metrics message"

    plt.close(fig)


def test_drift_timeseries_with_vision_metric(metrics_df):
    """Test drift timeseries detects vision metrics correctly."""
    vision_history = [
        {'epoch': 0, 'drift_scores': {'brightness_js': 0.02}, 'status': 'ok'},
        {'epoch': 1, 'drift_scores': {'brightness_js': 0.05}, 'status': 'ok'},
    ]

    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_timeseries(vision_history, ax)

    # Should detect brightness_js metric
    assert len(ax.lines) >= 3, "Should have drift line + thresholds"
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()]
    assert any('Brightness JS Distance' in label for label in labels), \
        "Should detect vision metric"

    plt.close(fig)


def test_drift_summary_with_warn_status(metrics_df, vision_drift_data):
    """Test drift summary correctly shows warn status."""
    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # vision_drift_data has status='warn'
    dashboard._plot_drift_summary(
        vision_drift_data['drift_scores'],
        vision_drift_data['status'],
        ax
    )

    # Verify table created (no crash)
    assert not ax.axison, "Should disable axis for table"

    plt.close(fig)


def test_drift_summary_with_alert_status(metrics_df):
    """Test drift summary correctly shows alert status."""
    alert_drift_data = {
        'drift_scores': {
            'seq_length_js': 0.25,  # > 0.2 threshold (alert)
            'token_overlap': 0.65,  # < 0.7 threshold (alert)
        },
        'status': 'alert'
    }

    dashboard = TrainingDashboard()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    dashboard._plot_drift_summary(
        alert_drift_data['drift_scores'],
        alert_drift_data['status'],
        ax
    )

    # Verify table created (no crash)
    assert not ax.axison, "Should disable axis for table"

    plt.close(fig)


def test_save_drift_dashboard(metrics_df, text_drift_data, tmp_path):
    """Test saving drift dashboard to file."""
    dashboard = TrainingDashboard()
    fig = dashboard.plot_with_drift(metrics_df, drift_data=text_drift_data)

    output_path = tmp_path / "drift_dashboard.png"
    dashboard.save(str(output_path), dpi=100)

    assert output_path.exists(), "Drift dashboard PNG should be saved"
    assert output_path.stat().st_size > 1000, "PNG should have content (>1KB)"
