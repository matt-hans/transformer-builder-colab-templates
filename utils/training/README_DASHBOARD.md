# TrainingDashboard - Comprehensive Training Visualization

Professional-grade 6-panel matplotlib dashboard for post-training analysis with MetricsTracker integration.

## Features

### 6-Panel Visualization Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Dashboard                       │
│  Config: lr=5e-5, batch=4, epochs=10 | Best: Epoch 7       │
├─────────────────────┬───────────────────┬───────────────────┤
│  1. Loss Curves     │  2. Perplexity    │  3. Accuracy      │
│  (train vs val)     │  (val only)       │  (train vs val)   │
│  - Smoothed lines   │  - Lower is better│  - Higher better  │
│  - Best epoch mark  │  - Best mark      │  - Best mark      │
├─────────────────────┼───────────────────┼───────────────────┤
│  4. Learning Rate   │  5. Gradient Norm │  6. Training Time │
│  (LR schedule)      │  (stability)      │  (epoch duration) │
│  - Warmup visible   │  - Clip threshold │  - Time per epoch │
│  - Decay curve      │  - Warning zones  │  - ETA estimate   │
└─────────────────────┴───────────────────┴───────────────────┘
```

### Panel Details

**Panel 1: Loss Curves**
- Train loss (blue) vs Validation loss (orange)
- Annotates best validation loss epoch
- Auto log-scale for >10x variation
- Grid lines for readability

**Panel 2: Perplexity**
- Validation perplexity (auto-computed from val/loss if missing)
- Lower-is-better indicator
- Reference line at perplexity=10

**Panel 3: Accuracy** (optional)
- Train accuracy vs Validation accuracy
- Percentage format (0-100%)
- Best validation accuracy marked
- Shows "N/A" if accuracy not tracked

**Panel 4: Learning Rate Schedule**
- LR over epochs
- Highlights warmup phase (first 10%, yellow shading)
- Shows decay pattern (linear/cosine)
- Auto log-scale if LR varies >10x

**Panel 5: Gradient Norms**
- Pre-clip gradient norm (blue)
- Post-clip gradient norm (orange)
- Clip threshold line (red dashed)
- Warning zone (red shading) for norms >5.0

**Panel 6: Training Time**
- Epoch duration (seconds) as bar chart
- Average time per epoch (red line)
- Helps identify performance bottlenecks

## Quick Start

### Basic Usage

```python
from utils.training.metrics_tracker import MetricsTracker
from utils.training.dashboard import TrainingDashboard

# After training
tracker = MetricsTracker(use_wandb=False)
# ... training loop with tracker.log_epoch() ...

# Create dashboard
metrics_df = tracker.get_summary()
dashboard = TrainingDashboard(figsize=(18, 12))
fig = dashboard.plot(metrics_df, title='My Training Dashboard')

# Save
dashboard.save('training_dashboard.png', dpi=150)
```

### With TrainingConfig

```python
from types import SimpleNamespace

config = SimpleNamespace(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10
)

fig = dashboard.plot(metrics_df, config=config, title='GPT-2 Fine-Tuning')
```

### Export Formats

```python
# PNG (default, high-resolution)
dashboard.save('dashboard.png', dpi=150)

# PDF (vector, publication-ready)
dashboard.save('dashboard.pdf', dpi=150)

# SVG (vector, web/editor)
dashboard.save('dashboard.svg')
```

## Expected DataFrame Schema

The dashboard expects a pandas DataFrame from `MetricsTracker.get_summary()`:

### Required Columns
- `epoch` (int): Epoch number
- `train/loss` (float): Training loss
- `val/loss` (float): Validation loss

### Optional Columns
- `val/perplexity` (float): Auto-computed from val/loss if missing
- `train/accuracy` (float): Training accuracy [0-1]
- `val/accuracy` (float): Validation accuracy [0-1]
- `learning_rate` (float): Current learning rate
- `gradients/pre_clip_norm` (float): Gradient norm before clipping
- `gradients/post_clip_norm` (float): Gradient norm after clipping
- `epoch_duration` (float): Time per epoch in seconds

### Missing Metrics

The dashboard gracefully handles missing optional metrics:
- **No accuracy**: Shows "N/A" message in accuracy panel
- **No learning rate**: Shows "N/A" in LR panel
- **No gradients**: Shows "N/A" in gradient panel
- **No timing**: Shows "N/A" in time panel
- **No perplexity**: Auto-computed from val/loss

## Examples

### Example 1: Full Metrics (All 6 Panels)

```python
import pandas as pd
from utils.training.dashboard import TrainingDashboard

# Full metrics DataFrame
full_metrics = pd.DataFrame({
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

dashboard = TrainingDashboard()
fig = dashboard.plot(full_metrics, title='Full Dashboard')
dashboard.save('full_dashboard.png', dpi=150)
```

### Example 2: Minimal Metrics (Loss Only)

```python
# Minimal metrics (only loss)
minimal_metrics = pd.DataFrame({
    'epoch': [1, 2, 3],
    'train/loss': [2.5, 2.0, 1.8],
    'val/loss': [2.6, 2.1, 1.9]
})

dashboard = TrainingDashboard()
fig = dashboard.plot(minimal_metrics, title='Minimal Dashboard')
# Perplexity auto-computed, other panels show N/A
```

### Example 3: Integration with Training Loop

```python
from utils.tier3_training_utilities import test_fine_tuning
from utils.training.dashboard import TrainingDashboard

# Run training
results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    use_wandb=True
)

# Create dashboard from results
metrics_df = results['metrics_summary']
dashboard = TrainingDashboard(figsize=(18, 12))
fig = dashboard.plot(metrics_df, config=config, title='Fine-Tuning Results')

# Export in multiple formats
dashboard.save('training_results.png', dpi=150)
dashboard.save('training_results.pdf', dpi=150)
dashboard.save('training_results.svg')
```

## Advanced Features

### Custom Figure Size

```python
# Larger dashboard for presentations
dashboard = TrainingDashboard(figsize=(24, 16))

# Compact dashboard for reports
dashboard = TrainingDashboard(figsize=(12, 8))
```

### Best Epoch Identification

The dashboard automatically identifies and annotates the best epoch (minimum validation loss):

- **Summary card**: Shows best epoch number and metrics
- **Loss panel**: Red star marker on best validation loss
- **Perplexity panel**: Marker on best perplexity
- **Accuracy panel**: Marker on best validation accuracy

### Automatic Scaling

- **Log scale**: Applied automatically if loss/LR varies >10x
- **Warmup detection**: First 10% of epochs highlighted if LR increases
- **Warning zones**: Gradient norms >5.0 highlighted in red

## Testing

Run comprehensive test suite:

```bash
pytest tests/test_dashboard.py -v
```

**Test Coverage**:
- ✅ 20 tests covering all features
- ✅ Full metrics (6 panels)
- ✅ Minimal metrics (loss only)
- ✅ Missing optional metrics
- ✅ Export formats (PNG, PDF, SVG)
- ✅ Error handling (empty DataFrame, invalid inputs)
- ✅ Edge cases (NaN values, single epoch, log scaling)

## Demo Script

Run the demo script to see all features:

```bash
python examples/dashboard_demo.py
```

**Generates**:
- `examples/outputs/full_dashboard.png` - Full metrics (20 epochs)
- `examples/outputs/minimal_dashboard.png` - Minimal metrics
- `examples/outputs/full_dashboard.pdf` - PDF export
- `examples/outputs/full_dashboard.svg` - SVG export

## Drift Metrics Quickstart (Tier 5 Preview)

You can compute simple input/output drift profiles and store them in `ExperimentDB` for later monitoring:

```python
from utils.training.drift_metrics import compute_dataset_profile, compare_profiles, log_profile_to_db
from utils.training.experiment_db import ExperimentDB

# Assume you have a dataset and TaskSpec
ref_profile = compute_dataset_profile(train_dataset, task_spec, sample_size=1000)
new_profile = compute_dataset_profile(production_sample, task_spec, sample_size=1000)

result = compare_profiles(ref_profile, new_profile)
print(result["status"], result["drift_scores"])

# Optional: log profile for the current run
db = ExperimentDB("experiments.db")
run_id = db.log_run("run-with-drift-profile", config_dict)
log_profile_to_db(db, run_id, new_profile, profile_name="eval_dataset")
```


## Error Handling

### Empty DataFrame
```python
dashboard.plot(pd.DataFrame())
# Raises: ValueError("DataFrame is empty")
```

### Missing Required Columns
```python
df = pd.DataFrame({'epoch': [1, 2], 'train/loss': [2.5, 2.0]})
dashboard.plot(df)
# Raises: ValueError("Missing required columns: ['val/loss']")
```

### Invalid Figure Size
```python
TrainingDashboard(figsize=(18, -12))
# Raises: ValueError("figsize dimensions must be positive")
```

### Save Before Plot
```python
dashboard = TrainingDashboard()
dashboard.save('output.png')
# Raises: RuntimeError("Must call plot() before save()")
```

### Unsupported Format
```python
dashboard.save('output.jpg')
# Raises: ValueError("Unsupported format .jpg. Use one of: {'.png', '.pdf', '.svg'}")
```

## Implementation Details

**File**: `utils/training/dashboard.py` (409 lines)

**Dependencies**:
- `matplotlib` - Plotting backend
- `pandas` - DataFrame handling
- `numpy` - Numerical operations
- `logging` - Diagnostics

**Key Methods**:
- `__init__(figsize)` - Initialize with custom size
- `plot(metrics_df, config, title)` - Create 6-panel visualization
- `save(filepath, dpi)` - Export to PNG/PDF/SVG
- `_validate_dataframe(df)` - Schema validation
- `_plot_loss_curves(df, ax)` - Panel 1: Loss
- `_plot_perplexity(df, ax)` - Panel 2: Perplexity
- `_plot_accuracy(df, ax)` - Panel 3: Accuracy
- `_plot_learning_rate(df, ax)` - Panel 4: LR schedule
- `_plot_gradient_norms(df, ax)` - Panel 5: Gradients
- `_plot_training_time(df, ax)` - Panel 6: Time
- `_add_summary_card(df, config, ax)` - Top summary

## API Reference

### TrainingDashboard

```python
class TrainingDashboard:
    """Comprehensive 6-panel training visualization dashboard."""

    def __init__(self, figsize: Tuple[int, int] = (18, 12)):
        """
        Args:
            figsize: Figure dimensions (width, height) in inches.
                    Default (18, 12) for good 6-panel layout.
        """

    def plot(
        self,
        metrics_df: pd.DataFrame,
        config: Optional[Any] = None,
        title: str = 'Training Dashboard'
    ) -> Figure:
        """
        Create 6-panel dashboard from metrics DataFrame.

        Args:
            metrics_df: DataFrame from MetricsTracker.get_summary()
            config: Optional TrainingConfig for hyperparameters
            title: Dashboard title

        Returns:
            matplotlib Figure object

        Raises:
            ValueError: If DataFrame empty or missing required columns
        """

    def save(self, filepath: str, dpi: int = 150) -> None:
        """
        Save dashboard to file.

        Args:
            filepath: Output path (.png, .pdf, .svg)
            dpi: Resolution for raster formats

        Raises:
            RuntimeError: If plot() not called yet
            ValueError: If unsupported file format
        """
```

## Best Practices

1. **Always save high-resolution**: Use `dpi=150` for publication-ready figures
2. **Export multiple formats**: PNG for quick preview, PDF for papers, SVG for editing
3. **Include config**: Pass `TrainingConfig` to show hyperparameters in summary
4. **Check best epoch**: Use dashboard to identify best model checkpoint
5. **Monitor gradients**: Check gradient panel for training instability
6. **Analyze timing**: Use time panel to identify performance bottlenecks

## Changelog

### v3.4.0 (Current)
- ✨ Initial release with 6-panel layout
- ✨ MetricsTracker integration
- ✨ Auto-computed perplexity from val/loss
- ✨ Graceful handling of missing metrics
- ✨ PNG/PDF/SVG export support
- ✨ Best epoch annotation across all panels
- ✨ Auto log-scale for wide value ranges
- ✨ Learning rate warmup detection
- ✨ Gradient explosion warning zones
- ✨ Comprehensive test suite (20 tests)

## License

Part of transformer-builder-colab-templates.
See repository LICENSE for details.
