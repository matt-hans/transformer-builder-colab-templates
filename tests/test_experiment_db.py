"""
Unit tests for ExperimentDB (SQLite-based experiment tracking).

Tests cover:
- Schema creation and initialization
- Run logging and retrieval
- Metric logging (epoch-level and step-level)
- Artifact tracking
- Run comparison and best run queries
- JSON serialization of configs
- Error handling

Run with:
    pytest tests/test_experiment_db.py -v
"""

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from utils.training.experiment_db import ExperimentDB


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    # Use a temporary directory and custom filename to avoid creation
    tmp_dir = tempfile.gettempdir()
    db_path = Path(tmp_dir) / f'test_exp_{id(object())}.db'

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def db(temp_db):
    """Create ExperimentDB instance with temporary database."""
    return ExperimentDB(temp_db)


class TestSchemaCreation:
    """Test database schema creation and initialization."""

    def test_creates_database_file(self, temp_db):
        """Test that database file is created on initialization."""
        assert not temp_db.exists()

        db = ExperimentDB(temp_db)

        assert temp_db.exists()
        assert db.db_path == temp_db

    def test_creates_runs_table(self, db, temp_db):
        """Test that runs table is created with correct schema."""
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
            )
            result = cursor.fetchone()

        assert result is not None
        assert result[0] == 'runs'

    def test_creates_metrics_table(self, db, temp_db):
        """Test that metrics table is created with correct schema."""
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
            )
            result = cursor.fetchone()

        assert result is not None
        assert result[0] == 'metrics'

    def test_creates_artifacts_table(self, db, temp_db):
        """Test that artifacts table is created with correct schema."""
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifacts'"
            )
            result = cursor.fetchone()

        assert result is not None
        assert result[0] == 'artifacts'

    def test_creates_metrics_index(self, db, temp_db):
        """Test that metrics index is created for performance."""
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_metrics_run_name'"
            )
            result = cursor.fetchone()

        assert result is not None

    def test_idempotent_schema_creation(self, temp_db):
        """Test that schema creation can be called multiple times safely."""
        db1 = ExperimentDB(temp_db)
        db2 = ExperimentDB(temp_db)  # Should not error

        assert db1.db_path == db2.db_path


class TestRunLogging:
    """Test run creation and retrieval."""

    def test_log_run_returns_run_id(self, db):
        """Test that log_run returns an integer run_id."""
        config = {'learning_rate': 5e-5, 'batch_size': 4}
        run_id = db.log_run('test-run', config, notes='Test notes')

        assert isinstance(run_id, int)
        assert run_id > 0

    def test_log_run_increments_ids(self, db):
        """Test that run IDs are auto-incremented."""
        config = {'learning_rate': 5e-5}

        run_id1 = db.log_run('run-1', config)
        run_id2 = db.log_run('run-2', config)

        assert run_id2 == run_id1 + 1

    def test_log_run_stores_config_as_json(self, db, temp_db):
        """Test that config is serialized to JSON."""
        config = {
            'learning_rate': 5e-5,
            'batch_size': 4,
            'nested': {'key': 'value'}
        }
        run_id = db.log_run('test-run', config)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT config FROM runs WHERE run_id = ?', (run_id,))
            config_json = cursor.fetchone()[0]

        stored_config = json.loads(config_json)
        assert stored_config == config

    def test_log_run_default_status_running(self, db, temp_db):
        """Test that new runs have status='running'."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT status FROM runs WHERE run_id = ?', (run_id,))
            status = cursor.fetchone()[0]

        assert status == 'running'

    def test_get_run_retrieves_metadata(self, db):
        """Test that get_run returns complete run metadata."""
        config = {
            'learning_rate': 5e-5,
            'batch_size': 4
        }
        run_id = db.log_run('test-run', config, notes='Test notes')

        run = db.get_run(run_id)

        assert run['run_id'] == run_id
        assert run['run_name'] == 'test-run'
        assert run['config'] == config
        assert run['notes'] == 'Test notes'
        assert run['status'] == 'running'
        assert 'created_at' in run

    def test_get_run_nonexistent_raises_error(self, db):
        """Test that get_run raises ValueError for missing run_id."""
        with pytest.raises(ValueError, match="Run 999 not found"):
            db.get_run(999)

    def test_update_run_status(self, db):
        """Test updating run status."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.update_run_status(run_id, 'completed')

        run = db.get_run(run_id)
        assert run['status'] == 'completed'


class TestMetricLogging:
    """Test metric logging and retrieval."""

    def test_log_epoch_metric(self, db):
        """Test logging epoch-level metrics."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_metric(run_id, 'train/loss', 0.42, epoch=0)

        metrics = db.get_metrics(run_id, 'train/loss')
        assert len(metrics) == 1
        assert metrics.iloc[0]['metric_name'] == 'train/loss'
        assert metrics.iloc[0]['value'] == 0.42
        assert metrics.iloc[0]['epoch'] == 0
        assert pd.isna(metrics.iloc[0]['step'])

    def test_log_step_metric(self, db):
        """Test logging step-level metrics."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_metric(run_id, 'train/batch_loss', 0.45, step=100, epoch=0)

        metrics = db.get_metrics(run_id, 'train/batch_loss')
        assert len(metrics) == 1
        assert metrics.iloc[0]['value'] == 0.45
        assert metrics.iloc[0]['step'] == 100
        assert metrics.iloc[0]['epoch'] == 0

    def test_log_multiple_metrics(self, db):
        """Test logging multiple metrics for same run."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        # Log 3 epochs
        expected_train = [0.5, 0.4, 0.3]
        expected_val = [0.6, 0.5, 0.4]

        for epoch in range(3):
            db.log_metric(run_id, 'train/loss', expected_train[epoch], epoch=epoch)
            db.log_metric(run_id, 'val/loss', expected_val[epoch], epoch=epoch)

        train_loss = db.get_metrics(run_id, 'train/loss')
        val_loss = db.get_metrics(run_id, 'val/loss')

        assert len(train_loss) == 3
        assert len(val_loss) == 3
        # Use approximate comparison for floating point values
        assert pytest.approx(list(train_loss['value'])) == expected_train
        assert pytest.approx(list(val_loss['value'])) == expected_val

    def test_get_metrics_all(self, db):
        """Test retrieving all metrics for a run."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_metric(run_id, 'train/loss', 0.42, epoch=0)
        db.log_metric(run_id, 'val/loss', 0.38, epoch=0)
        db.log_metric(run_id, 'train/accuracy', 0.85, epoch=0)

        all_metrics = db.get_metrics(run_id)

        assert len(all_metrics) == 3
        metric_names = set(all_metrics['metric_name'])
        assert metric_names == {'train/loss', 'val/loss', 'train/accuracy'}

    def test_get_metrics_filtered(self, db):
        """Test retrieving metrics filtered by name."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_metric(run_id, 'train/loss', 0.42, epoch=0)
        db.log_metric(run_id, 'val/loss', 0.38, epoch=0)

        train_loss = db.get_metrics(run_id, 'train/loss')

        assert len(train_loss) == 1
        assert train_loss.iloc[0]['metric_name'] == 'train/loss'

    def test_metrics_ordered_by_timestamp(self, db):
        """Test that metrics are returned in chronological order."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        # Log out of order
        db.log_metric(run_id, 'train/loss', 0.5, epoch=2)
        db.log_metric(run_id, 'train/loss', 0.7, epoch=0)
        db.log_metric(run_id, 'train/loss', 0.6, epoch=1)

        metrics = db.get_metrics(run_id, 'train/loss')

        # Should be ordered by timestamp (insertion order)
        assert list(metrics['epoch']) == [2, 0, 1]


class TestArtifactLogging:
    """Test artifact logging and retrieval."""

    def test_log_artifact_basic(self, db, temp_db):
        """Test logging artifact without metadata."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_artifact(run_id, 'checkpoint', 'checkpoints/epoch_5.pt')

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT artifact_type, filepath FROM artifacts WHERE run_id = ?',
                (run_id,)
            )
            artifact = cursor.fetchone()

        assert artifact[0] == 'checkpoint'
        assert artifact[1] == 'checkpoints/epoch_5.pt'

    def test_log_artifact_with_metadata(self, db, temp_db):
        """Test logging artifact with metadata dictionary."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        metadata = {'epoch': 5, 'val_loss': 0.38}
        db.log_artifact(run_id, 'checkpoint', 'checkpoints/epoch_5.pt', metadata=metadata)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT metadata FROM artifacts WHERE run_id = ?',
                (run_id,)
            )
            metadata_json = cursor.fetchone()[0]

        stored_metadata = json.loads(metadata_json)
        assert stored_metadata == metadata

    def test_log_multiple_artifacts(self, db, temp_db):
        """Test logging multiple artifacts for same run."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_artifact(run_id, 'checkpoint', 'checkpoints/epoch_5.pt')
        db.log_artifact(run_id, 'plot', 'training_curves.png')
        db.log_artifact(run_id, 'config', 'config.json')

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT COUNT(*) FROM artifacts WHERE run_id = ?',
                (run_id,)
            )
            count = cursor.fetchone()[0]

        assert count == 3

    def test_log_artifact_with_path_object(self, db, temp_db):
        """Test logging artifact with Path object."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        filepath = Path('checkpoints/epoch_5.pt')
        db.log_artifact(run_id, 'checkpoint', filepath)

        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT filepath FROM artifacts WHERE run_id = ?',
                (run_id,)
            )
            stored_path = cursor.fetchone()[0]

        assert stored_path == str(filepath)


class TestRunComparison:
    """Test run comparison utilities."""

    def test_compare_runs_basic(self, db):
        """Test comparing multiple runs."""
        # Create 2 runs
        config1 = {'learning_rate': 5e-5, 'batch_size': 4}
        config2 = {'learning_rate': 1e-4, 'batch_size': 8}

        run_id1 = db.log_run('run-1', config1)
        run_id2 = db.log_run('run-2', config2)

        # Log metrics
        db.log_metric(run_id1, 'train/loss', 0.5, epoch=0)
        db.log_metric(run_id1, 'val/loss', 0.45, epoch=0)

        db.log_metric(run_id2, 'train/loss', 0.4, epoch=0)
        db.log_metric(run_id2, 'val/loss', 0.38, epoch=0)

        comparison = db.compare_runs([run_id1, run_id2])

        assert len(comparison) == 2
        assert list(comparison['run_name']) == ['run-1', 'run-2']
        assert list(comparison['final_train_loss']) == [0.5, 0.4]
        assert list(comparison['final_val_loss']) == [0.45, 0.38]

    def test_compare_runs_best_metrics(self, db):
        """Test that comparison includes best metrics across epochs."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        # Log 3 epochs with decreasing then increasing loss
        db.log_metric(run_id, 'val/loss', 0.5, epoch=0)
        db.log_metric(run_id, 'val/loss', 0.3, epoch=1)  # Best
        db.log_metric(run_id, 'val/loss', 0.4, epoch=2)

        comparison = db.compare_runs([run_id])

        assert comparison.iloc[0]['final_val_loss'] == 0.4
        assert comparison.iloc[0]['best_val_loss'] == 0.3
        assert comparison.iloc[0]['best_epoch'] == 1

    def test_compare_runs_total_epochs(self, db):
        """Test that comparison counts total epochs correctly."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        for epoch in range(5):
            db.log_metric(run_id, 'train/loss', 0.5 - epoch * 0.05, epoch=epoch)

        comparison = db.compare_runs([run_id])

        assert comparison.iloc[0]['total_epochs'] == 5

    def test_compare_runs_skips_missing(self, db):
        """Test that comparison skips missing run IDs gracefully."""
        config = {'learning_rate': 5e-5}
        run_id1 = db.log_run('run-1', config)

        db.log_metric(run_id1, 'train/loss', 0.5, epoch=0)

        # Compare with missing run_id
        comparison = db.compare_runs([run_id1, 999])

        assert len(comparison) == 1  # Only valid run
        assert comparison.iloc[0]['run_id'] == run_id1

    def test_compare_runs_missing_metrics(self, db):
        """Test comparison with runs that have no metrics."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)
        # No metrics logged

        comparison = db.compare_runs([run_id])

        assert len(comparison) == 1
        assert pd.isna(comparison.iloc[0]['final_train_loss'])
        assert pd.isna(comparison.iloc[0]['final_val_loss'])
        assert comparison.iloc[0]['total_epochs'] == 0


class TestListRuns:
    """Test run listing functionality."""

    def test_list_runs_returns_dataframe(self, db):
        """Test that list_runs returns DataFrame."""
        config = {'learning_rate': 5e-5}
        db.log_run('test-run', config)

        runs = db.list_runs()

        assert isinstance(runs, pd.DataFrame)
        assert len(runs) == 1

    def test_list_runs_ordered_by_created_at(self, db):
        """Test that runs are listed newest first (by run_id when timestamps are same)."""
        config = {'learning_rate': 5e-5}

        db.log_run('run-1', config)
        db.log_run('run-2', config)
        db.log_run('run-3', config)

        runs = db.list_runs()

        # Newest first (by run_id DESC since timestamps are identical)
        assert list(runs['run_name']) == ['run-3', 'run-2', 'run-1']

    def test_list_runs_respects_limit(self, db):
        """Test that limit parameter works correctly."""
        config = {'learning_rate': 5e-5}

        for i in range(5):
            db.log_run(f'run-{i}', config)

        runs = db.list_runs(limit=3)

        assert len(runs) == 3

    def test_list_runs_includes_metadata(self, db):
        """Test that list_runs includes all metadata columns."""
        config = {'learning_rate': 5e-5}
        db.log_run('test-run', config, notes='Test notes')

        runs = db.list_runs()

        expected_columns = {'run_id', 'run_name', 'created_at', 'status', 'notes'}
        assert set(runs.columns) == expected_columns
        assert runs.iloc[0]['notes'] == 'Test notes'


class TestBestRun:
    """Test best run queries."""

    def test_get_best_run_min_mode(self, db):
        """Test finding best run with minimum metric."""
        config1 = {'learning_rate': 5e-5}
        config2 = {'learning_rate': 1e-4}

        run_id1 = db.log_run('run-1', config1)
        run_id2 = db.log_run('run-2', config2)

        # Run 2 has better (lower) loss
        db.log_metric(run_id1, 'val/loss', 0.5, epoch=0)
        db.log_metric(run_id2, 'val/loss', 0.3, epoch=0)

        best = db.get_best_run('val/loss', mode='min')

        assert best['run_id'] == run_id2
        assert best['run_name'] == 'run-2'
        assert best['best_value'] == 0.3
        assert best['best_epoch'] == 0

    def test_get_best_run_max_mode(self, db):
        """Test finding best run with maximum metric."""
        config1 = {'learning_rate': 5e-5}
        config2 = {'learning_rate': 1e-4}

        run_id1 = db.log_run('run-1', config1)
        run_id2 = db.log_run('run-2', config2)

        # Run 2 has better (higher) accuracy
        db.log_metric(run_id1, 'val/accuracy', 0.85, epoch=0)
        db.log_metric(run_id2, 'val/accuracy', 0.92, epoch=0)

        best = db.get_best_run('val/accuracy', mode='max')

        assert best['run_id'] == run_id2
        assert best['best_value'] == 0.92

    def test_get_best_run_across_epochs(self, db):
        """Test finding best value across multiple epochs."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)

        db.log_metric(run_id, 'val/loss', 0.5, epoch=0)
        db.log_metric(run_id, 'val/loss', 0.3, epoch=1)  # Best
        db.log_metric(run_id, 'val/loss', 0.4, epoch=2)

        best = db.get_best_run('val/loss', mode='min')

        assert best['best_value'] == 0.3
        assert best['best_epoch'] == 1

    def test_get_best_run_invalid_mode_raises_error(self, db):
        """Test that invalid mode raises ValueError."""
        config = {'learning_rate': 5e-5}
        run_id = db.log_run('test-run', config)
        db.log_metric(run_id, 'val/loss', 0.5, epoch=0)

        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            db.get_best_run('val/loss', mode='invalid')

    def test_get_best_run_no_metric_raises_error(self, db):
        """Test that missing metric raises ValueError."""
        config = {'learning_rate': 5e-5}
        db.log_run('test-run', config)
        # No metrics logged

        with pytest.raises(ValueError, match="No runs found with metric"):
            db.get_best_run('val/loss', mode='min')

    def test_get_best_run_includes_config(self, db):
        """Test that best run includes configuration."""
        config = {
            'learning_rate': 5e-5,
            'batch_size': 4,
            'epochs': 10
        }
        run_id = db.log_run('test-run', config)
        db.log_metric(run_id, 'val/loss', 0.5, epoch=0)

        best = db.get_best_run('val/loss', mode='min')

        assert best['config'] == config
        assert best['config']['learning_rate'] == 5e-5


class TestJSONSerialization:
    """Test JSON serialization of complex configs."""

    def test_nested_dict_serialization(self, db):
        """Test serializing nested dictionaries."""
        config = {
            'model': {
                'n_layers': 12,
                'n_heads': 8
            },
            'optimizer': {
                'name': 'AdamW',
                'betas': [0.9, 0.999]
            }
        }

        run_id = db.log_run('test-run', config)
        run = db.get_run(run_id)

        assert run['config'] == config
        assert run['config']['model']['n_layers'] == 12
        assert run['config']['optimizer']['betas'] == [0.9, 0.999]

    def test_float_precision_preserved(self, db):
        """Test that float precision is preserved in serialization."""
        config = {
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'epsilon': 1e-8
        }

        run_id = db.log_run('test-run', config)
        run = db.get_run(run_id)

        assert run['config']['learning_rate'] == 5e-5
        assert run['config']['weight_decay'] == 1e-4
        assert run['config']['epsilon'] == 1e-8

    def test_list_serialization(self, db):
        """Test serializing lists in config."""
        config = {
            'hidden_sizes': [256, 512, 1024],
            'dropout_rates': [0.1, 0.2, 0.3]
        }

        run_id = db.log_run('test-run', config)
        run = db.get_run(run_id)

        assert run['config']['hidden_sizes'] == [256, 512, 1024]
        assert run['config']['dropout_rates'] == [0.1, 0.2, 0.3]

    def test_none_values_serialization(self, db):
        """Test serializing None values in config."""
        config = {
            'learning_rate': 5e-5,
            'scheduler': None,
            'warmup_steps': None
        }

        run_id = db.log_run('test-run', config)
        run = db.get_run(run_id)

        assert run['config']['scheduler'] is None
        assert run['config']['warmup_steps'] is None
