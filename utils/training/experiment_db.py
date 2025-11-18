"""
SQLite-based experiment tracking for local development.

This module provides a lightweight alternative to Weights & Biases (W&B) for
tracking machine learning experiments locally. It stores run configurations,
metrics (epoch-level and step-level), and artifacts in a SQLite database.

Example Usage:
    >>> from utils.training.experiment_db import ExperimentDB
    >>> from utils.training.training_config import TrainingConfig
    >>>
    >>> # Initialize database
    >>> db = ExperimentDB('experiments.db')
    >>>
    >>> # Log new run
    >>> config = TrainingConfig(learning_rate=5e-5, batch_size=4)
    >>> run_id = db.log_run('baseline-exp', config.to_dict(), notes='Initial baseline')
    >>>
    >>> # Log metrics during training
    >>> db.log_metric(run_id, 'train/loss', 0.42, epoch=0)
    >>> db.log_metric(run_id, 'val/loss', 0.38, epoch=0)
    >>>
    >>> # Log artifacts
    >>> db.log_artifact(run_id, 'checkpoint', 'checkpoints/epoch_5.pt')
    >>>
    >>> # Compare runs
    >>> comparison = db.compare_runs([1, 2, 3])
    >>>
    >>> # Find best run
    >>> best_run = db.get_best_run('val/loss', mode='min')

Author: MLOps Agent 6
Version: 3.4.0
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentDB:
    """SQLite-based experiment tracking for local development.

    This class provides persistent storage for ML experiment runs, including:
    - Run metadata and configuration
    - Epoch-level and step-level metrics
    - Artifact paths (checkpoints, plots, configs)
    - Comparison and query utilities

    Attributes:
        db_path: Path to SQLite database file.

    Schema:
        runs: Run metadata (run_id, run_name, config, notes, timestamps)
        metrics: Metric values (run_id, metric_name, value, step, epoch)
        artifacts: Artifact paths (run_id, artifact_type, filepath, metadata)
    """

    def __init__(self, db_path: Union[str, Path] = 'experiments.db'):
        """Initialize database with schema creation.

        Args:
            db_path: Path to SQLite database file. Created if doesn't exist.

        Example:
            >>> db = ExperimentDB('my_experiments.db')
            >>> db = ExperimentDB()  # Uses default 'experiments.db'
        """
        self.db_path = Path(db_path)
        self._create_schema()
        logger.info(f"Initialized ExperimentDB at {self.db_path}")

    def _create_schema(self) -> None:
        """Create database schema if tables don't exist.

        Creates three tables:
        1. runs: Experiment run metadata
        2. metrics: Time-series metric values
        3. artifacts: File paths and metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT NOT NULL,
                    config TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    sweep_id TEXT,
                    sweep_params TEXT
                )
            ''')

            # Metrics table (supports both epoch-level and step-level)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER,
                    epoch INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
            ''')

            # Index for faster metric queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_run_name
                ON metrics (run_id, metric_name)
            ''')

            # Artifacts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    artifact_type TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
                )
            ''')

            conn.commit()
            logger.debug("Database schema created/validated")

            # Ensure columns exist on older DBs (idempotent)
            try:
                cursor.execute("PRAGMA table_info(runs)")
                cols = {row[1] for row in cursor.fetchall()}
                if 'sweep_id' not in cols:
                    cursor.execute("ALTER TABLE runs ADD COLUMN sweep_id TEXT")
                if 'sweep_params' not in cols:
                    cursor.execute("ALTER TABLE runs ADD COLUMN sweep_params TEXT")
                if 'gist_id' not in cols:
                    cursor.execute("ALTER TABLE runs ADD COLUMN gist_id TEXT")
                if 'gist_revision' not in cols:
                    cursor.execute("ALTER TABLE runs ADD COLUMN gist_revision TEXT")
                if 'gist_sha256' not in cols:
                    cursor.execute("ALTER TABLE runs ADD COLUMN gist_sha256 TEXT")
                conn.commit()
            except Exception:
                pass

    def log_run(
        self,
        run_name: str,
        config: Dict[str, Any],
        notes: str = '',
        *,
        sweep_id: str | None = None,
        sweep_params: Dict[str, Any] | None = None,
        gist_id: str | None = None,
        gist_revision: str | None = None,
        gist_sha256: str | None = None,
    ) -> int:
        """Create new experiment run and return run_id.

        Args:
            run_name: Human-readable name for the run (e.g., 'baseline-exp-1').
            config: Configuration dictionary (e.g., from TrainingConfig.to_dict()).
            notes: Optional notes/description for this experiment.

        Returns:
            run_id: Integer ID for the newly created run.

        Example:
            >>> config = {'learning_rate': 5e-5, 'batch_size': 4}
            >>> run_id = db.log_run('baseline-v1', config, notes='First baseline')
            >>> print(f"Created run {run_id}")
        """
        config_json = json.dumps(config, indent=2)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO runs (run_name, config, notes, status, sweep_id, sweep_params, gist_id, gist_revision, gist_sha256)
                VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)
                ''',
                (
                    run_name,
                    config_json,
                    notes,
                    sweep_id,
                    json.dumps(sweep_params) if sweep_params else None,
                    gist_id,
                    gist_revision,
                    gist_sha256,
                )
            )
            run_id = cursor.lastrowid
            conn.commit()

        logger.info(f"Created run {run_id}: '{run_name}'")
        return run_id

    def log_metric(
        self,
        run_id: int,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ) -> None:
        """Log a metric value (epoch-level or step-level).

        Args:
            run_id: Run ID from log_run().
            metric_name: Metric name (e.g., 'train/loss', 'val/accuracy').
            value: Metric value (float).
            step: Optional global step number (for per-batch logging).
            epoch: Optional epoch number (for per-epoch logging).

        Example:
            >>> # Epoch-level metric
            >>> db.log_metric(run_id, 'train/loss', 0.42, epoch=0)
            >>>
            >>> # Step-level metric
            >>> db.log_metric(run_id, 'train/batch_loss', 0.45, step=100, epoch=0)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO metrics (run_id, metric_name, value, step, epoch)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (run_id, metric_name, value, step, epoch)
            )
            conn.commit()

    def log_artifact(
        self,
        run_id: int,
        artifact_type: str,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log artifact (checkpoint, plot, config file).

        Args:
            run_id: Run ID from log_run().
            artifact_type: Type of artifact ('checkpoint', 'plot', 'config', 'model').
            filepath: Path to artifact file (relative or absolute).
            metadata: Optional metadata dictionary (e.g., {'epoch': 5, 'loss': 0.42}).

        Example:
            >>> db.log_artifact(run_id, 'checkpoint', 'checkpoints/epoch_5.pt',
            ...                 metadata={'epoch': 5, 'val_loss': 0.38})
            >>> db.log_artifact(run_id, 'plot', 'training_curves.png')
        """
        filepath_str = str(filepath)
        metadata_json = json.dumps(metadata) if metadata else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO artifacts (run_id, artifact_type, filepath, metadata)
                VALUES (?, ?, ?, ?)
                ''',
                (run_id, artifact_type, filepath_str, metadata_json)
            )
            conn.commit()

        logger.debug(f"Logged artifact: {artifact_type} -> {filepath_str}")

    def update_run_status(self, run_id: int, status: str) -> None:
        """Update run status.

        Args:
            run_id: Run ID to update.
            status: New status ('running', 'completed', 'failed').

        Example:
            >>> db.update_run_status(run_id, 'completed')
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE runs SET status = ? WHERE run_id = ?',
                (status, run_id)
            )
            conn.commit()

        logger.info(f"Updated run {run_id} status: {status}")

    def get_run(self, run_id: int) -> Dict[str, Any]:
        """Retrieve run metadata and config.

        Args:
            run_id: Run ID to retrieve.

        Returns:
            Dictionary with run metadata:
                - run_id: int
                - run_name: str
                - config: dict (deserialized from JSON)
                - notes: str
                - created_at: str (ISO timestamp)
                - status: str

        Raises:
            ValueError: If run_id doesn't exist.

        Example:
            >>> run = db.get_run(1)
            >>> print(run['run_name'])
            >>> print(run['config']['learning_rate'])
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM runs WHERE run_id = ?',
                (run_id,)
            )
            row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Run {run_id} not found in database")

        run_data = dict(row)
        run_data['config'] = json.loads(run_data['config']) if run_data['config'] else {}

        return run_data

    def get_metrics(
        self,
        run_id: int,
        metric_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Get metrics for a run, optionally filtered by name.

        Args:
            run_id: Run ID to retrieve metrics for.
            metric_name: Optional metric name filter (e.g., 'train/loss').
                        If None, returns all metrics.

        Returns:
            DataFrame with columns: [metric_id, run_id, metric_name, value,
                                    step, epoch, timestamp]

        Example:
            >>> # Get all metrics
            >>> all_metrics = db.get_metrics(run_id)
            >>>
            >>> # Get specific metric
            >>> train_loss = db.get_metrics(run_id, 'train/loss')
            >>> print(train_loss[['epoch', 'value']])
        """
        query = 'SELECT * FROM metrics WHERE run_id = ?'
        params: List[Union[int, str]] = [run_id]

        if metric_name is not None:
            query += ' AND metric_name = ?'
            params.append(metric_name)

        query += ' ORDER BY timestamp ASC'

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        return df

    def compare_runs(self, run_ids: List[int]) -> pd.DataFrame:
        """Compare metrics across multiple runs.

        Args:
            run_ids: List of run IDs to compare.

        Returns:
            DataFrame with summary statistics for each run:
                - run_id: int
                - run_name: str
                - created_at: str
                - status: str
                - final_train_loss: float (last epoch)
                - final_val_loss: float (last epoch)
                - best_val_loss: float (minimum)
                - best_epoch: int (epoch with best val_loss)
                - total_epochs: int

        Example:
            >>> comparison = db.compare_runs([1, 2, 3])
            >>> print(comparison[['run_name', 'final_val_loss', 'best_epoch']])
            >>> print(comparison.sort_values('best_val_loss'))
        """
        run_summaries = []

        for run_id in run_ids:
            try:
                run = self.get_run(run_id)
                metrics = self.get_metrics(run_id)

                summary = {
                    'run_id': run_id,
                    'run_name': run['run_name'],
                    'created_at': run['created_at'],
                    'status': run['status']
                }

                # Extract final and best metrics
                train_loss = metrics[metrics['metric_name'] == 'train/loss']
                val_loss = metrics[metrics['metric_name'] == 'val/loss']

                if not train_loss.empty:
                    summary['final_train_loss'] = train_loss.iloc[-1]['value']
                else:
                    summary['final_train_loss'] = None

                if not val_loss.empty:
                    summary['final_val_loss'] = val_loss.iloc[-1]['value']
                    summary['best_val_loss'] = val_loss['value'].min()
                    best_idx = val_loss['value'].idxmin()
                    summary['best_epoch'] = val_loss.loc[best_idx, 'epoch']
                else:
                    summary['final_val_loss'] = None
                    summary['best_val_loss'] = None
                    summary['best_epoch'] = None

                # Count epochs
                epoch_metrics = metrics[metrics['epoch'].notna()]
                summary['total_epochs'] = int(epoch_metrics['epoch'].max()) + 1 if not epoch_metrics.empty else 0

                run_summaries.append(summary)

            except ValueError as e:
                logger.warning(f"Skipping run {run_id}: {e}")

        return pd.DataFrame(run_summaries)

    def list_runs(self, limit: int = 10) -> pd.DataFrame:
        """List recent runs with summary statistics.

        Args:
            limit: Maximum number of runs to return (most recent first).

        Returns:
            DataFrame with columns: [run_id, run_name, created_at, status, notes]

        Example:
            >>> recent_runs = db.list_runs(limit=5)
            >>> print(recent_runs[['run_id', 'run_name', 'status']])
        """
        query = '''
            SELECT run_id, run_name, created_at, status, notes
            FROM runs
            ORDER BY created_at DESC, run_id DESC
            LIMIT ?
        '''

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))

        return df

    def get_best_run(
        self,
        metric_name: str,
        mode: str = 'min'
    ) -> Dict[str, Any]:
        """Find best run by metric (min loss, max accuracy).

        Args:
            metric_name: Metric to optimize (e.g., 'val/loss', 'val/accuracy').
            mode: Optimization mode ('min' or 'max').

        Returns:
            Dictionary with best run information:
                - run_id: int
                - run_name: str
                - best_value: float (best metric value)
                - best_epoch: int (epoch where best value occurred)
                - config: dict (run configuration)

        Raises:
            ValueError: If mode not in ['min', 'max'] or no runs found.

        Example:
            >>> # Find run with lowest validation loss
            >>> best_run = db.get_best_run('val/loss', mode='min')
            >>> print(f"Best run: {best_run['run_name']}")
            >>> print(f"Val loss: {best_run['best_value']:.4f}")
            >>>
            >>> # Find run with highest accuracy
            >>> best_run = db.get_best_run('val/accuracy', mode='max')
        """
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        # Get all runs that have this metric
        query = '''
            SELECT DISTINCT run_id
            FROM metrics
            WHERE metric_name = ?
        '''

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (metric_name,))
            run_ids = [row[0] for row in cursor.fetchall()]

        if not run_ids:
            raise ValueError(f"No runs found with metric '{metric_name}'")

        # Find best value across all runs
        best_run_id = None
        best_value = float('inf') if mode == 'min' else float('-inf')
        best_epoch = None

        for run_id in run_ids:
            metrics = self.get_metrics(run_id, metric_name)

            if metrics.empty:
                continue

            if mode == 'min':
                run_best_value = metrics['value'].min()
                if run_best_value < best_value:
                    best_value = run_best_value
                    best_run_id = run_id
                    best_idx = metrics['value'].idxmin()
                    best_epoch = metrics.loc[best_idx, 'epoch']
            else:  # mode == 'max'
                run_best_value = metrics['value'].max()
                if run_best_value > best_value:
                    best_value = run_best_value
                    best_run_id = run_id
                    best_idx = metrics['value'].idxmax()
                    best_epoch = metrics.loc[best_idx, 'epoch']

        if best_run_id is None:
            raise ValueError(f"Could not find best run for metric '{metric_name}'")

        run = self.get_run(best_run_id)

        return {
            'run_id': best_run_id,
            'run_name': run['run_name'],
            'best_value': best_value,
            'best_epoch': int(best_epoch) if pd.notna(best_epoch) else None,
            'config': run['config']
        }

    def get_runs_for_sweep(self, sweep_id: str) -> pd.DataFrame:
        """Return runs logged under a given sweep_id."""
        query = '''
            SELECT run_id, run_name, created_at, status, notes, sweep_params
            FROM runs
            WHERE sweep_id = ?
            ORDER BY created_at ASC
        '''

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(sweep_id,))
        return df
