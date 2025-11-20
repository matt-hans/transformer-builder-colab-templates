"""
Model Registry for production model versioning and metadata tracking.

This module provides a SQLite-based registry for managing trained model versions,
their metadata, performance metrics, and deployment tags. It enables:
- Semantic versioning for models (major.minor.patch)
- Model lineage tracking (parent model, training run)
- Tag-based organization (production, staging, experimental)
- Performance metrics storage and comparison
- Export format tracking (ONNX, TorchScript, PyTorch)
- Model size and memory requirements

The registry integrates with CheckpointManager for automatic registration and
with ExperimentDB for linking models to training runs.

Example Usage:
    >>> from utils.training.model_registry import ModelRegistry
    >>>
    >>> # Initialize registry
    >>> registry = ModelRegistry('models.db')
    >>>
    >>> # Register a model
    >>> model_id = registry.register_model(
    ...     name="transformer-v1",
    ...     version="1.0.0",
    ...     checkpoint_path="checkpoints/epoch_10.pt",
    ...     task_type="language_modeling",
    ...     metrics={"val_loss": 0.38, "perplexity": 1.46},
    ...     config_hash="abc123...",
    ...     training_run_id=42
    ... )
    >>>
    >>> # Promote to production
    >>> registry.promote_model(model_id, "production")
    >>>
    >>> # Retrieve production model
    >>> model = registry.get_model(tag="production")
    >>>
    >>> # Compare models
    >>> comparison = registry.compare_models([1, 2, 3])
    >>> print(comparison[['version', 'val_loss', 'model_size_mb']])

Architecture:
    The registry uses SQLite for local storage with a schema supporting:
    - models: Core model metadata and versioning
    - model_exports: Export format availability (ONNX, TorchScript, etc.)
    - model_tags: Many-to-many relationship for flexible tagging

    All operations are transactional with proper error handling and logging.
    The registry is designed for single-node usage (Colab, local dev).

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRegistryEntry:
    """
    Model registry entry with complete metadata.

    Attributes:
        model_id: Unique integer ID (auto-assigned by database)
        name: Human-readable model name (e.g., "transformer-v1")
        version: Semantic version (major.minor.patch, e.g., "1.0.0")
        checkpoint_path: Path to PyTorch checkpoint file
        task_type: Task category (language_modeling, classification, vision, etc.)
        config_hash: SHA-256 hash of model architecture config
        training_run_id: Link to ExperimentDB run (optional)
        parent_model_id: ID of parent model for lineage (optional)
        created_at: ISO timestamp of registration
        metrics: Performance metrics dictionary (JSON serialized)
        export_formats: Available export formats (JSON list)
        model_size_mb: Model size in megabytes
        memory_req_gb: Estimated GPU memory requirement in GB
        metadata: Additional metadata dictionary (JSON serialized)
        status: Model status (active, retired, experimental)
    """
    model_id: int
    name: str
    version: str
    checkpoint_path: str
    task_type: str
    config_hash: str
    training_run_id: Optional[int]
    parent_model_id: Optional[int]
    created_at: str
    metrics: str  # JSON string
    export_formats: str  # JSON string
    model_size_mb: float
    memory_req_gb: float
    metadata: str  # JSON string
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON deserialization."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'version': self.version,
            'checkpoint_path': self.checkpoint_path,
            'task_type': self.task_type,
            'config_hash': self.config_hash,
            'training_run_id': self.training_run_id,
            'parent_model_id': self.parent_model_id,
            'created_at': self.created_at,
            'metrics': json.loads(self.metrics) if self.metrics else {},
            'export_formats': json.loads(self.export_formats) if self.export_formats else [],
            'model_size_mb': self.model_size_mb,
            'memory_req_gb': self.memory_req_gb,
            'metadata': json.loads(self.metadata) if self.metadata else {},
            'status': self.status
        }


class ModelRegistry:
    """
    SQLite-based model registry for versioning and metadata tracking.

    This class provides comprehensive model lifecycle management including:
    - Model registration with automatic version tracking
    - Tag-based organization (production, staging, experimental)
    - Performance metrics storage and comparison
    - Model lineage tracking (parent-child relationships)
    - Export format availability tracking
    - Query and filtering by metrics, tags, task type

    Attributes:
        db_path: Path to SQLite database file.

    Schema:
        models: Model metadata (model_id, name, version, paths, metrics, etc.)
        model_tags: Tag assignments (model_id, tag_name, created_at)
        model_exports: Export format tracking (model_id, format, path, metadata)

    Integration:
        - CheckpointManager: Auto-register on checkpoint save
        - ExperimentDB: Link models to training runs via training_run_id
        - Export utilities: Track available export formats
    """

    def __init__(self, db_path: Union[str, Path] = 'model_registry.db'):
        """
        Initialize model registry with schema creation.

        Args:
            db_path: Path to SQLite database file. Created if doesn't exist.

        Example:
            >>> registry = ModelRegistry('models.db')
            >>> registry = ModelRegistry()  # Uses default 'model_registry.db'
        """
        self.db_path = Path(db_path)
        self._create_schema()
        logger.info(f"Initialized ModelRegistry at {self.db_path}")

    def _create_schema(self) -> None:
        """
        Create database schema if tables don't exist.

        Creates three tables:
        1. models: Core model metadata and versioning
        2. model_tags: Many-to-many tag assignments
        3. model_exports: Export format availability tracking
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    training_run_id INTEGER,
                    parent_model_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metrics TEXT,
                    export_formats TEXT,
                    model_size_mb REAL,
                    memory_req_gb REAL,
                    metadata TEXT,
                    status TEXT DEFAULT 'active',
                    UNIQUE(name, version),
                    FOREIGN KEY (parent_model_id) REFERENCES models (model_id) ON DELETE SET NULL
                )
            ''')

            # Index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_models_task_type
                ON models (task_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_models_status
                ON models (status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_models_config_hash
                ON models (config_hash)
            ''')

            # Model tags table (many-to-many)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_tags (
                    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    tag_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_id, tag_name),
                    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_tags_tag_name
                ON model_tags (tag_name)
            ''')

            # Model exports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_exports (
                    export_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER NOT NULL,
                    export_format TEXT NOT NULL,
                    export_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
                )
            ''')

            conn.commit()
            logger.debug("Model registry schema created/validated")

    def register_model(
        self,
        name: str,
        version: str,
        checkpoint_path: Union[str, Path],
        task_type: str,
        config_hash: str,
        metrics: Dict[str, float],
        export_formats: Optional[List[str]] = None,
        model_size_mb: Optional[float] = None,
        memory_req_gb: Optional[float] = None,
        training_run_id: Optional[int] = None,
        parent_model_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Register a new model with metadata.

        Args:
            name: Human-readable model name (e.g., "transformer-base")
            version: Semantic version string (e.g., "1.0.0")
            checkpoint_path: Path to PyTorch checkpoint file
            task_type: Task category (language_modeling, classification, vision)
            config_hash: SHA-256 hash of model architecture config
            metrics: Performance metrics dict (e.g., {"val_loss": 0.38})
            export_formats: Available export formats (e.g., ["onnx", "torchscript"])
            model_size_mb: Model size in MB (computed if None)
            memory_req_gb: Estimated GPU memory requirement in GB
            training_run_id: Link to ExperimentDB run ID (optional)
            parent_model_id: ID of parent model for lineage tracking (optional)
            metadata: Additional metadata dict (optional)
            tags: Initial tags to assign (e.g., ["experimental", "baseline"])

        Returns:
            model_id: Integer ID for the newly registered model

        Raises:
            ValueError: If model with same name and version already exists
            FileNotFoundError: If checkpoint_path doesn't exist

        Example:
            >>> model_id = registry.register_model(
            ...     name="gpt-small",
            ...     version="1.0.0",
            ...     checkpoint_path="checkpoints/epoch_10.pt",
            ...     task_type="language_modeling",
            ...     config_hash="abc123...",
            ...     metrics={"val_loss": 0.38, "perplexity": 1.46},
            ...     export_formats=["onnx", "torchscript"],
            ...     tags=["baseline", "production"]
            ... )
        """
        # Validate checkpoint exists
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Compute model size if not provided
        if model_size_mb is None:
            model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        # Serialize JSON fields
        metrics_json = json.dumps(metrics, indent=2)
        export_formats_json = json.dumps(export_formats or [])
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    '''
                    INSERT INTO models (
                        name, version, checkpoint_path, task_type, config_hash,
                        training_run_id, parent_model_id, metrics, export_formats,
                        model_size_mb, memory_req_gb, metadata, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    ''',
                    (
                        name,
                        version,
                        str(checkpoint_path),
                        task_type,
                        config_hash,
                        training_run_id,
                        parent_model_id,
                        metrics_json,
                        export_formats_json,
                        model_size_mb,
                        memory_req_gb,
                        metadata_json
                    )
                )
                model_id = cursor.lastrowid
                conn.commit()

                # Add initial tags
                if tags:
                    for tag in tags:
                        self._add_tag(model_id, tag, conn)

                logger.info(f"Registered model {model_id}: '{name}' v{version}")
                return model_id

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    raise ValueError(
                        f"Model '{name}' version '{version}' already exists. "
                        f"Use a different version number or update existing model."
                    )
                raise

    def get_model(
        self,
        model_id: Optional[int] = None,
        name: Optional[str] = None,
        version: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve model by ID, name+version, or tag.

        Args:
            model_id: Unique model ID (highest priority)
            name: Model name (requires version)
            version: Model version (requires name)
            tag: Retrieve model with this tag (e.g., "production")

        Returns:
            Dictionary with model metadata or None if not found:
                - model_id: int
                - name: str
                - version: str
                - checkpoint_path: str
                - task_type: str
                - metrics: dict (deserialized)
                - export_formats: list (deserialized)
                - ... (all other fields)

        Raises:
            ValueError: If invalid argument combination provided

        Example:
            >>> # Get by ID
            >>> model = registry.get_model(model_id=1)
            >>>
            >>> # Get by name and version
            >>> model = registry.get_model(name="gpt-small", version="1.0.0")
            >>>
            >>> # Get production model
            >>> model = registry.get_model(tag="production")
            >>> print(model['checkpoint_path'])
        """
        if model_id is not None:
            return self._get_model_by_id(model_id)
        elif name is not None and version is not None:
            return self._get_model_by_name_version(name, version)
        elif tag is not None:
            return self._get_model_by_tag(tag)
        else:
            raise ValueError(
                "Must provide either model_id, (name and version), or tag"
            )

    def _get_model_by_id(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def _get_model_by_name_version(
        self, name: str, version: str
    ) -> Optional[Dict[str, Any]]:
        """Get model by name and version."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM models WHERE name = ? AND version = ?',
                (name, version)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def _get_model_by_tag(self, tag: str) -> Optional[Dict[str, Any]]:
        """Get most recent model with given tag."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT m.* FROM models m
                JOIN model_tags t ON m.model_id = t.model_id
                WHERE t.tag_name = ?
                ORDER BY m.created_at DESC
                LIMIT 1
                ''',
                (tag,)
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary with JSON deserialization."""
        model_data = dict(row)
        model_data['metrics'] = json.loads(model_data['metrics']) if model_data['metrics'] else {}
        model_data['export_formats'] = json.loads(model_data['export_formats']) if model_data['export_formats'] else []
        model_data['metadata'] = json.loads(model_data['metadata']) if model_data['metadata'] else {}
        return model_data

    def list_models(
        self,
        task_type: Optional[str] = None,
        tag: Optional[str] = None,
        status: str = 'active',
        min_metric_value: Optional[float] = None,
        metric_name: Optional[str] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        List models with optional filtering.

        Args:
            task_type: Filter by task type (e.g., "language_modeling")
            tag: Filter by tag (e.g., "production")
            status: Filter by status (default: "active")
            min_metric_value: Minimum value for metric_name (requires metric_name)
            metric_name: Metric to filter by (e.g., "val_loss")
            limit: Maximum number of models to return

        Returns:
            DataFrame with columns: [model_id, name, version, task_type,
            created_at, status, metrics, export_formats]

        Example:
            >>> # List all active models
            >>> models = registry.list_models()
            >>>
            >>> # List production language models
            >>> models = registry.list_models(
            ...     task_type="language_modeling",
            ...     tag="production"
            ... )
            >>>
            >>> # List models with val_loss < 0.5
            >>> models = registry.list_models(
            ...     metric_name="val_loss",
            ...     min_metric_value=0.5
            ... )
        """
        query = 'SELECT * FROM models WHERE 1=1'
        params: List[Any] = []

        if task_type is not None:
            query += ' AND task_type = ?'
            params.append(task_type)

        if status is not None:
            query += ' AND status = ?'
            params.append(status)

        if tag is not None:
            query = f'''
                SELECT m.* FROM models m
                JOIN model_tags t ON m.model_id = t.model_id
                WHERE t.tag_name = ?
            '''
            params = [tag]
            if task_type is not None:
                query += ' AND m.task_type = ?'
                params.append(task_type)
            if status is not None:
                query += ' AND m.status = ?'
                params.append(status)

        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Filter by metric if specified
        if metric_name is not None and min_metric_value is not None:
            def filter_metric(metrics_json: str) -> bool:
                if not metrics_json:
                    return False
                metrics = json.loads(metrics_json)
                return metrics.get(metric_name, float('inf')) >= min_metric_value

            df = df[df['metrics'].apply(filter_metric)]

        return df

    def promote_model(
        self,
        model_id: int,
        tag: str,
        remove_from_others: bool = True
    ) -> None:
        """
        Promote model to a tag (e.g., "production").

        Args:
            model_id: Model ID to promote
            tag: Tag to assign (e.g., "production", "staging")
            remove_from_others: Remove tag from other models (default: True)

        Example:
            >>> # Promote model to production (demote previous production model)
            >>> registry.promote_model(model_id=5, tag="production")
            >>>
            >>> # Add staging tag without removing from others
            >>> registry.promote_model(
            ...     model_id=5,
            ...     tag="staging",
            ...     remove_from_others=False
            ... )
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Remove tag from other models if specified
            if remove_from_others:
                cursor.execute(
                    'DELETE FROM model_tags WHERE tag_name = ?',
                    (tag,)
                )

            # Add tag to specified model (idempotent)
            try:
                cursor.execute(
                    'INSERT OR IGNORE INTO model_tags (model_id, tag_name) VALUES (?, ?)',
                    (model_id, tag)
                )
                conn.commit()
                logger.info(f"Promoted model {model_id} to tag '{tag}'")
            except sqlite3.IntegrityError as e:
                if "FOREIGN KEY constraint failed" in str(e):
                    raise ValueError(f"Model {model_id} not found in registry")
                raise

    def retire_model(self, model_id: int) -> None:
        """
        Mark model as retired (does not delete).

        Args:
            model_id: Model ID to retire

        Example:
            >>> registry.retire_model(model_id=3)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE models SET status = ? WHERE model_id = ?',
                ('retired', model_id)
            )
            conn.commit()

        logger.info(f"Retired model {model_id}")

    def delete_model(self, model_id: int, force: bool = False) -> None:
        """
        Delete model from registry (use with caution).

        Args:
            model_id: Model ID to delete
            force: Delete even if model is tagged (default: False)

        Raises:
            ValueError: If model is tagged and force=False

        Example:
            >>> # Delete experimental model
            >>> registry.delete_model(model_id=7)
            >>>
            >>> # Force delete tagged model
            >>> registry.delete_model(model_id=7, force=True)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check for tags
            if not force:
                cursor.execute(
                    'SELECT COUNT(*) FROM model_tags WHERE model_id = ?',
                    (model_id,)
                )
                tag_count = cursor.fetchone()[0]
                if tag_count > 0:
                    raise ValueError(
                        f"Model {model_id} has {tag_count} tag(s). "
                        f"Use force=True to delete anyway or remove tags first."
                    )

            # Delete model (cascades to tags and exports)
            cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
            conn.commit()

        logger.warning(f"Deleted model {model_id} from registry")

    def compare_models(
        self,
        model_ids: List[int],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple models.

        Args:
            model_ids: List of model IDs to compare
            metrics: Specific metrics to compare (None for all)

        Returns:
            DataFrame with side-by-side comparison:
                - model_id: int
                - name: str
                - version: str
                - task_type: str
                - created_at: str
                - status: str
                - metric columns (one per metric)

        Example:
            >>> comparison = registry.compare_models([1, 2, 3])
            >>> print(comparison[['name', 'version', 'val_loss', 'perplexity']])
            >>>
            >>> # Compare specific metrics
            >>> comparison = registry.compare_models(
            ...     [1, 2, 3],
            ...     metrics=["val_loss", "accuracy"]
            ... )
        """
        summaries = []

        for model_id in model_ids:
            model = self._get_model_by_id(model_id)
            if model is None:
                logger.warning(f"Model {model_id} not found, skipping")
                continue

            summary = {
                'model_id': model_id,
                'name': model['name'],
                'version': model['version'],
                'task_type': model['task_type'],
                'created_at': model['created_at'],
                'status': model['status'],
                'model_size_mb': model['model_size_mb'],
                'memory_req_gb': model['memory_req_gb']
            }

            # Add metrics
            model_metrics = model['metrics']
            if metrics is None:
                # Include all metrics
                summary.update(model_metrics)
            else:
                # Include only specified metrics
                for metric_name in metrics:
                    summary[metric_name] = model_metrics.get(metric_name)

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def get_model_lineage(self, model_id: int) -> List[Dict[str, Any]]:
        """
        Get model lineage (parent chain).

        Args:
            model_id: Model ID to trace lineage for

        Returns:
            List of model dictionaries from oldest ancestor to specified model

        Example:
            >>> lineage = registry.get_model_lineage(model_id=5)
            >>> for m in lineage:
            ...     print(f"{m['name']} v{m['version']}")
        """
        lineage = []
        current_id = model_id

        while current_id is not None:
            model = self._get_model_by_id(current_id)
            if model is None:
                break
            lineage.append(model)
            current_id = model['parent_model_id']

        return list(reversed(lineage))  # Oldest first

    def add_export_format(
        self,
        model_id: int,
        export_format: str,
        export_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add export format for a model.

        Args:
            model_id: Model ID
            export_format: Format name (e.g., "onnx", "torchscript")
            export_path: Path to exported model file
            metadata: Additional metadata (e.g., {"opset_version": 14})

        Example:
            >>> registry.add_export_format(
            ...     model_id=1,
            ...     export_format="onnx",
            ...     export_path="exports/model.onnx",
            ...     metadata={"opset_version": 14}
            ... )
        """
        metadata_json = json.dumps(metadata or {})

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO model_exports (model_id, export_format, export_path, metadata)
                VALUES (?, ?, ?, ?)
                ''',
                (model_id, export_format, str(export_path), metadata_json)
            )

            # Update model's export_formats list
            cursor.execute(
                'SELECT export_formats FROM models WHERE model_id = ?',
                (model_id,)
            )
            row = cursor.fetchone()
            if row:
                formats = json.loads(row[0]) if row[0] else []
                if export_format not in formats:
                    formats.append(export_format)
                    cursor.execute(
                        'UPDATE models SET export_formats = ? WHERE model_id = ?',
                        (json.dumps(formats), model_id)
                    )

            conn.commit()

        logger.info(f"Added {export_format} export for model {model_id}")

    def _add_tag(
        self,
        model_id: int,
        tag: str,
        conn: Optional[sqlite3.Connection] = None
    ) -> None:
        """Add tag to model (internal helper)."""
        if conn is None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT OR IGNORE INTO model_tags (model_id, tag_name) VALUES (?, ?)',
                    (model_id, tag)
                )
                conn.commit()
        else:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR IGNORE INTO model_tags (model_id, tag_name) VALUES (?, ?)',
                (model_id, tag)
            )

    @staticmethod
    def compute_config_hash(config: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of model config.

        Args:
            config: Model configuration dictionary

        Returns:
            Hexadecimal hash string (64 characters)

        Example:
            >>> config = {"d_model": 768, "num_layers": 12, "vocab_size": 50257}
            >>> config_hash = ModelRegistry.compute_config_hash(config)
            >>> print(config_hash)  # "abc123..."
        """
        config_json = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()


# Public API
__all__ = [
    'ModelRegistry',
    'ModelRegistryEntry',
]
