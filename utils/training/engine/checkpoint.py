"""
Checkpoint Manager for PyTorch training.

Provides comprehensive state management for training resume and recovery.
Works standalone (no PyTorch Lightning dependency).

Features:
- Epoch-based and step-based checkpointing
- RNG state preservation for reproducibility
- Best model tracking by metric
- Configurable retention policy
- Google Drive backup for Colab persistence
- Atomic writes (corruption-safe)
- Custom state injection (for strategies, trackers)
"""

import os
import shutil
import random
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np


@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Metadata for checkpoint tracking.

    Attributes:
        epoch: Training epoch number
        global_step: Total batches processed across all epochs
        best_metric: Best monitored metric value so far
        timestamp: ISO 8601 timestamp of checkpoint creation
        git_commit: Git commit hash (if available)
        metrics: Dictionary of metrics at this checkpoint (stored as JSON string for hashability)
        config: Training configuration snapshot (stored as JSON string for hashability)
    """
    epoch: int
    global_step: int
    best_metric: float
    timestamp: str
    git_commit: Optional[str] = None
    metrics: Optional[str] = None  # JSON string for hashability
    config: Optional[str] = None  # JSON string for hashability

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'timestamp': self.timestamp,
            'git_commit': self.git_commit,
            'metrics': json.loads(self.metrics) if self.metrics else None,
            'config': json.loads(self.config) if self.config else None
        }
        return result


class CheckpointManager:
    """
    Comprehensive checkpoint management for training.

    Handles save, load, resume, and cleanup of training checkpoints with
    full state preservation (model, optimizer, scheduler, RNG, custom state).

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='./checkpoints',
        ...     keep_best_k=3,
        ...     keep_last_n=5,
        ...     monitor='val_loss',
        ...     mode='min'
        ... )
        >>>
        >>> # Save checkpoint
        >>> path = manager.save(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     epoch=5,
        ...     metrics={'val_loss': 0.38, 'train_loss': 0.42},
        ...     custom_state={'strategy_config': {...}}
        ... )
        >>>
        >>> # Load checkpoint
        >>> state = manager.load(path)
        >>> model.load_state_dict(state['model_state_dict'])
        >>> optimizer.load_state_dict(state['optimizer_state_dict'])
        >>>
        >>> # Resume training
        >>> start_epoch = state['epoch'] + 1
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_k: int = 3,
        keep_last_n: int = 5,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_interval_epochs: int = 1,
        drive_backup: bool = False,
        drive_backup_path: Optional[str] = None
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_k: Number of best checkpoints to keep (by monitor metric)
            keep_last_n: Number of most recent checkpoints to keep
            monitor: Metric to monitor (e.g., 'val_loss', 'val_accuracy')
            mode: 'min' or 'max' for monitored metric
            save_interval_epochs: Save checkpoint every N epochs
            drive_backup: Enable Google Drive backup (Colab only)
            drive_backup_path: Path in Drive for backups (e.g., '/content/drive/MyDrive/checkpoints')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_best_k = keep_best_k
        self.keep_last_n = keep_last_n
        self.monitor = monitor
        self.mode = mode
        self.save_interval_epochs = save_interval_epochs
        self.drive_backup = drive_backup
        self.drive_backup_path = Path(drive_backup_path) if drive_backup_path else None

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.drive_backup and self.drive_backup_path:
            self.drive_backup_path.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.metadata_file = self.checkpoint_dir / 'checkpoint_registry.json'
        self.checkpoints: List[CheckpointMetadata] = self._load_registry()

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        global_step: Optional[int] = None,
        custom_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint with full training state.

        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            metrics: Dictionary of metrics (must include monitor metric)
            global_step: Total batches processed (optional)
            custom_state: Additional state to save (e.g., loss strategy config)

        Returns:
            Path to saved checkpoint file

        Raises:
            ValueError: If monitor metric not in metrics dict
        """
        if self.monitor not in metrics:
            raise ValueError(
                f"Monitor metric '{self.monitor}' not found in metrics. "
                f"Available: {list(metrics.keys())}"
            )

        # Compute global step if not provided
        if global_step is None:
            global_step = epoch

        # Get current best metric
        best_metric = metrics[self.monitor]

        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:04d}_step{global_step:06d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / filename

        # Prepare checkpoint state
        checkpoint_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'monitor': self.monitor,
            'monitor_value': best_metric,
            'rng_state': self._capture_rng_state(),
            'timestamp': timestamp,
            'git_commit': self._get_git_commit(),
            'custom_state': custom_state or {}
        }

        # Atomic write: save to temporary file, then rename
        tmp_path = checkpoint_path.with_suffix('.tmp')
        try:
            torch.save(checkpoint_state, tmp_path)
            tmp_path.rename(checkpoint_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to save checkpoint: {e}")

        # Update metadata registry (serialize dicts to JSON strings for hashability)
        metadata = CheckpointMetadata(
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            timestamp=timestamp,
            git_commit=checkpoint_state['git_commit'],
            metrics=json.dumps(metrics),
            config=json.dumps(custom_state.get('training_config')) if custom_state and custom_state.get('training_config') else None
        )
        self.checkpoints.append(metadata)
        self._save_registry()

        # Drive backup if enabled
        if self.drive_backup and self.drive_backup_path:
            self._backup_to_drive(checkpoint_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        print(f"✓ Checkpoint saved: {checkpoint_path.name}")
        return checkpoint_path

    def load(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load checkpoint and return state dictionary.

        Args:
            checkpoint_path: Path to checkpoint (None for best checkpoint)

        Returns:
            Dictionary with checkpoint state:
                - epoch: int
                - global_step: int
                - model_state_dict: OrderedDict
                - optimizer_state_dict: dict
                - scheduler_state_dict: dict
                - metrics: dict
                - rng_state: dict
                - custom_state: dict

        Raises:
            FileNotFoundError: If no checkpoints found
            RuntimeError: If checkpoint file is corrupted
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_best()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found in registry")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"📂 Loading checkpoint: {checkpoint_path.name}")

        try:
            # PyTorch 2.6+ requires weights_only=False for full checkpoint loading (RNG state, optimizer, etc.)
            checkpoint_raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint (file may be corrupted): {e}\n"
                f"Recovery: Delete corrupted file and resume from earlier checkpoint."
            )

        # Cast to Dict for type safety (torch.load returns Any)
        if not isinstance(checkpoint_raw, dict):
            raise RuntimeError("Checkpoint file contains invalid data (not a dictionary)")
        checkpoint: Dict[str, Any] = checkpoint_raw

        # Validate checkpoint structure
        required_keys = ['epoch', 'model_state_dict', 'metrics']
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {missing}")

        # Restore RNG state for reproducibility
        if 'rng_state' in checkpoint:
            self._restore_rng_state(checkpoint['rng_state'])
            print("✓ RNG state restored (reproducible resume)")

        print(f"✓ Checkpoint loaded: epoch {checkpoint['epoch']}, step {checkpoint.get('global_step', '?')}")
        return checkpoint

    def get_best(self) -> Optional[Path]:
        """
        Get path to best checkpoint (by monitor metric).

        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None

        # Sort by monitor metric
        if self.mode == 'min':
            best_ckpt = min(self.checkpoints, key=lambda c: c.best_metric)
        else:
            best_ckpt = max(self.checkpoints, key=lambda c: c.best_metric)

        # Find checkpoint file
        checkpoint_pattern = f"checkpoint_epoch{best_ckpt.epoch:04d}_step{best_ckpt.global_step:06d}_*.pt"
        matches = list(self.checkpoint_dir.glob(checkpoint_pattern))
        return matches[0] if matches else None

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """
        List all checkpoints sorted by epoch (descending).

        Returns:
            List of CheckpointMetadata objects
        """
        return sorted(self.checkpoints, key=lambda c: c.epoch, reverse=True)

    def _capture_rng_state(self) -> Dict[str, Any]:
        """Capture RNG states for reproducibility."""
        rng_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            rng_state['cuda'] = torch.cuda.get_rng_state_all()

        return rng_state

    def _restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        """Restore RNG states for reproducible resume."""
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])

        if torch.cuda.is_available() and 'cuda' in rng_state:
            torch.cuda.set_rng_state_all(rng_state['cuda'])

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=False,
                timeout=1
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _backup_to_drive(self, checkpoint_path: Path) -> None:
        """Backup checkpoint to Google Drive (Colab only)."""
        if not self.drive_backup_path:
            return

        try:
            drive_ckpt_path = self.drive_backup_path / checkpoint_path.name
            shutil.copy2(checkpoint_path, drive_ckpt_path)
            print(f"☁️  Backed up to Drive: {drive_ckpt_path}")
        except Exception as e:
            print(f"⚠️  Drive backup failed: {e}")

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints based on retention policy."""
        # Sort by metric (best first)
        if self.mode == 'min':
            sorted_by_metric = sorted(self.checkpoints, key=lambda c: c.best_metric)
        else:
            sorted_by_metric = sorted(self.checkpoints, key=lambda c: c.best_metric, reverse=True)

        # Sort by epoch (most recent first)
        sorted_by_epoch = sorted(self.checkpoints, key=lambda c: c.epoch, reverse=True)

        # Keep best K + last N (union)
        keep_best = set(sorted_by_metric[:self.keep_best_k])
        keep_recent = set(sorted_by_epoch[:self.keep_last_n])
        keep = keep_best | keep_recent

        # Delete checkpoints not in keep set
        for ckpt in self.checkpoints[:]:
            if ckpt not in keep:
                pattern = f"checkpoint_epoch{ckpt.epoch:04d}_step{ckpt.global_step:06d}_*.pt"
                for file in self.checkpoint_dir.glob(pattern):
                    file.unlink()
                    print(f"🗑️  Removed old checkpoint: {file.name}")
                self.checkpoints.remove(ckpt)

        self._save_registry()

    def _load_registry(self) -> List[CheckpointMetadata]:
        """Load checkpoint registry from JSON."""
        if not self.metadata_file.exists():
            return []

        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)

            # Convert dict format to CheckpointMetadata (serialize metrics/config back to JSON strings)
            checkpoints = []
            for item in data:
                checkpoints.append(CheckpointMetadata(
                    epoch=item['epoch'],
                    global_step=item['global_step'],
                    best_metric=item['best_metric'],
                    timestamp=item['timestamp'],
                    git_commit=item.get('git_commit'),
                    metrics=json.dumps(item['metrics']) if item.get('metrics') else None,
                    config=json.dumps(item['config']) if item.get('config') else None
                ))
            return checkpoints
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint registry: {e}")
            return []

    def _save_registry(self) -> None:
        """Save checkpoint registry to JSON."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump([c.to_dict() for c in self.checkpoints], f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint registry: {e}")
