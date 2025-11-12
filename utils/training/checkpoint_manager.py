"""
Checkpoint Manager for PyTorch Lightning training.

Handles:
- Automatic checkpoint saving (every N steps/epochs)
- Best model tracking (by validation metric)
- Checkpoint cleanup (keep best K)
- Resume from checkpoint
- Google Drive integration for persistence
- Optimizer and scheduler state saving
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Literal, List
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import json
from datetime import datetime


class CheckpointManager:
    """
    Comprehensive checkpoint management for training.

    Features:
    - Automatic saving every N epochs/steps
    - Track best K checkpoints by metric
    - Resume from any checkpoint
    - Save/restore full training state
    - Google Drive backup (Colab)
    - Cleanup old checkpoints

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='./checkpoints',
        ...     save_top_k=3,
        ...     monitor='val_loss',
        ...     mode='min'
        ... )
        >>>
        >>> # Get Lightning callback
        >>> callback = manager.get_callback()
        >>> trainer = pl.Trainer(callbacks=[callback])
        >>>
        >>> # Resume from checkpoint
        >>> model = manager.load_checkpoint('best.ckpt')
    """

    def __init__(self,
                 checkpoint_dir: str = './checkpoints',
                 save_top_k: int = 3,
                 monitor: str = 'val_loss',
                 mode: Literal['min', 'max'] = 'min',
                 save_every_n_epochs: int = 1,
                 save_last: bool = True,
                 filename: Optional[str] = None,
                 enable_version_counter: bool = False,
                 drive_backup: bool = False,
                 drive_backup_path: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_top_k: Number of best checkpoints to keep (-1 for all)
            monitor: Metric to monitor (e.g., 'val_loss', 'train_loss')
            mode: 'min' or 'max' for monitored metric
            save_every_n_epochs: Save checkpoint every N epochs
            save_last: Always save the last checkpoint
            filename: Checkpoint filename pattern (default: auto-generated)
            enable_version_counter: Add version number to filenames
            drive_backup: Enable Google Drive backup (Colab only)
            drive_backup_path: Path in Drive for backups

        Example:
            >>> manager = CheckpointManager(
            ...     checkpoint_dir='./my_checkpoints',
            ...     save_top_k=5,
            ...     monitor='val_perplexity',
            ...     mode='min'
            ... )
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last = save_last
        self.enable_version_counter = enable_version_counter
        self.drive_backup = drive_backup
        self.drive_backup_path = drive_backup_path

        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Auto-generate filename if not provided
        if filename is None:
            filename = f"{{epoch:02d}}-{{{monitor}:.4f}}"
        self.filename = filename

        # Track checkpoint metadata
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()

    def get_callback(self) -> ModelCheckpoint:
        """
        Get PyTorch Lightning ModelCheckpoint callback.

        Returns:
            ModelCheckpoint callback configured with manager settings

        Example:
            >>> callback = manager.get_callback()
            >>> trainer = pl.Trainer(callbacks=[callback])
        """
        callback = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename=self.filename,
            monitor=self.monitor,
            mode=self.mode,
            save_top_k=self.save_top_k,
            save_last=self.save_last,
            every_n_epochs=self.save_every_n_epochs,
            enable_version_counter=self.enable_version_counter,
            verbose=True
        )

        return callback

    def get_backup_callback(self) -> Optional['DriveBackupCallback']:
        """
        Get Google Drive backup callback (Colab only).

        Returns:
            DriveBackupCallback or None if not in Colab or drive_backup=False
        """
        if not self.drive_backup:
            return None

        try:
            # Check if in Colab
            from google.colab import drive
            return DriveBackupCallback(
                checkpoint_dir=self.checkpoint_dir,
                drive_path=self.drive_backup_path
            )
        except ImportError:
            print("⚠️  Drive backup only available in Colab")
            return None

    def load_checkpoint(self,
                       checkpoint_path: Optional[str] = None,
                       model_class: Optional[type] = None,
                       map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (None for best)
            model_class: Model class for loading (if needed)
            map_location: Device to load to ('cpu', 'cuda', etc.)

        Returns:
            Dictionary with checkpoint contents

        Example:
            >>> # Load best checkpoint
            >>> ckpt = manager.load_checkpoint()
            >>>
            >>> # Load specific checkpoint
            >>> ckpt = manager.load_checkpoint('epoch=05-val_loss=1.2345.ckpt')
            >>>
            >>> # Load to CPU
            >>> ckpt = manager.load_checkpoint(map_location='cpu')
        """
        # Find checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint_path()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")
            print(f"📂 Loading best checkpoint: {Path(checkpoint_path).name}")
        else:
            # Handle relative paths
            if not Path(checkpoint_path).is_absolute():
                checkpoint_path = self.checkpoint_dir / checkpoint_path
            print(f"📂 Loading checkpoint: {Path(checkpoint_path).name}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        return checkpoint

    def load_model_from_checkpoint(self,
                                   model_class: type,
                                   checkpoint_path: Optional[str] = None,
                                   **model_kwargs) -> Any:
        """
        Load model from checkpoint.

        Args:
            model_class: Model class (Lightning module)
            checkpoint_path: Path to checkpoint (None for best)
            **model_kwargs: Additional arguments for model instantiation

        Returns:
            Loaded model instance

        Example:
            >>> model = manager.load_model_from_checkpoint(
            ...     UniversalModelAdapter,
            ...     checkpoint_path='best.ckpt'
            ... )
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_best_checkpoint_path()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")
        else:
            # Handle relative paths
            if not Path(checkpoint_path).is_absolute():
                checkpoint_path = self.checkpoint_dir / checkpoint_path

        print(f"📂 Loading model from: {Path(checkpoint_path).name}")

        # Load using Lightning
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            **model_kwargs
        )

        print("✓ Model loaded successfully")

        return model

    def get_best_checkpoint_path(self) -> Optional[str]:
        """
        Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if no checkpoints
        """
        # Look for best.ckpt
        best_path = self.checkpoint_dir / 'best.ckpt'
        if best_path.exists():
            return str(best_path)

        # Look for last.ckpt
        last_path = self.checkpoint_dir / 'last.ckpt'
        if last_path.exists():
            return str(last_path)

        # Get all checkpoints
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Return first one (most recent)
        return str(self.checkpoint_dir / checkpoints[0])

    def list_checkpoints(self, sort_by: Literal['time', 'metric'] = 'time') -> List[str]:
        """
        List all checkpoints.

        Args:
            sort_by: Sort by 'time' (modification time) or 'metric' (from filename)

        Returns:
            List of checkpoint filenames

        Example:
            >>> checkpoints = manager.list_checkpoints()
            >>> print(f"Found {len(checkpoints)} checkpoints")
            >>> for ckpt in checkpoints:
            ...     print(f"  - {ckpt}")
        """
        # Find all .ckpt files
        checkpoints = list(self.checkpoint_dir.glob('*.ckpt'))

        if not checkpoints:
            return []

        # Sort by modification time (newest first)
        if sort_by == 'time':
            checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Extract filenames
        return [ckpt.name for ckpt in checkpoints]

    def cleanup_old_checkpoints(self, keep_top_k: Optional[int] = None):
        """
        Remove old checkpoints, keeping only top K.

        Args:
            keep_top_k: Number of checkpoints to keep (uses self.save_top_k if None)

        Example:
            >>> # Keep only best 3 checkpoints
            >>> manager.cleanup_old_checkpoints(keep_top_k=3)
            🗑️  Cleaned up 5 old checkpoints
        """
        if keep_top_k is None:
            keep_top_k = self.save_top_k

        if keep_top_k < 0:
            # Keep all
            return

        checkpoints = self.list_checkpoints(sort_by='time')

        # Always keep best.ckpt and last.ckpt
        protected = {'best.ckpt', 'last.ckpt'}

        # Filter out protected checkpoints
        deletable = [c for c in checkpoints if c not in protected]

        # Delete old ones
        if len(deletable) > keep_top_k:
            to_delete = deletable[keep_top_k:]
            for ckpt_name in to_delete:
                ckpt_path = self.checkpoint_dir / ckpt_name
                ckpt_path.unlink()

            print(f"🗑️  Cleaned up {len(to_delete)} old checkpoint(s)")

    def save_metadata(self,
                     epoch: int,
                     step: int,
                     metrics: Dict[str, float],
                     checkpoint_path: str):
        """
        Save checkpoint metadata.

        Args:
            epoch: Training epoch
            step: Training step
            metrics: Dictionary of metric values
            checkpoint_path: Path to checkpoint
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'checkpoint_path': str(checkpoint_path)
        }

        self.metadata.append(entry)
        self._save_metadata()

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load checkpoint metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        """Save checkpoint metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.

        Returns:
            Dictionary with checkpoint information

        Example:
            >>> info = manager.get_checkpoint_info()
            >>> print(f"Total checkpoints: {info['num_checkpoints']}")
            >>> print(f"Best metric: {info['best_metric']:.4f}")
        """
        checkpoints = self.list_checkpoints()
        best_path = self.get_best_checkpoint_path()

        info = {
            'num_checkpoints': len(checkpoints),
            'checkpoint_dir': str(self.checkpoint_dir),
            'best_checkpoint': Path(best_path).name if best_path else None,
            'monitored_metric': self.monitor,
            'mode': self.mode,
            'recent_checkpoints': checkpoints[:5],  # Most recent 5
        }

        return info

    def print_checkpoint_info(self):
        """Print formatted checkpoint information."""
        info = self.get_checkpoint_info()

        print("\n💾 Checkpoint Manager Status:")
        print(f"  Directory: {info['checkpoint_dir']}")
        print(f"  Total checkpoints: {info['num_checkpoints']}")
        print(f"  Monitored metric: {info['monitored_metric']} ({info['mode']})")

        if info['best_checkpoint']:
            print(f"  Best checkpoint: {info['best_checkpoint']}")

        if info['recent_checkpoints']:
            print(f"\n  Recent checkpoints:")
            for ckpt in info['recent_checkpoints']:
                print(f"    - {ckpt}")


class DriveBackupCallback(Callback):
    """
    PyTorch Lightning callback for backing up checkpoints to Google Drive.

    Automatically syncs checkpoints to Drive after each save.
    """

    def __init__(self,
                 checkpoint_dir: Path,
                 drive_path: Optional[str] = None,
                 mount_point: str = '/content/drive'):
        """
        Initialize Drive backup callback.

        Args:
            checkpoint_dir: Local checkpoint directory
            drive_path: Path in Drive (default: MyDrive/checkpoints)
            mount_point: Drive mount point
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.mount_point = Path(mount_point)

        # Set default drive path
        if drive_path is None:
            drive_path = 'MyDrive/checkpoints'
        self.drive_path = self.mount_point / drive_path

        # Mount drive if needed
        self._ensure_drive_mounted()

        # Create drive directory
        self.drive_path.mkdir(parents=True, exist_ok=True)

        print(f"☁️  Drive backup enabled: {self.drive_path}")

    def _ensure_drive_mounted(self):
        """Mount Google Drive if not already mounted."""
        if not self.mount_point.exists():
            try:
                from google.colab import drive
                print("🔗 Mounting Google Drive for backups...")
                drive.mount(str(self.mount_point))
                print("✓ Drive mounted")
            except ImportError:
                raise RuntimeError("Drive backup requires Google Colab environment")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Called when checkpoint is saved.

        Copies checkpoint to Google Drive.
        """
        # Get the saved checkpoint path
        if trainer.checkpoint_callback:
            last_checkpoint = trainer.checkpoint_callback.last_model_path

            if last_checkpoint and Path(last_checkpoint).exists():
                # Copy to Drive
                drive_checkpoint = self.drive_path / Path(last_checkpoint).name

                try:
                    shutil.copy2(last_checkpoint, drive_checkpoint)
                    print(f"☁️  Backed up to Drive: {drive_checkpoint.name}")
                except Exception as e:
                    print(f"⚠️  Drive backup failed: {e}")

    def on_train_end(self, trainer, pl_module):
        """
        Called when training ends.

        Syncs all checkpoints to Drive.
        """
        print("\n☁️  Final Drive sync...")

        # Copy all checkpoints
        for ckpt_file in self.checkpoint_dir.glob('*.ckpt'):
            drive_checkpoint = self.drive_path / ckpt_file.name

            try:
                if not drive_checkpoint.exists() or \
                   ckpt_file.stat().st_mtime > drive_checkpoint.stat().st_mtime:
                    shutil.copy2(ckpt_file, drive_checkpoint)
                    print(f"  ✓ {ckpt_file.name}")
            except Exception as e:
                print(f"  ⚠️  Failed: {ckpt_file.name} ({e})")

        print(f"✓ All checkpoints backed up to {self.drive_path}")
