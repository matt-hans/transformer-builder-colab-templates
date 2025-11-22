"""
Checkpoint Recovery Utilities.

Provides clean API for recovering training results from checkpoints.
Useful for interrupted training, analysis, and resume workflows.

Example:
    >>> from utils.training.engine.recovery import recover_training_results
    >>>
    >>> # Recover from specific checkpoint
    >>> results = recover_training_results('checkpoint_epoch0009_step000009_20251122_065455.pt')
    >>> print(f"Final loss: {results['loss_history'][-1]}")
    >>>
    >>> # Recover from best checkpoint in directory
    >>> results = recover_training_results(checkpoint_dir='./checkpoints')
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import torch


def recover_training_results(
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> Dict[str, Any]:
    """
    Recover training results from checkpoint.

    Extracts metrics history and formats it exactly like Trainer.train() return value.
    Provides backward-compatible API (loss_history) for notebook compatibility.

    Args:
        checkpoint_path: Path to specific checkpoint file (takes precedence)
        checkpoint_dir: Directory to search for best checkpoint
        monitor: Metric to use for finding best checkpoint (if checkpoint_path not provided)
        mode: 'min' or 'max' for finding best checkpoint

    Returns:
        Dictionary matching Trainer.train() return format:
            {
                'metrics_summary': pd.DataFrame,  # Per-epoch metrics
                'best_epoch': int,                # Best epoch number
                'final_loss': float,              # Final training loss
                'checkpoint_path': str,           # Path to loaded checkpoint
                'training_time': float,           # Total training time (if available)
                'loss_history': List[float],      # Backward-compatible
                'val_loss_history': List[float]   # Backward-compatible
            }

    Raises:
        FileNotFoundError: If checkpoint not found
        ValueError: If checkpoint missing required data

    Example:
        >>> # Recover from specific checkpoint
        >>> results = recover_training_results('checkpoint_epoch0009.pt')
        >>>
        >>> # Recover from best checkpoint in directory
        >>> results = recover_training_results(checkpoint_dir='./checkpoints', monitor='val_loss')
        >>>
        >>> # Use results in notebook
        >>> print(f"Train Loss: {results['loss_history'][-1]:.4f}")
        >>> print(f"Val Loss: {results['val_loss_history'][-1]:.4f}")
    """
    # Load checkpoint
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"📂 Loaded checkpoint: {ckpt_path.name}")
    elif checkpoint_dir:
        # Find best checkpoint in directory
        ckpt_path = _find_best_checkpoint(checkpoint_dir, monitor, mode)
        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoints found in: {checkpoint_dir}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"📂 Loaded best checkpoint: {ckpt_path.name}")
    else:
        raise ValueError("Must provide either checkpoint_path or checkpoint_dir")

    # Extract metrics history from custom_state
    custom_state = checkpoint.get('custom_state', {})
    metrics_history = custom_state.get('metrics_history')

    if metrics_history is None:
        raise ValueError(
            "Checkpoint does not contain metrics_history. "
            "This checkpoint was created before v4.0 or training failed before first epoch."
        )

    # Convert to DataFrame (modern API)
    metrics_df = pd.DataFrame(metrics_history)

    # Get basic checkpoint info
    epoch = checkpoint.get('epoch', len(metrics_history) - 1)
    training_time = custom_state.get('training_time', 0.0)

    # Extract workspace_root and run_name from checkpoint (v4.0+)
    # Fallback to path parsing for legacy checkpoints (v3.x)
    if 'workspace_root' in custom_state:
        workspace_root = custom_state['workspace_root']
        run_name = custom_state.get('run_name', 'recovered_run')
        print(f"✅ Session metadata loaded from checkpoint (v4.0+)")
    else:
        # Legacy checkpoint (v3.x) - infer from path
        workspace_root = str(ckpt_path.parent.parent) if ckpt_path.parent.name == 'checkpoints' else str(ckpt_path.parent)
        run_name = '_'.join(ckpt_path.stem.split('_')[:3]) if '_' in ckpt_path.stem else ckpt_path.stem
        print(f"⚠️  Legacy checkpoint detected (v3.x) - workspace_root/run_name inferred from path")

    # Format results exactly like Trainer._format_results() (L1090-1114)
    results = {
        'metrics_summary': metrics_df,
        'best_epoch': epoch,
        'final_loss': metrics_df['train/loss'].iloc[-1] if not metrics_df.empty else 0.0,
        'checkpoint_path': str(ckpt_path),
        'training_time': training_time,
        'workspace_root': workspace_root,
        'run_name': run_name,

        # Legacy compatibility (v3.x) - matches trainer.py L1100-1114
        'loss_history': metrics_df['train/loss'].tolist() if not metrics_df.empty else [],
        'val_loss_history': metrics_df['val/loss'].tolist() if 'val/loss' in metrics_df.columns and not metrics_df.empty else []
    }

    print(f"✅ Recovered {len(metrics_history)} epochs of training history")
    print(f"   Final train_loss: {results['final_loss']:.4f}")
    if results['val_loss_history']:
        print(f"   Final val_loss: {results['val_loss_history'][-1]:.4f}")

    return results


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    List all checkpoints in directory with metadata.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint metadata dictionaries, sorted by epoch (descending)

    Example:
        >>> checkpoints = list_checkpoints('./checkpoints')
        >>> for ckpt in checkpoints:
        ...     print(f"Epoch {ckpt['epoch']}: val_loss={ckpt['val_loss']:.4f}")
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for ckpt_file in ckpt_dir.glob('checkpoint_*.pt'):
        try:
            # Load checkpoint metadata only (fast)
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            metrics = ckpt.get('metrics', {})

            checkpoints.append({
                'path': str(ckpt_file),
                'filename': ckpt_file.name,
                'epoch': ckpt.get('epoch', 0),
                'global_step': ckpt.get('global_step', 0),
                'train_loss': metrics.get('train_loss', 0.0),
                'val_loss': metrics.get('val_loss', 0.0),
                'timestamp': ckpt.get('timestamp', ''),
            })
        except Exception as e:
            print(f"⚠️  Skipping corrupted checkpoint {ckpt_file.name}: {e}")
            continue

    # Sort by epoch (most recent first)
    return sorted(checkpoints, key=lambda c: c['epoch'], reverse=True)


def _find_best_checkpoint(
    checkpoint_dir: str,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> Optional[Path]:
    """Find best checkpoint by metric."""
    checkpoints = list_checkpoints(checkpoint_dir)

    if not checkpoints:
        return None

    # Map monitor to checkpoint metadata key
    metric_key = monitor.replace('/', '_')  # 'val/loss' → 'val_loss'

    # Filter checkpoints that have the metric
    valid_ckpts = [c for c in checkpoints if metric_key in c]

    if not valid_ckpts:
        # Fallback to most recent
        return Path(checkpoints[0]['path'])

    # Find best by metric
    if mode == 'min':
        best = min(valid_ckpts, key=lambda c: c[metric_key])
    else:
        best = max(valid_ckpts, key=lambda c: c[metric_key])

    return Path(best['path'])


__all__ = [
    'recover_training_results',
    'list_checkpoints',
]
