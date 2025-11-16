"""
Training resume utilities for non-Lightning workflows.

Provides a helper to resume from state_dict checkpoints saved by
save_checkpoint_with_progress (epoch_*.pt or best.pt).
"""

from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .checkpoint_manager import (
        find_latest_checkpoint_in_dir,
        load_checkpoint_with_progress,
    )
    from .seed_manager import set_random_seed
except Exception:
    # Fallback for direct import by file path (tests)
    import importlib.util as _ilu
    base = Path(__file__).parent
    cm_path = base / 'checkpoint_manager.py'
    sm_path = base / 'seed_manager.py'
    spec_cm = _ilu.spec_from_file_location('checkpoint_manager', str(cm_path))
    mod_cm = _ilu.module_from_spec(spec_cm)
    assert spec_cm and spec_cm.loader
    spec_cm.loader.exec_module(mod_cm)  # type: ignore
    try:
        spec_sm = _ilu.spec_from_file_location('seed_manager', str(sm_path))
        mod_sm = _ilu.module_from_spec(spec_sm)
        assert spec_sm and spec_sm.loader
        spec_sm.loader.exec_module(mod_sm)  # type: ignore
        set_random_seed = mod_sm.set_random_seed
    except Exception:
        # No-op fallback
        def set_random_seed(seed: int, deterministic: bool = False):  # type: ignore
            return None
    find_latest_checkpoint_in_dir = mod_cm.find_latest_checkpoint_in_dir
    load_checkpoint_with_progress = mod_cm.load_checkpoint_with_progress


def resume_training_from_checkpoint(checkpoint_dir: str,
                                    model,
                                    optimizer=None,
                                    lr_scheduler=None,
                                    best_metric_key: str = 'val_loss') -> Dict[str, Any]:
    """
    Resume training from the latest state_dict checkpoint in a directory.

    Returns a dict with start_epoch, metrics, and config.
    """
    d = Path(checkpoint_dir)
    latest = find_latest_checkpoint_in_dir(str(d))
    if latest is None:
        print("ℹ️  No state_dict checkpoint found - starting from scratch")
        return {'start_epoch': 0, 'metrics': {}, 'config': None}

    print("=" * 60)
    print(f"📂 Found checkpoint: {Path(latest).name}")
    ckpt = load_checkpoint_with_progress(latest, model, optimizer)
    start_epoch = int(ckpt.get('epoch', -1)) + 1
    metrics = ckpt.get('metrics', {}) or {}
    cfg = ckpt.get('config')

    # Restore LR scheduler if state present
    if lr_scheduler is not None and 'scheduler_state_dict' in ckpt:
        try:
            lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("✅ Learning rate scheduler restored")
        except Exception:
            pass

    # Restore seed if present
    try:
        seed = ckpt.get('config', {}).get('random_seed')
        if seed is not None:
            set_random_seed(int(seed))
            print("✅ Random seed restored")
    except Exception:
        pass

    print(f"🚀 Resuming training from epoch {start_epoch}")
    return {'start_epoch': start_epoch, 'metrics': metrics, 'config': cfg}
