"""
Tests for Drive backup integration behavior (graceful fallback when not in Colab).

These tests avoid requiring torch or google.colab; they validate that
requesting a backup callback in a non-Colab environment does not raise
and returns None, keeping training functional.
"""

import os
import sys


def test_get_backup_callback_non_colab_graceful():
    # Ensure google.colab import fails
    if 'google' in sys.modules:
        sys.modules.pop('google')
    if 'google.colab' in sys.modules:
        sys.modules.pop('google.colab')

    # Ensure repo root on sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Insert a dummy 'torch' module to satisfy imports (we don't call torch APIs here)
    import types
    dummy_torch = types.ModuleType('torch')
    dummy_torch.Tensor = object
    dummy_torch.load = lambda *args, **kwargs: {}
    dummy_torch.save = lambda *args, **kwargs: None
    sys.modules['torch'] = dummy_torch

    # Import checkpoint_manager module directly to avoid package-level imports requiring real torch
    import importlib.util
    cm_path = os.path.join(repo_root, 'utils', 'training', 'checkpoint_manager.py')
    spec = importlib.util.spec_from_file_location('checkpoint_manager', cm_path)
    cm = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cm)  # type: ignore

    CheckpointManager = cm.CheckpointManager

    cm = CheckpointManager(
        checkpoint_dir='./tmp_ckpt_test',
        drive_backup=True,
        drive_backup_path='MyDrive/checkpoints/test_run'
    )

    cb = cm.get_backup_callback()
    # In non-Colab env, should gracefully return None (not raise)
    assert cb is None
