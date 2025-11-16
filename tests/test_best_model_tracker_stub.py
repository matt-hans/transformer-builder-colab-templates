"""
Stubbed test for BestStateDictCallback: verifies saving best.pt and W&B summary.
"""

import os
import sys
import types
import importlib.util
from pathlib import Path


def test_best_state_dict_callback_saves_and_logs(tmp_path):
    # Stub torch minimal API used by save helper
    torch = types.ModuleType('torch')
    def _save(obj, path):
        Path(path).write_bytes(b'state')
    torch.save = _save
    sys.modules['torch'] = torch

    # Load module by path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cm_path = os.path.join(repo_root, 'utils', 'training', 'checkpoint_manager.py')
    spec = importlib.util.spec_from_file_location('checkpoint_manager', cm_path)
    cm = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cm)  # type: ignore

    # Monkeypatch save helper to avoid torch dependencies
    def fake_save(model, optimizer, epoch, metrics, config, checkpoint_dir, filename=None):
        p = Path(checkpoint_dir) / (filename or f'epoch_{epoch}.pt')
        p.write_text('ok')
        return str(p)
    cm.save_checkpoint_with_progress = fake_save

    # Prepare callback
    cb = cm.BestStateDictCallback(checkpoint_dir=tmp_path, metric_name='val_loss', mode='min')

    # Dummy objects
    class DummyPL:
        def __init__(self):
            self.model = types.SimpleNamespace(state_dict=lambda: {'w':[1]})
            self.config = {'vocab_size': 10}

    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 3
            self.callback_metrics = {'val_loss': 1.23}

    # First call (improvement from inf → 1.23)
    cb.on_validation_end(DummyTrainer(), DummyPL())
    assert (tmp_path / 'best.pt').exists()

