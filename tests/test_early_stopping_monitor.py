"""
Unit tests for EarlyStoppingMonitor and W&B logging callback (without Lightning).

We simulate validation epochs and verify that the monitor triggers after
the configured patience and that the callback attempts to log to W&B.
"""

import sys
import os
import types


def test_monitor_improvement_and_trigger():
    # Ensure repo root on path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Import module directly by path to avoid heavy package imports
    import importlib.util
    du_path = os.path.join(repo_root, 'utils', 'training', 'early_stopping.py')
    spec = importlib.util.spec_from_file_location('early_stopping', du_path)
    es = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(es)  # type: ignore
    EarlyStoppingMonitor = es.EarlyStoppingMonitor

    m = EarlyStoppingMonitor(patience=3, min_delta=0.1, mode='min')

    # Initial improvement
    improved, triggered = m.update(1.0)
    assert improved is True
    assert triggered is False

    # No improvement within min_delta
    improved, triggered = m.update(0.95)
    assert improved is False
    assert triggered is False
    assert m.epochs_without_improvement == 1

    # Still no improvement
    improved, triggered = m.update(0.92)
    assert improved is False
    assert triggered is False
    assert m.epochs_without_improvement == 2

    # Exceed patience
    improved, triggered = m.update(0.91)
    assert improved is False
    assert triggered is True


def test_wandb_callback_logs_event_when_triggered(monkeypatch):
    # Stub wandb
    wandb = types.ModuleType('wandb')
    class Run:
        pass
    wandb.run = Run()
    logged = {}
    def _log(data, step=None):
        logged['data'] = data
        logged['step'] = step
    wandb.log = _log
    sys.modules['wandb'] = wandb

    # Import callback
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib.util
    du_path = os.path.join(repo_root, 'utils', 'training', 'early_stopping.py')
    spec = importlib.util.spec_from_file_location('early_stopping', du_path)
    es = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(es)  # type: ignore
    EarlyStoppingWandbCallback = es.EarlyStoppingWandbCallback

    cb = EarlyStoppingWandbCallback(patience=2, min_delta=0.0, mode='min')

    class DummyTrainer:
        def __init__(self):
            self.current_epoch = 2
            self.callback_metrics = {'val_loss': 1.0}

    t = DummyTrainer()

    # First call: improvement from inf to 1.0
    cb.on_validation_end(t, None)
    assert 'data' not in logged

    # No improvement (patience 2)
    t.callback_metrics = {'val_loss': 1.0}
    cb.on_validation_end(t, None)
    assert 'data' not in logged

    # Exceed patience → should log
    cb.on_validation_end(t, None)
    assert 'data' in logged
    assert 'events/early_stopping_triggered' in logged['data']
    assert logged['data']['events/early_stopping_triggered'] == 1
