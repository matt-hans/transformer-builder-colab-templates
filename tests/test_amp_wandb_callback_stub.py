import os
import sys
import types


def test_amp_wandb_callback_logs_loss_scale_and_flags(monkeypatch):
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

    # Import callback directly
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib.util
    mod_path = os.path.join(repo_root, 'utils', 'training', 'amp_utils.py')
    spec = importlib.util.spec_from_file_location('amp_utils', mod_path)
    au = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(au)  # type: ignore

    AmpWandbCallback = au.AmpWandbCallback

    # Create a dummy trainer exposing strategy.precision_plugin.scaler.get_scale
    class DummyScaler:
        def get_scale(self):
            return 1024.0

    class DummyPrecisionPlugin:
        def __init__(self):
            self.scaler = DummyScaler()

    class DummyStrategy:
        def __init__(self):
            self.precision_plugin = DummyPrecisionPlugin()

    class DummyTrainer:
        def __init__(self):
            self.strategy = DummyStrategy()
            self.current_epoch = 3

    trainer = DummyTrainer()
    cb = AmpWandbCallback(enabled=True, precision='16')

    cb.on_train_epoch_end(trainer, None)

    assert 'data' in logged
    data = logged['data']
    assert data.get('amp/enabled') == 1
    assert data.get('amp/precision') == '16'
    assert data.get('amp/loss_scale') == 1024.0


def test_amp_wandb_callback_handles_missing_scaler(monkeypatch):
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

    # Import callback directly
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib.util
    mod_path = os.path.join(repo_root, 'utils', 'training', 'amp_utils.py')
    spec = importlib.util.spec_from_file_location('amp_utils2', mod_path)
    au = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(au)  # type: ignore

    AmpWandbCallback = au.AmpWandbCallback

    class DummyPrecisionPlugin:
        scaler = None

    class DummyStrategy:
        precision_plugin = DummyPrecisionPlugin()

    class DummyTrainer:
        strategy = DummyStrategy()
        current_epoch = 0

    trainer = DummyTrainer()
    cb = AmpWandbCallback(enabled=True, precision='16')

    cb.on_train_epoch_end(trainer, None)

    assert 'data' in logged
    data = logged['data']
    assert data.get('amp/enabled') == 1
    assert data.get('amp/precision') == '16'
    assert 'amp/loss_scale' not in data

