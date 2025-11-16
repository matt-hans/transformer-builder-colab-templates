import os
import sys
import types
import importlib.util
from pathlib import Path


def test_resume_from_state_dict_latest(tmp_path):
    # Stub torch for import of checkpoint_manager helpers
    tmod = types.ModuleType('torch')
    # Provide a fake load that returns expected fields
    def _load(path, map_location=None):
        return {'model_state_dict': {}, 'epoch': 3, 'metrics': {'val_loss': 1.0}, 'config': {'random_seed': 42}}
    tmod.load = _load
    sys.modules['torch'] = tmod

    # Create a fake state_dict checkpoint metadata
    ckpt_dir = tmp_path
    (ckpt_dir / 'epoch_3.pt').write_bytes(b'fake')
    meta_path = ckpt_dir / 'epoch_3.json'
    meta_path.write_text('{"epoch": 3, "metrics": {"val_loss": 1.0}, "config": {"random_seed": 42}}')

    # Ensure package import path and import resume utils
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Import resume utils by file path (avoids full utils package import)
    ru_path = os.path.join(repo_root, 'utils', 'training', 'resume_utils.py')
    spec = importlib.util.spec_from_file_location('resume_utils', ru_path)
    ru = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ru)  # type: ignore

    # Dummy model/optimizer
    class M:
        def state_dict(self):
            return {}
        def load_state_dict(self, s):
            pass
        def parameters(self):
            return []
    info = ru.resume_training_from_checkpoint(str(ckpt_dir), model=M())
    assert info['start_epoch'] == 4
    assert info['metrics']['val_loss'] == 1.0
