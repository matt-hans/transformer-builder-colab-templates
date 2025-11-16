import os, sys
from pathlib import Path
import importlib.util, types


def test_detect_resume_prefers_best_ckpt(tmp_path):
    # Create fake files
    (tmp_path / 'last.ckpt').write_bytes(b'x')
    (tmp_path / 'best.ckpt').write_bytes(b'y')
    (tmp_path / 'epoch_1.pt').write_bytes(b'z')

    # Load module
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cm_path = os.path.join(repo_root, 'utils', 'training', 'checkpoint_manager.py')
    # Stub torch to import module without heavy deps
    sys.modules['torch'] = types.ModuleType('torch')
    spec = importlib.util.spec_from_file_location('checkpoint_manager', cm_path)
    cm = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cm)  # type: ignore

    info = cm.detect_resume_checkpoint(str(tmp_path), prefer='best')
    assert info['type'] == 'lightning'
    assert info['path'].endswith('best.ckpt')
