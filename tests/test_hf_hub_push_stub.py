"""
Stubbed tests for HF Hub push: verifies graceful fallback and API calls.
"""

import os
import sys
import types
import importlib.util
from pathlib import Path


def test_push_model_to_hub_fallback_no_hub(tmp_path, capsys):
    # Ensure huggingface_hub is missing
    if 'huggingface_hub' in sys.modules:
        sys.modules.pop('huggingface_hub')

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mod_path = os.path.join(repo_root, 'utils', 'training', 'hf_hub.py')
    spec = importlib.util.spec_from_file_location('hf_hub', mod_path)
    hub = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hub)  # type: ignore

    class M:
        def __init__(self):
            self._state = {}
        def state_dict(self):
            return self._state
        def parameters(self):
            return []

    # Stub torch.save to write a file
    t = types.ModuleType('torch')
    def _save(obj, path):
        Path(path).write_bytes(b'bin')
    t.save = _save
    sys.modules['torch'] = t

    url = hub.push_model_to_hub(M(), {'vocab_size': 10}, {'val_loss': 1.0}, 'user/repo', private=True, local_dir=str(tmp_path))
    out = capsys.readouterr()
    assert url is None
    assert (tmp_path / 'pytorch_model.bin').exists()
    assert (tmp_path / 'config.json').exists()
    assert (tmp_path / 'README.md').exists()


def test_push_model_to_hub_calls_api(tmp_path):
    # Stub huggingface_hub API
    hub_mod = types.ModuleType('huggingface_hub')
    calls = {}
    class _API:
        def upload_folder(self, folder_path, repo_id, commit_message):
            calls['folder_path'] = folder_path
            calls['repo_id'] = repo_id
            calls['commit_message'] = commit_message
    def create_repo(repo_id, private=False, exist_ok=True):
        calls['create_repo'] = (repo_id, private, exist_ok)
    hub_mod.HfApi = _API
    hub_mod.create_repo = create_repo
    sys.modules['huggingface_hub'] = hub_mod

    # Load module
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mod_path = os.path.join(repo_root, 'utils', 'training', 'hf_hub.py')
    spec = importlib.util.spec_from_file_location('hf_hub', mod_path)
    hub = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hub)  # type: ignore

    # Stub torch.save
    t = types.ModuleType('torch')
    def _save(obj, path):
        Path(path).write_bytes(b'bin')
    t.save = _save
    sys.modules['torch'] = t

    class M:
        def state_dict(self):
            return {}
        def parameters(self):
            return []

    url = hub.push_model_to_hub(M(), {'vocab_size': 10}, {'val_loss': 1.0}, 'user/repo', private=False, local_dir=str(tmp_path))
    assert url.endswith('user/repo')
    assert calls['create_repo'][0] == 'user/repo'
    assert calls['repo_id'] == 'user/repo'
    assert Path(calls['folder_path']).exists()

