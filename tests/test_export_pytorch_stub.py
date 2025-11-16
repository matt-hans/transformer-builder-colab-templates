"""
Stubbed test for export_state_dict without real torch.

We inject a fake torch module with save() to verify files are written and
metadata/config are saved. This avoids heavy framework deps.
"""

import os
import json
import shutil
import sys
import types
import importlib.util
from pathlib import Path


def test_export_state_dict_stub_tmp(tmp_path):
    # Create a stub torch with save and __version__
    torch = types.ModuleType('torch')
    saved = {}
    def _save(obj, path):
        # write a simple file to simulate save
        Path(path).write_bytes(b'ptstate')
        saved['path'] = str(path)
    torch.save = _save
    torch.__version__ = '0.0.0-stub'
    torch.nn = types.ModuleType('torch.nn')
    setattr(torch.nn, 'Module', object)
    # Stub torch.utils.data to satisfy import
    # Create module hierarchy torch.utils.data
    tu = types.ModuleType('torch.utils')
    tud = types.ModuleType('torch.utils.data')
    setattr(tud, 'DataLoader', object)
    sys.modules['torch.utils'] = tu
    sys.modules['torch.utils.data'] = tud
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn

    # Load export_utilities via file path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eu_path = os.path.join(repo_root, 'utils', 'training', 'export_utilities.py')
    spec = importlib.util.spec_from_file_location('export_utilities', eu_path)
    eu = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(eu)  # type: ignore

    # Dummy model with state_dict
    class DummyModel:
        def __init__(self):
            self._state = {'w': [1, 2, 3]}
        def state_dict(self):
            return self._state
        def parameters(self):
            class P:
                def numel(self):
                    return 42
            return [P()]

    class DummyTok:
        def __init__(self):
            self.saved = False
        def save_pretrained(self, d):
            Path(os.path.join(d, 'tokenizer.json')).write_text('{}')
            self.saved = True

    out_dir = tmp_path / 'exported'
    cfg = {'vocab_size': 10}
    tok = DummyTok()
    export_path = eu.export_state_dict(DummyModel(), output_dir=str(out_dir), config=cfg, tokenizer=tok, metrics={'val_loss': 1.23})
    assert os.path.isdir(export_path)
    # Files exist
    assert (out_dir / 'pytorch_model.bin').exists()
    assert (out_dir / 'config.json').exists()
    assert (out_dir / 'metadata.json').exists()
    # Validate metadata content
    meta = json.load(open(out_dir / 'metadata.json'))
    assert meta['final_metrics']['val_loss'] == 1.23
