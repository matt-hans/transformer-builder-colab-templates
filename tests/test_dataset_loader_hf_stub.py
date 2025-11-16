"""
Unit test for DatasetLoader.load_huggingface using a stubbed `datasets` module.

This avoids network and external deps while verifying that cache_dir and
core arguments are passed correctly to load_dataset, and that a Dataset-like
object is returned.
"""

import sys
import os
import types


def test_load_huggingface_uses_cache_dir_and_args():
    # Create a stub `datasets` module with Dataset, DatasetDict, load_dataset
    calls = {}

    class DummyDataset:
        def __init__(self, n=10):
            self._n = n
        def __len__(self):
            return self._n

    class DummyDatasetDict(dict):
        pass

    def load_dataset(name, config, split=None, streaming=False, trust_remote_code=False, cache_dir=None):
        calls['name'] = name
        calls['config'] = config
        calls['split'] = split
        calls['streaming'] = streaming
        calls['trust_remote_code'] = trust_remote_code
        calls['cache_dir'] = cache_dir
        return DummyDataset(7)

    stub = types.ModuleType('datasets')
    stub.Dataset = DummyDataset
    stub.DatasetDict = DummyDatasetDict
    stub.load_dataset = load_dataset

    sys.modules['datasets'] = stub
    # Stub pandas to avoid import-time failure
    pd_stub = types.ModuleType('pandas')
    def _no_read_csv(*args, **kwargs):
        raise RuntimeError('read_csv should not be called in this test')
    pd_stub.read_csv = _no_read_csv
    sys.modules['pandas'] = pd_stub
    # Stub tqdm
    tqdm_auto = types.ModuleType('tqdm.auto')
    def _tqdm(*args, **kwargs):
        class _Dummy:
            def __iter__(self):
                return iter([])
        return _Dummy()
    tqdm_auto.tqdm = _tqdm
    sys.modules['tqdm'] = types.ModuleType('tqdm')
    sys.modules['tqdm.auto'] = tqdm_auto

    # Now import the module under test (binds to our stub)
    # Ensure repo root on path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Insert a dummy 'torch' to satisfy package imports pulled by utils.__init__
    dummy_torch = types.ModuleType('torch')
    sys.modules['torch'] = dummy_torch

    # Import dataset_utilities module directly by path to avoid utils/__init__ side effects
    import importlib.util
    du_path = os.path.join(repo_root, 'utils', 'training', 'dataset_utilities.py')
    spec = importlib.util.spec_from_file_location('dataset_utilities', du_path)
    du = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(du)  # type: ignore

    loader = du.DatasetLoader(cache_dir='/tmp/mycache')
    ds = loader.load_huggingface(
        dataset_name='wikitext',
        config_name='wikitext-2-raw-v1',
        split='train',
        streaming=False,
        trust_remote_code=False
    )

    # Verify return type and captured args
    assert isinstance(ds, du.Dataset)
    assert len(ds) == 7
    assert calls['name'] == 'wikitext'
    assert calls['config'] == 'wikitext-2-raw-v1'
    assert calls['split'] == 'train'
    assert calls['streaming'] is False
    assert calls['trust_remote_code'] is False
    assert calls['cache_dir'] == '/tmp/mycache'
