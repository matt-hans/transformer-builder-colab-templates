import os
import sys
import types


def test_compute_effective_precision_cpu_behavior():
    # Import amp_utils directly
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib.util
    mod_path = os.path.join(repo_root, 'utils', 'training', 'amp_utils.py')
    spec = importlib.util.spec_from_file_location('amp_utils_mod', mod_path)
    au = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(au)  # type: ignore

    compute = au.compute_effective_precision

    # No override → keep requested
    assert compute('16', None, False, False) == '16'
    # CPU only → force 32
    assert compute('16', True, False, True) == '32'
    assert compute('bf16', True, False, True) == '32'
    # AMP disabled → 32 regardless of requested
    assert compute('16', False, True, True) == '32'

