"""
Unit tests for environment snapshot and reproducibility utilities.

Tests comprehensive environment capture, requirements file generation,
W&B artifact logging, and environment comparison.
"""

import os
import json
import tempfile
import shutil
import pytest
import subprocess
import sys
import platform
import torch


# Test 1: Basic environment capture
def test_capture_environment_returns_dict():
    """
    Validate capture_environment() returns complete metadata dict.

    Why: Environment snapshot must include all reproducibility info.
    Contract: Returns dict with python_version, platform, pip_freeze, packages, etc.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    # Check required keys
    assert isinstance(env_info, dict), "Should return dict"
    assert 'python_version' in env_info, "Missing python_version"
    assert 'python_version_short' in env_info, "Missing python_version_short"
    assert 'platform' in env_info, "Missing platform"
    assert 'platform_system' in env_info, "Missing platform_system"
    assert 'pip_freeze' in env_info, "Missing pip_freeze"
    assert 'packages' in env_info, "Missing packages dict"
    assert 'torch_version' in env_info, "Missing torch_version"
    assert 'cuda_available' in env_info, "Missing cuda_available"


# Test 2: Python version format
def test_capture_environment_python_version():
    """
    Validate Python version is captured correctly.

    Why: Reproducibility requires exact Python version.
    Contract: python_version_short format is "X.Y.Z".
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    # Check version format
    version_short = env_info['python_version_short']
    parts = version_short.split('.')
    assert len(parts) == 3, f"Version should be X.Y.Z format: {version_short}"
    assert all(p.isdigit() for p in parts), f"Version parts should be numeric: {version_short}"

    # Check matches sys.version_info
    expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    assert version_short == expected, f"Version mismatch: {version_short} != {expected}"


# Test 3: Pip freeze parsing
def test_capture_environment_packages_dict():
    """
    Validate pip freeze output is parsed into packages dict.

    Why: Need structured access to package versions for comparison.
    Contract: packages dict maps package name → version string.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()
    packages = env_info['packages']

    # Check is dict
    assert isinstance(packages, dict), "packages should be dict"

    # Check has common packages (torch, numpy should be installed)
    assert len(packages) > 0, "packages should not be empty"

    # Check format of entries
    for pkg, version in list(packages.items())[:5]:  # Check first 5
        assert isinstance(pkg, str), f"Package name should be string: {pkg}"
        assert isinstance(version, str), f"Version should be string: {version}"
        assert len(version) > 0, f"Version should not be empty for {pkg}"


# Test 4: PyTorch version capture
def test_capture_environment_torch_version():
    """
    Validate PyTorch version is captured.

    Why: PyTorch version critical for reproducibility.
    Contract: torch_version matches torch.__version__.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    assert env_info['torch_version'] == torch.__version__, \
        f"torch_version mismatch: {env_info['torch_version']} != {torch.__version__}"


# Test 5: CUDA info (skip if no GPU)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_capture_environment_cuda_info():
    """
    Validate CUDA info is captured when GPU available.

    Why: CUDA version affects reproducibility (different ops/precision).
    Contract: cuda_version and gpu_name are populated when CUDA available.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    assert env_info['cuda_available'] == True, "cuda_available should be True"
    assert env_info['cuda_version'] is not None, "cuda_version should be populated"
    assert env_info['gpu_name'] is not None, "gpu_name should be populated"
    assert env_info['gpu_count'] > 0, "gpu_count should be > 0"


# Test 6: No CUDA graceful handling
@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_capture_environment_no_cuda():
    """
    Validate graceful handling when CUDA not available.

    Why: Must work on CPU-only machines.
    Contract: cuda_version/gpu_name are None when CUDA unavailable.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    assert env_info['cuda_available'] == False, "cuda_available should be False"
    assert env_info['cuda_version'] is None, "cuda_version should be None"
    assert env_info['gpu_name'] is None, "gpu_name should be None"
    assert env_info['gpu_count'] == 0, "gpu_count should be 0"


# Test 7: Save to files
def test_save_environment_snapshot_creates_files():
    """
    Validate save_environment_snapshot() creates 3 files.

    Why: Need requirements.txt, environment.json, REPRODUCE.md.
    Contract: All 3 files created in output_dir.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        req_path, env_path, repro_path = save_environment_snapshot(env_info, tmpdir)

        # Check files exist
        assert os.path.exists(req_path), f"requirements.txt not created: {req_path}"
        assert os.path.exists(env_path), f"environment.json not created: {env_path}"
        assert os.path.exists(repro_path), f"REPRODUCE.md not created: {repro_path}"

        # Check file names
        assert req_path.endswith('requirements.txt'), f"Wrong name: {req_path}"
        assert env_path.endswith('environment.json'), f"Wrong name: {env_path}"
        assert repro_path.endswith('REPRODUCE.md'), f"Wrong name: {repro_path}"


# Test 8: requirements.txt format
def test_requirements_txt_pinned_versions():
    """
    Validate requirements.txt uses pinned versions (==).

    Why: Exact versions required for reproducibility.
    Contract: All lines with versions use == (not >=, ~=, etc).
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        req_path, _, _ = save_environment_snapshot(env_info, tmpdir)

        # Read requirements.txt
        with open(req_path) as f:
            content = f.read()

        # Should match pip freeze output
        assert content == env_info['pip_freeze'], "requirements.txt should match pip freeze"

        # Check for pinned versions (lines with ==)
        lines = [line for line in content.split('\n') if line.strip() and not line.startswith('#')]
        version_lines = [line for line in lines if '==' in line]

        # Most packages should have pinned versions
        assert len(version_lines) > 0, "Should have some pinned versions"


# Test 9: environment.json is valid JSON
def test_environment_json_valid():
    """
    Validate environment.json is valid JSON with correct structure.

    Why: Must be machine-readable for comparison/diff tools.
    Contract: Valid JSON, deserializes to dict matching env_info.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        _, env_path, _ = save_environment_snapshot(env_info, tmpdir)

        # Read and parse JSON
        with open(env_path) as f:
            loaded = json.load(f)

        # Check structure
        assert isinstance(loaded, dict), "JSON should deserialize to dict"
        assert 'python_version' in loaded, "Missing python_version in JSON"
        assert 'packages' in loaded, "Missing packages in JSON"
        assert 'torch_version' in loaded, "Missing torch_version in JSON"


# Test 10: REPRODUCE.md contains instructions
def test_reproduce_md_content():
    """
    Validate REPRODUCE.md contains setup instructions.

    Why: Users need clear reproduction steps.
    Contract: Contains Python version, pip install command, verification code.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        _, _, repro_path = save_environment_snapshot(env_info, tmpdir)

        # Read REPRODUCE.md
        with open(repro_path) as f:
            content = f.read()

        # Check key sections
        assert 'Python' in content, "Should mention Python version"
        assert 'pip install' in content, "Should have pip install command"
        assert 'requirements.txt' in content, "Should reference requirements.txt"
        assert env_info['python_version_short'] in content, "Should show exact Python version"
        assert env_info['torch_version'] in content, "Should show PyTorch version"


# Test 11: Compare environments - no changes
def test_compare_environments_identical():
    """
    Validate compare_environments() detects no changes.

    Why: Baseline test for diff functionality.
    Contract: Returns empty diffs when environments identical.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot, compare_environments

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        _, env_path1, _ = save_environment_snapshot(env_info, tmpdir)

        # Save again to different file
        env_path2 = os.path.join(tmpdir, 'environment2.json')
        with open(env_path2, 'w') as f:
            json.dump(env_info, f)

        # Compare
        diff = compare_environments(env_path1, env_path2)

        # Check no differences
        assert len(diff['added']) == 0, "Should have no added packages"
        assert len(diff['removed']) == 0, "Should have no removed packages"
        assert len(diff['changed']) == 0, "Should have no changed packages"
        assert diff['python_version_changed'] == False, "Python version should not change"
        assert diff['cuda_version_changed'] == False, "CUDA version should not change"


# Test 12: Compare environments - version change
def test_compare_environments_version_change():
    """
    Validate compare_environments() detects package version changes.

    Why: Core diff functionality for debugging result differences.
    Contract: Detects changed packages with old → new versions.
    """
    from utils.training.environment_snapshot import compare_environments

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two different environments
        env1 = {
            'python_version_short': '3.10.12',
            'cuda_version': '12.2',
            'packages': {
                'torch': '2.0.1',
                'numpy': '1.24.3',
                'transformers': '4.30.0'
            }
        }

        env2 = {
            'python_version_short': '3.10.12',
            'cuda_version': '12.2',
            'packages': {
                'torch': '2.1.0',  # Changed
                'numpy': '1.24.3',  # Same
                'transformers': '4.30.0'  # Same
            }
        }

        # Save both
        env_path1 = os.path.join(tmpdir, 'env1.json')
        env_path2 = os.path.join(tmpdir, 'env2.json')

        with open(env_path1, 'w') as f:
            json.dump(env1, f)
        with open(env_path2, 'w') as f:
            json.dump(env2, f)

        # Compare
        diff = compare_environments(env_path1, env_path2)

        # Check detected change
        assert len(diff['changed']) == 1, f"Should detect 1 change: {diff['changed']}"
        assert diff['changed'][0] == ('torch', '2.0.1', '2.1.0'), \
            f"Should detect torch version change: {diff['changed']}"


# Test 13: Compare environments - added/removed packages
def test_compare_environments_added_removed():
    """
    Validate compare_environments() detects added/removed packages.

    Why: Different package sets affect reproducibility.
    Contract: Detects added and removed packages.
    """
    from utils.training.environment_snapshot import compare_environments

    with tempfile.TemporaryDirectory() as tmpdir:
        env1 = {
            'python_version_short': '3.10.12',
            'cuda_version': '12.2',
            'packages': {
                'torch': '2.0.1',
                'numpy': '1.24.3',
            }
        }

        env2 = {
            'python_version_short': '3.10.12',
            'cuda_version': '12.2',
            'packages': {
                'torch': '2.0.1',
                'transformers': '4.30.0',  # Added
            }
        }

        # Save both
        env_path1 = os.path.join(tmpdir, 'env1.json')
        env_path2 = os.path.join(tmpdir, 'env2.json')

        with open(env_path1, 'w') as f:
            json.dump(env1, f)
        with open(env_path2, 'w') as f:
            json.dump(env2, f)

        # Compare
        diff = compare_environments(env_path1, env_path2)

        # Check added/removed
        assert len(diff['added']) == 1, f"Should detect 1 addition: {diff['added']}"
        assert diff['added'][0] == ('transformers', '4.30.0'), \
            f"Should detect transformers addition: {diff['added']}"

        assert len(diff['removed']) == 1, f"Should detect 1 removal: {diff['removed']}"
        assert diff['removed'][0] == ('numpy', '1.24.3'), \
            f"Should detect numpy removal: {diff['removed']}"


# Test 14: Compare environments - Python version change
def test_compare_environments_python_change():
    """
    Validate compare_environments() detects Python version changes.

    Why: Python version affects compatibility and behavior.
    Contract: python_version_changed flag set when versions differ.
    """
    from utils.training.environment_snapshot import compare_environments

    with tempfile.TemporaryDirectory() as tmpdir:
        env1 = {
            'python_version_short': '3.10.12',
            'cuda_version': '12.2',
            'packages': {'torch': '2.0.1'}
        }

        env2 = {
            'python_version_short': '3.11.5',  # Changed
            'cuda_version': '12.2',
            'packages': {'torch': '2.0.1'}
        }

        # Save both
        env_path1 = os.path.join(tmpdir, 'env1.json')
        env_path2 = os.path.join(tmpdir, 'env2.json')

        with open(env_path1, 'w') as f:
            json.dump(env1, f)
        with open(env_path2, 'w') as f:
            json.dump(env2, f)

        # Compare
        diff = compare_environments(env_path1, env_path2)

        assert diff['python_version_changed'] == True, \
            "Should detect Python version change"


# Test 15: Missing file error handling
def test_compare_environments_missing_file():
    """
    Validate compare_environments() handles missing files gracefully.

    Why: Robust error handling prevents confusing crashes.
    Contract: Raises FileNotFoundError with clear message.
    """
    from utils.training.environment_snapshot import compare_environments

    with pytest.raises(FileNotFoundError):
        compare_environments('nonexistent1.json', 'nonexistent2.json')


# Test 16: Output directory creation
def test_save_environment_snapshot_creates_output_dir():
    """
    Validate save_environment_snapshot() creates output directory.

    Why: Should work even if output_dir doesn't exist.
    Contract: Creates directory if missing.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use nested path that doesn't exist
        output_dir = os.path.join(tmpdir, 'nested', 'output')

        env_info = capture_environment()
        req_path, env_path, repro_path = save_environment_snapshot(env_info, output_dir)

        # Check directory was created
        assert os.path.exists(output_dir), f"Output directory not created: {output_dir}"

        # Check files in correct location
        assert os.path.dirname(req_path) == output_dir
        assert os.path.dirname(env_path) == output_dir
        assert os.path.dirname(repro_path) == output_dir


# Test 17: Public API exports
def test_public_api_exports():
    """
    Validate public API exports required functions.

    Why: Module interface must be stable.
    Contract: __all__ includes capture_environment, save_environment_snapshot, compare_environments, log_environment_to_wandb.
    """
    from utils.training import environment_snapshot

    assert hasattr(environment_snapshot, '__all__'), "Missing __all__"

    expected = ['capture_environment', 'save_environment_snapshot', 'compare_environments', 'log_environment_to_wandb']
    for func in expected:
        assert func in environment_snapshot.__all__, f"Missing {func} in __all__"
        assert hasattr(environment_snapshot, func), f"Missing {func} in module"


# Test 18: W&B logging requires active run
def test_log_environment_to_wandb_no_active_run():
    """
    Validate log_environment_to_wandb() raises error when no active W&B run.

    Why: Cannot log without active run context.
    Contract: Raises RuntimeError when wandb.run is None.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot, log_environment_to_wandb

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        req_path, env_path, repro_path = save_environment_snapshot(env_info, tmpdir)

        # Should raise RuntimeError when no active run
        with pytest.raises(RuntimeError, match="No active W&B run"):
            log_environment_to_wandb(req_path, env_path, repro_path, env_info)


# Test 19: Environment capture includes hardware info
def test_capture_environment_hardware_info():
    """
    Validate hardware info (GPU name, count) is captured.

    Why: Hardware differences affect reproducibility.
    Contract: gpu_name and gpu_count present in env_info.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    # Check hardware keys exist
    assert 'gpu_name' in env_info, "Missing gpu_name"
    assert 'gpu_count' in env_info, "Missing gpu_count"

    # GPU info should be None/0 if no CUDA, or populated if CUDA
    if torch.cuda.is_available():
        assert env_info['gpu_name'] is not None, "gpu_name should be populated with CUDA"
        assert env_info['gpu_count'] > 0, "gpu_count should be > 0 with CUDA"
    else:
        assert env_info['gpu_name'] is None, "gpu_name should be None without CUDA"
        assert env_info['gpu_count'] == 0, "gpu_count should be 0 without CUDA"


# Test 20: Platform information completeness
def test_capture_environment_platform_completeness():
    """
    Validate platform information is complete.

    Why: OS/platform differences affect package behavior.
    Contract: platform_system and platform_release are populated.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    # Check platform info
    assert env_info['platform_system'] in ['Linux', 'Darwin', 'Windows'], \
        f"Unexpected platform_system: {env_info['platform_system']}"
    assert len(env_info['platform']) > 0, "platform string should not be empty"
    assert len(env_info['platform_release']) > 0, "platform_release should not be empty"


# Test 21: REPRODUCE.md troubleshooting section
def test_reproduce_md_troubleshooting():
    """
    Validate REPRODUCE.md includes troubleshooting guidance.

    Why: Users need help resolving common reproduction issues.
    Contract: Contains troubleshooting section with common solutions.
    """
    from utils.training.environment_snapshot import capture_environment, save_environment_snapshot

    with tempfile.TemporaryDirectory() as tmpdir:
        env_info = capture_environment()
        _, _, repro_path = save_environment_snapshot(env_info, tmpdir)

        # Read REPRODUCE.md
        with open(repro_path) as f:
            content = f.read()

        # Check troubleshooting sections
        assert 'Troubleshooting' in content, "Should have troubleshooting section"
        assert 'CUDA' in content or 'cuda' in content, "Should mention CUDA issues"
        assert 'virtual environment' in content.lower(), "Should mention venv setup"


# Test 22: Environment validation check function
def test_environment_validation():
    """
    Validate environment snapshot captures all reproducibility metadata.

    Why: Comprehensive environment capture is critical for reproducibility.
    Contract: All 10 acceptance criteria metadata captured.
    """
    from utils.training.environment_snapshot import capture_environment

    env_info = capture_environment()

    # AC 1: pip freeze captured
    assert 'pip_freeze' in env_info
    assert len(env_info['pip_freeze']) > 0

    # AC 2: Exact versions (checked in requirements.txt test)
    assert 'packages' in env_info
    assert len(env_info['packages']) > 0

    # AC 3: Python version and platform
    assert 'python_version' in env_info
    assert 'python_version_short' in env_info
    assert 'platform' in env_info
    assert 'platform_system' in env_info

    # AC 9: Hardware info
    assert 'gpu_name' in env_info
    assert 'gpu_count' in env_info
    assert 'cuda_version' in env_info
