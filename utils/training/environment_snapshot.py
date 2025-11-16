"""
Environment snapshot and reproducibility utilities.

Captures complete Python environment (pip freeze) at training time to enable
exact environment recreation for reproducing results months or years later.

Key Features:
- Capture pip freeze output with all package versions
- Save requirements.txt with pinned versions (==)
- Include Python version, platform, CUDA info
- Log environment to W&B artifacts
- Environment diff comparison utility
- Auto-generate reproduction instructions

Usage:
    >>> # At training start
    >>> env_info = capture_environment()
    >>> req_path, env_path, repro_path = save_environment_snapshot(env_info, "./environment")
    >>>
    >>> # Log to W&B
    >>> if wandb.run:
    ...     log_environment_to_wandb(req_path, env_path, repro_path, env_info)
    >>>
    >>> # Compare two environments
    >>> diff = compare_environments('env1.json', 'env2.json')
"""

import os
import sys
import platform
import subprocess
import json
import logging
from typing import Dict, Tuple, Any

# Initialize logger
logger = logging.getLogger(__name__)


def capture_environment() -> Dict[str, Any]:
    """
    Capture complete Python environment snapshot.

    Collects:
    - Python version (full and short X.Y.Z format)
    - Platform information (OS, architecture)
    - pip freeze output (all installed packages)
    - Parsed packages dict (package → version mapping)
    - PyTorch version
    - CUDA availability and version (if GPU available)
    - GPU hardware info (name, count)

    Returns:
        Dict containing environment metadata:
        - python_version: Full Python version string
        - python_version_short: "X.Y.Z" format
        - platform: Full platform string
        - platform_system: OS name (Linux, Darwin, Windows)
        - platform_release: OS release version
        - pip_freeze: Raw pip freeze output
        - packages: Dict mapping package names to versions
        - torch_version: PyTorch version string
        - cuda_available: Whether CUDA is available
        - cuda_version: CUDA version string (None if unavailable)
        - cudnn_version: cuDNN version (None if unavailable)
        - gpu_name: GPU device name (None if unavailable)
        - gpu_count: Number of GPUs (0 if unavailable)

    Example:
        >>> env_info = capture_environment()
        >>> print(f"Python {env_info['python_version_short']}")
        Python 3.10.12
        >>> print(f"PyTorch {env_info['torch_version']}")
        PyTorch 2.1.0+cu121
        >>> print(f"Packages: {len(env_info['packages'])}")
        Packages: 127

    Note:
        - Requires pip to be available in PATH
        - CUDA info only populated if torch.cuda.is_available()
        - GPU name requires CUDA-enabled PyTorch
    """
    import torch  # Import here to capture version

    # Get pip freeze output with error handling
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'],
            timeout=30,
            stderr=subprocess.STDOUT
        ).decode('utf-8')
    except subprocess.CalledProcessError as e:
        logger.error(f"pip freeze failed with exit code {e.returncode}: {e.output}")
        pip_freeze = ""  # Empty string as fallback
    except subprocess.TimeoutExpired:
        logger.error("pip freeze command timed out after 30 seconds")
        pip_freeze = ""  # Empty string as fallback
    except Exception as e:
        logger.error(f"Unexpected error running pip freeze: {e}")
        pip_freeze = ""  # Empty string as fallback

    # Parse pip freeze into dict
    # Format: package==version or package @ git+https://...
    packages = {}
    for line in pip_freeze.strip().split('\n'):
        if '==' in line:
            # Standard versioned package: numpy==1.24.3
            pkg, version = line.split('==', 1)
            packages[pkg] = version
        elif ' @ ' in line:
            # Git/URL package: package @ git+https://...
            pkg = line.split(' @ ')[0]
            packages[pkg] = 'git+url'  # Mark as git install

    # Collect environment info
    env_info = {
        # Python version
        'python_version': sys.version,
        'python_version_short': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",

        # Platform
        'platform': platform.platform(),
        'platform_system': platform.system(),
        'platform_release': platform.release(),

        # Package data
        'pip_freeze': pip_freeze,
        'packages': packages,

        # PyTorch version
        'torch_version': torch.__version__,

        # CUDA info (populated if available)
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,

        # Hardware info
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    return env_info


def save_environment_snapshot(
    env_info: Dict[str, Any],
    output_dir: str = "./"
) -> Tuple[str, str, str]:
    """
    Save environment snapshot to files.

    Creates three files:
    1. requirements.txt - pip freeze format for pip install
    2. environment.json - Full metadata for programmatic access
    3. REPRODUCE.md - Human-readable reproduction guide

    Args:
        env_info: Environment dict from capture_environment()
        output_dir: Directory to save files (created if missing)

    Returns:
        Tuple of (requirements_path, environment_json_path, reproduce_md_path)

    Example:
        >>> env_info = capture_environment()
        >>> req_path, env_path, repro_path = save_environment_snapshot(
        ...     env_info,
        ...     "./environment"
        ... )
        ✅ Environment snapshot saved:
           - ./environment/requirements.txt
           - ./environment/environment.json
           - ./environment/REPRODUCE.md

    Side Effects:
        - Creates output_dir if it doesn't exist
        - Writes 3 files to disk
        - Prints confirmation message
    """
    # Create output directory if needed
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise

    # Save requirements.txt (pip freeze format)
    requirements_path = os.path.join(output_dir, "requirements.txt")
    try:
        with open(requirements_path, 'w') as f:
            f.write(env_info['pip_freeze'])
    except IOError as e:
        logger.error(f"Failed to write requirements.txt: {e}")
        raise

    # Save full environment info as JSON
    env_json_path = os.path.join(output_dir, "environment.json")
    try:
        with open(env_json_path, 'w') as f:
            json.dump(env_info, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to write environment.json: {e}")
        raise
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize environment data to JSON: {e}")
        raise

    # Save reproduction instructions
    repro_path = os.path.join(output_dir, "REPRODUCE.md")
    try:
        _write_reproduction_guide(env_info, repro_path)
    except IOError as e:
        logger.error(f"Failed to write REPRODUCE.md: {e}")
        raise

    # Log success with both logger and user-facing print
    logger.info(f"Environment snapshot saved to {output_dir}")
    logger.debug("Files created: requirements.txt, environment.json, REPRODUCE.md")

    print(f"✅ Environment snapshot saved:")
    print(f"   - {requirements_path}")
    print(f"   - {env_json_path}")
    print(f"   - {repro_path}")

    return requirements_path, env_json_path, repro_path


def _write_reproduction_guide(env_info: Dict[str, Any], output_path: str) -> None:
    """
    Write REPRODUCE.md with setup instructions.

    Internal helper for save_environment_snapshot().

    Args:
        env_info: Environment metadata dict
        output_path: Path to write REPRODUCE.md

    Side Effects:
        - Writes REPRODUCE.md file
    """
    # Extract key package versions
    transformers_version = env_info['packages'].get('transformers', 'N/A')
    numpy_version = env_info['packages'].get('numpy', 'N/A')

    content = f"""# Environment Reproduction Guide

## Quick Setup

```bash
# Python version required
python --version  # Should be {env_info['python_version_short']}

# Install exact package versions
pip install -r requirements.txt
```

## System Information

- **Python**: {env_info['python_version_short']}
- **Platform**: {env_info['platform']}
- **PyTorch**: {env_info['torch_version']}
- **CUDA**: {env_info['cuda_version'] or 'N/A (CPU only)'}
- **GPU**: {env_info['gpu_name'] or 'N/A'}

## Key Package Versions

- torch=={env_info['torch_version']}
- transformers=={transformers_version}
- numpy=={numpy_version}

## Verification

After installation, verify with:

```python
import torch
print(f"PyTorch: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
```

Expected output:
```
PyTorch: {env_info['torch_version']}
CUDA available: {env_info['cuda_available']}
```

## Notes

- If using Google Colab, GPU type may differ (T4 vs A100 vs V100)
- Some CUDA operations are non-deterministic even with same seed
- Use deterministic mode for bit-exact reproduction (slower):
  ```python
  from utils.training.seed_manager import set_random_seed
  set_random_seed(42, deterministic=True)
  ```

## Troubleshooting

### Different Python version
If you have Python {env_info['python_version_short'].rsplit('.', 1)[0]}.X instead of {env_info['python_version_short']}:
- Minor version differences (3.10.X) usually work
- Major version differences (3.9 vs 3.10) may cause issues

### CUDA version mismatch
If you get CUDA errors:
- Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/
- Or use CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Package conflicts
If pip install fails with conflicts:
1. Create fresh virtual environment: `python -m venv .venv`
2. Activate: `source .venv/bin/activate` (Linux/Mac) or `.venv\\Scripts\\activate` (Windows)
3. Install: `pip install -r requirements.txt`
"""

    try:
        with open(output_path, 'w') as f:
            f.write(content)
    except IOError as e:
        logger.error(f"Failed to write reproduction guide to {output_path}: {e}")
        raise


def compare_environments(env1_path: str, env2_path: str) -> Dict[str, Any]:
    """
    Compare two environment snapshots and show differences.

    Identifies:
    - Added packages (in env2 but not env1)
    - Removed packages (in env1 but not env2)
    - Changed package versions
    - Python version changes
    - CUDA version changes

    Args:
        env1_path: Path to first environment.json
        env2_path: Path to second environment.json

    Returns:
        Dict with diff information:
        - added: List of (package, version) tuples for new packages
        - removed: List of (package, version) tuples for removed packages
        - changed: List of (package, old_version, new_version) tuples
        - python_version_changed: Bool indicating Python version change
        - cuda_version_changed: Bool indicating CUDA version change

    Raises:
        FileNotFoundError: If either environment file doesn't exist

    Example:
        >>> diff = compare_environments('run1/environment.json', 'run2/environment.json')
        🔍 Environment Differences:

          Python: 3.10.12 → 3.10.12
          CUDA: 12.2 → 12.2

          📦 Changed packages (1):
            - torch: 2.0.1 → 2.1.0

    Side Effects:
        - Prints diff summary to stdout
    """
    # Load both environments with error handling
    try:
        with open(env1_path) as f:
            env1 = json.load(f)
    except FileNotFoundError:
        logger.error(f"Environment file not found: {env1_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted JSON in {env1_path}: {e}")
        raise ValueError(f"Invalid JSON in {env1_path}: {e}")
    except IOError as e:
        logger.error(f"Failed to read {env1_path}: {e}")
        raise

    try:
        with open(env2_path) as f:
            env2 = json.load(f)
    except FileNotFoundError:
        logger.error(f"Environment file not found: {env2_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted JSON in {env2_path}: {e}")
        raise ValueError(f"Invalid JSON in {env2_path}: {e}")
    except IOError as e:
        logger.error(f"Failed to read {env2_path}: {e}")
        raise

    # Compare package versions
    packages1 = env1['packages']
    packages2 = env2['packages']

    all_packages = set(packages1.keys()) | set(packages2.keys())

    differences = {
        'added': [],
        'removed': [],
        'changed': [],
        'python_version_changed': env1['python_version_short'] != env2['python_version_short'],
        'cuda_version_changed': env1.get('cuda_version') != env2.get('cuda_version'),
    }

    for pkg in sorted(all_packages):
        v1 = packages1.get(pkg)
        v2 = packages2.get(pkg)

        if v1 is None:
            # Package added in env2
            differences['added'].append((pkg, v2))
        elif v2 is None:
            # Package removed in env2
            differences['removed'].append((pkg, v1))
        elif v1 != v2:
            # Version changed
            differences['changed'].append((pkg, v1, v2))

    # Log comparison
    logger.info(f"Comparing environments: {env1_path} vs {env2_path}")
    logger.debug(f"Python: {env1['python_version_short']} → {env2['python_version_short']}")
    logger.debug(f"CUDA: {env1.get('cuda_version', 'N/A')} → {env2.get('cuda_version', 'N/A')}")

    # Print summary
    print("🔍 Environment Differences:")
    print(f"\n  Python: {env1['python_version_short']} → {env2['python_version_short']}")
    print(f"  CUDA: {env1.get('cuda_version', 'N/A')} → {env2.get('cuda_version', 'N/A')}")

    if differences['changed']:
        print(f"\n  📦 Changed packages ({len(differences['changed'])}):")
        for pkg, v1, v2 in differences['changed'][:10]:  # Show first 10
            print(f"    - {pkg}: {v1} → {v2}")
        if len(differences['changed']) > 10:
            print(f"    ... and {len(differences['changed']) - 10} more")

    if differences['added']:
        print(f"\n  ➕ Added packages: {len(differences['added'])}")
        for pkg, v2 in differences['added'][:5]:  # Show first 5
            print(f"    - {pkg}=={v2}")
        if len(differences['added']) > 5:
            print(f"    ... and {len(differences['added']) - 5} more")

    if differences['removed']:
        print(f"\n  ➖ Removed packages: {len(differences['removed'])}")
        for pkg, v1 in differences['removed'][:5]:  # Show first 5
            print(f"    - {pkg}=={v1}")
        if len(differences['removed']) > 5:
            print(f"    ... and {len(differences['removed']) - 5} more")

    if not any([differences['changed'], differences['added'], differences['removed'],
                differences['python_version_changed'], differences['cuda_version_changed']]):
        print("\n  ✅ Environments are identical")

    return differences


def log_environment_to_wandb(
    requirements_path: str,
    env_json_path: str,
    reproduce_path: str,
    env_info: Dict[str, Any]
) -> None:
    """
    Log environment snapshot to W&B as artifact.

    Uploads the three environment files (requirements.txt, environment.json,
    REPRODUCE.md) as a W&B artifact and logs key versions to run config.

    Args:
        requirements_path: Path to requirements.txt
        env_json_path: Path to environment.json
        reproduce_path: Path to REPRODUCE.md
        env_info: Environment metadata dict from capture_environment()

    Example:
        >>> import wandb
        >>> wandb.init(project="my-project", name="experiment-1")
        >>> env_info = capture_environment()
        >>> req_path, env_path, repro_path = save_environment_snapshot(env_info)
        >>> log_environment_to_wandb(req_path, env_path, repro_path, env_info)
        ✅ Environment logged to W&B

    Side Effects:
        - Creates W&B artifact with environment files
        - Updates W&B run config with key versions
        - Prints confirmation message

    Raises:
        ImportError: If wandb not installed
        RuntimeError: If wandb.run is None (no active run)
    """
    try:
        import wandb
    except ImportError as e:
        logger.error(f"Failed to import wandb: {e}")
        raise ImportError(
            "wandb not installed. Install with: pip install wandb"
        ) from e

    if wandb.run is None:
        logger.error("Attempted to log environment but no active W&B run found")
        raise RuntimeError(
            "No active W&B run. Call wandb.init() before logging environment"
        )

    # Create artifact with error handling
    try:
        env_artifact = wandb.Artifact(
            name=f"{wandb.run.name}-environment",
            type="environment",
            description="Python environment snapshot for reproducibility",
            metadata={
                'python_version': env_info['python_version_short'],
                'torch_version': env_info['torch_version'],
                'cuda_version': env_info['cuda_version'],
                'gpu_name': env_info['gpu_name'],
                'platform': env_info['platform_system'],
            }
        )

        # Add files
        env_artifact.add_file(requirements_path)
        env_artifact.add_file(env_json_path)
        env_artifact.add_file(reproduce_path)

        # Log artifact
        wandb.log_artifact(env_artifact)
        logger.info(f"Environment artifact '{env_artifact.name}' logged to W&B")

        # Also log key versions to config for easy filtering
        wandb.config.update({
            'python_version': env_info['python_version_short'],
            'torch_version': env_info['torch_version'],
            'cuda_version': env_info['cuda_version'],
            'platform': env_info['platform_system'],
        }, allow_val_change=True)
        logger.info("Environment versions added to W&B config")

        print("✅ Environment logged to W&B")

    except Exception as e:
        logger.error(f"Failed to log environment to W&B: {e}", exc_info=True)
        raise RuntimeError(f"W&B artifact logging failed: {e}") from e


# Public API
__all__ = [
    'capture_environment',
    'save_environment_snapshot',
    'compare_environments',
    'log_environment_to_wandb',
]
