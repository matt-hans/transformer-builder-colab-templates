#!/usr/bin/env python3
"""
Common environment setup utilities for Google Colab.

Handles dependency installation, environment configuration,
and workspace initialization.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def install_packages(packages: List[str], quiet: bool = True):
    """
    Install Python packages using pip.

    Args:
        packages: List of package specifications (e.g., ['torch==2.0.0', 'numpy'])
        quiet: Suppress installation output
    """
    cmd = [sys.executable, '-m', 'pip', 'install']

    if quiet:
        cmd.append('-q')

    cmd.extend(packages)

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Installed: {', '.join(packages)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        raise


def install_apt_packages(packages: List[str], quiet: bool = True):
    """
    Install system packages using apt.

    Args:
        packages: List of package names
        quiet: Suppress installation output
    """
    cmd = ['apt', '-y', 'install']

    if quiet:
        cmd.append('-qq')

    cmd.extend(packages)

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Installed (apt): {', '.join(packages)}")
    except subprocess.CalledProcessError as e:
        print(f"❌ apt installation failed: {e}")
        raise


def setup_pytorch(version: str = 'latest', cuda_version: Optional[str] = None):
    """
    Install PyTorch with CUDA support.

    Args:
        version: PyTorch version ('latest', '2.0.0', etc.)
        cuda_version: CUDA version ('11.8', '12.1', etc.)
    """
    if version == 'latest':
        packages = ['torch', 'torchvision', 'torchaudio']
    else:
        packages = [f'torch=={version}', f'torchvision', f'torchaudio']

    # Add CUDA-specific index URL if specified
    if cuda_version:
        cuda_tag = cuda_version.replace('.', '')
        index_url = f'--index-url https://download.pytorch.org/whl/cu{cuda_tag}'
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages + [index_url]
        subprocess.run(cmd, check=True)
    else:
        install_packages(packages)


def clone_repository(repo_url: str, dest_dir: Optional[Path] = None, branch: Optional[str] = None):
    """
    Clone a Git repository.

    Args:
        repo_url: Repository URL
        dest_dir: Destination directory (defaults to repo name)
        branch: Specific branch to clone
    """
    cmd = ['git', 'clone']

    if branch:
        cmd.extend(['-b', branch])

    cmd.append(repo_url)

    if dest_dir:
        cmd.append(str(dest_dir))

    try:
        subprocess.run(cmd, check=True)
        repo_name = dest_dir if dest_dir else Path(repo_url).stem
        print(f"✅ Cloned: {repo_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Clone failed: {e}")
        raise


def mount_google_drive(mount_point: str = '/content/drive'):
    """
    Mount Google Drive in Colab.

    Args:
        mount_point: Drive mount location
    """
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"✅ Google Drive mounted at {mount_point}")
    except ImportError:
        print("❌ Not running in Google Colab")
    except Exception as e:
        print(f"❌ Drive mount failed: {e}")


def setup_workspace(workspace_dir: Path, use_drive: bool = False):
    """
    Initialize a workspace directory.

    Args:
        workspace_dir: Workspace root directory
        use_drive: Whether to use Google Drive for persistence
    """
    if use_drive:
        mount_google_drive()
        workspace_dir = Path('/content/drive/MyDrive') / workspace_dir.name

    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subdirectories
    (workspace_dir / 'models').mkdir(exist_ok=True)
    (workspace_dir / 'outputs').mkdir(exist_ok=True)
    (workspace_dir / 'data').mkdir(exist_ok=True)

    print(f"✅ Workspace initialized: {workspace_dir}")
    print(f"   - models/")
    print(f"   - outputs/")
    print(f"   - data/")

    return workspace_dir


def install_common_ml_packages():
    """Install commonly used ML/AI packages."""
    packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pillow',
        'tqdm',
        'ipywidgets'
    ]
    install_packages(packages)


if __name__ == '__main__':
    print("Google Colab Environment Setup")
    print("=" * 50)

    # Example usage
    print("\nInstalling common ML packages...")
    install_common_ml_packages()

    print("\nInstalling system dependencies...")
    install_apt_packages(['aria2', 'ffmpeg'])

    print("\nSetting up PyTorch...")
    setup_pytorch()

    print("\nInitializing workspace...")
    workspace = setup_workspace(Path('workspace'), use_drive=False)

    print("\n✅ Environment setup complete!")
