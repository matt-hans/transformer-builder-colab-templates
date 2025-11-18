#!/usr/bin/env python3
"""
Efficient model download utility for Google Colab using aria2c.

This script provides fast, resumable downloads with parallel connections
optimized for Colab's network environment.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Union


def download_with_aria2c(
    url: str,
    dest_dir: Union[str, Path],
    filename: Optional[str] = None,
    max_connections: int = 16,
    split_size: int = 16,
    silent: bool = False
) -> bool:
    """
    Download a file using aria2c with optimized settings for Colab.

    Args:
        url: Download URL
        dest_dir: Destination directory
        filename: Optional custom filename (extracted from URL if not provided)
        max_connections: Maximum parallel connections (default: 16)
        split_size: Number of splits per connection (default: 16)
        silent: Suppress aria2c output (default: False)

    Returns:
        True if download succeeded, False otherwise
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Build aria2c command
    cmd = [
        'aria2c',
        '--console-log-level=error' if silent else '--console-log-level=info',
        '-c',  # Continue incomplete downloads
        f'-x{max_connections}',  # Max connections per server
        f'-s{split_size}',  # Split downloads
        '-k', '1M',  # Min split size
        url,
        '-d', str(dest_dir)
    ]

    if filename:
        cmd.extend(['-o', filename])

    try:
        result = subprocess.run(cmd, check=True, capture_output=silent)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        print("aria2c not found. Install with: !apt -y install -qq aria2")
        return False


def download_from_civitai(
    model_id: str,
    dest_dir: Union[str, Path],
    filename: Optional[str] = None,
    api_token: Optional[str] = None
) -> bool:
    """
    Download a model from Civitai with authentication.

    Args:
        model_id: Civitai model ID
        dest_dir: Destination directory
        filename: Optional custom filename
        api_token: Civitai API token (required for some models)

    Returns:
        True if download succeeded, False otherwise
    """
    base_url = f"https://civitai.com/api/download/models/{model_id}"

    if api_token:
        # Use wget for authenticated downloads
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        output_flag = f'-O {filename}' if filename else ''
        cmd = f'wget --content-disposition --header="Authorization: Bearer {api_token}" {base_url} {output_flag} -P {dest_dir}'

        try:
            result = subprocess.run(cmd, shell=True, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    else:
        # Use aria2c for public models
        return download_with_aria2c(base_url, dest_dir, filename)


def install_aria2c():
    """Install aria2c if not already present."""
    try:
        subprocess.run(['aria2c', '--version'], capture_output=True, check=True)
        print("aria2c already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing aria2c...")
        subprocess.run(['apt', '-y', 'install', '-qq', 'aria2'], check=True)
        print("aria2c installed successfully")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: python model_download.py <url> <dest_dir> [filename]")
        sys.exit(1)

    url = sys.argv[1]
    dest_dir = sys.argv[2]
    filename = sys.argv[3] if len(sys.argv) > 3 else None

    # Ensure aria2c is installed
    install_aria2c()

    # Download
    success = download_with_aria2c(url, dest_dir, filename)
    sys.exit(0 if success else 1)
