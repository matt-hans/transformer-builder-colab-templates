#!/usr/bin/env python3
"""
Test script to identify which package in requirements-colab.txt corrupts numpy.
This should be run in a fresh Colab environment.
"""

import subprocess
import sys

# Packages to test (from requirements-colab.txt)
packages_to_test = [
    "datasets>=2.16.0,<3.0.0",
    "tokenizers>=0.15.0,<1.0.0",
    "huggingface-hub>=0.20.0,<1.0.0",
    "torchinfo>=1.8.0,<3.0.0",
    "optuna>=3.0.0,<4.0.0",
    "pytest>=7.4.0,<8.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
]

def test_numpy_integrity():
    """Test if numpy C extensions are intact."""
    try:
        from numpy import rec, core
        from numpy._core import umath
        from numpy._core.umath import _center
        return True
    except ImportError as e:
        print(f"  ❌ numpy corrupted: {e}")
        return False

def test_package(package_spec):
    """Test if installing a package corrupts numpy."""
    print(f"\n{'='*80}")
    print(f"Testing: {package_spec}")
    print('='*80)

    # Test numpy before installation
    print("Before installation:")
    before = test_numpy_integrity()

    # Install the package
    print(f"\nInstalling {package_spec}...")
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', package_spec],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    # Test numpy after installation
    print("After installation:")
    after = test_numpy_integrity()

    if before and not after:
        print(f"\n🚨 FOUND CULPRIT: {package_spec} corrupts numpy!")
        return package_spec
    elif not before:
        print(f"\n⚠️  numpy was already corrupted before installing {package_spec}")
    else:
        print(f"\n✅ {package_spec} is safe")

    return None

if __name__ == "__main__":
    print("="*80)
    print("NUMPY CORRUPTION DIAGNOSTIC TEST")
    print("="*80)
    print("\nTesting each package individually to find the culprit...")

    # Initial numpy check
    print("\nInitial numpy integrity check:")
    initial_state = test_numpy_integrity()

    if not initial_state:
        print("\n❌ numpy is already corrupted! Please restart runtime.")
        sys.exit(1)

    print("\n✅ numpy is intact initially\n")

    # Test each package
    culprits = []
    for package in packages_to_test:
        culprit = test_package(package)
        if culprit:
            culprits.append(culprit)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if culprits:
        print(f"\n🚨 Found {len(culprits)} package(s) that corrupt numpy:")
        for c in culprits:
            print(f"  - {c}")
    else:
        print("\n✅ None of the tested packages corrupted numpy individually")
        print("⚠️  The issue may be a combination of packages or pytorch-lightning itself")
