#!/usr/bin/env python3
"""
Verification script for training.ipynb Cell 20 fix.

This script validates that Cell 20 correctly implements URL hash parameter
extraction using google.colab.output.eval_js().

Usage:
    python verify_cell20_fix.py
"""

import json
import sys

def verify_cell20_fix(notebook_path='training.ipynb'):
    """Verify Cell 20 has the correct implementation."""

    print("=" * 70)
    print(" " * 15 + "CELL 20 FIX VERIFICATION")
    print("=" * 70)

    # Load notebook
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        print(f"✅ Loaded notebook: {notebook_path}")
    except Exception as e:
        print(f"❌ Failed to load notebook: {e}")
        return False

    # Get Cell 20
    try:
        cell = nb['cells'][20]
        source = cell.get('source', '')
        print(f"✅ Found Cell 20 (type: {cell.get('cell_type')})")
    except IndexError:
        print("❌ Cell 20 not found (notebook has fewer than 21 cells)")
        return False

    # Verification checks
    checks = {
        'Uses output.eval_js()': {
            'pattern': 'output.eval_js(js_code)',
            'description': 'Executes JavaScript and captures return value'
        },
        'Returns JSON from JavaScript': {
            'pattern': 'return JSON.stringify',
            'description': 'JavaScript returns serialized JSON'
        },
        'Parses JSON in Python': {
            'pattern': 'json.loads(url_params_json)',
            'description': 'Deserializes JavaScript return value'
        },
        'Extracts gist_id from URL': {
            'pattern': 'gist_id_from_url',
            'description': 'Stores URL-extracted gist_id separately'
        },
        'Has manual form fallback': {
            'pattern': 'gist_id_manual = ""  #@param',
            'description': 'Colab form for manual input'
        },
        'Sets final gist_id variable': {
            'pattern': 'gist_id = gist_id_from_url or',
            'description': 'Merges sources with priority'
        },
        'Has priority system': {
            'pattern': 'gist_id_from_url or gist_id_manual or gist_id_env',
            'description': 'URL > Manual > Environment priority'
        },
        'Has error handling': {
            'pattern': 'except Exception as e:',
            'description': 'Graceful fallback on JavaScript errors'
        },
        'Displays source information': {
            'pattern': 'source = "URL hash" if gist_id_from_url',
            'description': 'Shows where gist_id came from'
        },
        'Uses Colab output module': {
            'pattern': 'from google.colab import output',
            'description': 'Imports correct Colab API'
        }
    }

    print("\nVerification Results:")
    print("-" * 70)

    all_passed = True
    for check_name, check_info in checks.items():
        pattern = check_info['pattern']
        description = check_info['description']
        passed = pattern in source

        status = '✅' if passed else '❌'
        print(f"{status} {check_name}")
        print(f"   {description}")

        if not passed:
            print(f"   Missing pattern: {pattern}")
            all_passed = False

    print("-" * 70)

    # Overall result
    print("\nOverall Result:")
    if all_passed:
        print("✅ ALL CHECKS PASSED - Cell 20 fix is correct!")
        print("\nExpected Behavior:")
        print("  1. Open notebook with URL: .../training.ipynb#gist_id=abc123&name=MyModel")
        print("  2. Run Cell 20")
        print("  3. Verify output: '✅ Model Source: URL hash'")
        print("  4. Check variables: gist_id='abc123', model_name='MyModel'")
    else:
        print("❌ SOME CHECKS FAILED - Cell 20 fix may be incomplete")
        return False

    print("=" * 70)
    return all_passed


def show_cell20_source(notebook_path='training.ipynb'):
    """Display Cell 20 source code."""

    print("\n" + "=" * 70)
    print(" " * 20 + "CELL 20 SOURCE CODE")
    print("=" * 70)

    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)

        source = nb['cells'][20].get('source', '')

        # Show first 30 lines
        lines = source.split('\n')
        for i, line in enumerate(lines[:30], 1):
            print(f"{i:3d} | {line}")

        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")

        print("=" * 70)

    except Exception as e:
        print(f"❌ Could not display source: {e}")


if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'training.ipynb'

    # Run verification
    success = verify_cell20_fix(notebook_path)

    # Optionally show source
    if '--show-source' in sys.argv:
        show_cell20_source(notebook_path)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
