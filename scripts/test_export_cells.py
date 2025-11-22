#!/usr/bin/env python3
"""Integration test for export cells (35, 43, 44)."""

import json
import ast
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

def test_cell_35_config_path_resolution():
    """Verify Cell 35 resolves config_path."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_35_source = ''.join(nb['cells'][35]['source'])

    # Check for config_path resolution logic
    required = [
        "config_path",
        "glob.glob",
        "config_{run_name}_*.json",
        "os.path.getmtime",
        "Not found (export will skip config)",
    ]

    for keyword in required:
        assert keyword in cell_35_source, f"Missing required keyword: {keyword}"

    print("✅ Cell 35: config_path resolution present")

def test_cell_35_syntax():
    """Verify Cell 35 has valid Python syntax."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_35_source = ''.join(nb['cells'][35]['source'])

    try:
        ast.parse(cell_35_source)
        print("✅ Cell 35: Python syntax valid")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error at line {e.lineno}: {e.msg}")

def test_cell_43_guards_config_path():
    """Verify Cell 43 guards config_path reference."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_43_source = ''.join(nb['cells'][43]['source'])

    # Check for guarded config_path
    assert "'config_path' in globals()" in cell_43_source, "Missing config_path guard"
    assert "os.path.basename(config_path)" in cell_43_source, "Missing config_path usage"
    assert "'(not found)'" in cell_43_source or '"(not found)"' in cell_43_source, "Missing fallback message"

    # Should NOT have unguarded reference
    lines = nb['cells'][43]['source']
    for line in lines:
        if "os.path.basename(config_path)" in line and "'config_path' in globals()" not in line:
            # Check if it's the guarded conditional expression
            if "if (" not in line and "else" not in line:
                raise AssertionError(f"Unguarded config_path reference: {line}")

    print("✅ Cell 43: config_path properly guarded")

def test_cell_43_syntax():
    """Verify Cell 43 has valid Python syntax."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_43_source = ''.join(nb['cells'][43]['source'])

    try:
        ast.parse(cell_43_source)
        print("✅ Cell 43: Python syntax valid")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error at line {e.lineno}: {e.msg}")

def test_cell_44_uses_run_name():
    """Verify Cell 44 uses run_name (not config.run_name)."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_44_source = ''.join(nb['cells'][44]['source'])

    # Should NOT reference config.run_name in code (comments are OK)
    # Check each line (excluding comments)
    for line in nb['cells'][44]['source']:
        # Skip comment-only lines
        if line.strip().startswith('#'):
            continue
        # Check for config.run_name in actual code
        if 'config.run_name' in line and not line.strip().startswith('#'):
            # Allow in comments after code
            if '#' in line:
                code_part = line.split('#')[0]
                if 'config.run_name' in code_part:
                    raise AssertionError(f"Cell 44 still references config.run_name in code: {line}")
            else:
                raise AssertionError(f"Cell 44 still references config.run_name: {line}")

    # Should use run_name variable
    assert "{rn}_" in cell_44_source or "{run_name}_" in cell_44_source, "Missing run_name usage"

    print("✅ Cell 44: Uses run_name (not config.run_name)")

def test_cell_44_guards_config_path():
    """Verify Cell 44 guards config_path download."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_44_source = ''.join(nb['cells'][44]['source'])

    # Check for guarded config_path download
    assert "'config_path' in globals()" in cell_44_source, "Missing config_path guard"
    assert "os.path.exists(config_path)" in cell_44_source, "Missing config_path existence check"

    print("✅ Cell 44: config_path download properly guarded")

def test_cell_44_syntax():
    """Verify Cell 44 has valid Python syntax."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_44_source = ''.join(nb['cells'][44]['source'])

    try:
        ast.parse(cell_44_source)
        print("✅ Cell 44: Python syntax valid")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error at line {e.lineno}: {e.msg}")

def test_export_workflow_resilience():
    """Verify export cells handle missing variables gracefully."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Cell 35: Must set config_path (even if None)
    cell_35_source = ''.join(nb['cells'][35]['source'])
    assert "config_path = None" in cell_35_source, "Cell 35 doesn't set config_path = None fallback"

    # Cell 43: Must handle None config_path
    cell_43_source = ''.join(nb['cells'][43]['source'])
    assert "'config_path' in globals()" in cell_43_source, "Cell 43 doesn't check if config_path exists"

    # Cell 44: Must handle None config_path
    cell_44_source = ''.join(nb['cells'][44]['source'])
    assert "'config_path' in globals()" in cell_44_source, "Cell 44 doesn't check if config_path exists"

    print("✅ Export workflow: Resilient to missing variables")

if __name__ == "__main__":
    print("=" * 70)
    print("EXPORT CELLS INTEGRATION TESTS")
    print("=" * 70)
    print()

    # Cell 35 tests
    test_cell_35_config_path_resolution()
    test_cell_35_syntax()
    print()

    # Cell 43 tests
    test_cell_43_guards_config_path()
    test_cell_43_syntax()
    print()

    # Cell 44 tests
    test_cell_44_uses_run_name()
    test_cell_44_guards_config_path()
    test_cell_44_syntax()
    print()

    # Workflow tests
    test_export_workflow_resilience()
    print()

    print("=" * 70)
    print("🎉 All export cell integration tests passed!")
    print("=" * 70)
