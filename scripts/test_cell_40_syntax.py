#!/usr/bin/env python3
"""Integration test for Cell 40 (Display Metrics Table)."""

import json
import ast
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

def test_cell_structure():
    """Verify Cell 40 has correct Jupyter structure."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_40 = nb['cells'][40]

    # Check required fields
    assert cell_40['cell_type'] == 'code', "Cell 40 must be code cell"
    assert 'source' in cell_40, "Missing source"
    assert len(cell_40['source']) > 0, "Source is empty"

    print("✅ Cell 40 structure valid")

def test_cell_syntax():
    """Verify Cell 40 has valid Python syntax."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_40_source = ''.join(nb['cells'][40]['source'])

    # Validate Python syntax with AST
    try:
        ast.parse(cell_40_source)
        print("✅ Cell 40 Python syntax valid")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error at line {e.lineno}: {e.msg}")

def test_cell_content():
    """Verify Cell 40 has all required sections."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_40_source = ''.join(nb['cells'][40]['source'])

    # Check required sections
    required = [
        "Display metrics table",
        "import pandas as pd",
        "import os",
        "if 'metrics_df' not in globals():",
        "pd.set_option",
        "display_cols",
        "metrics_df[available_cols]",
        "Export to CSV",
        "if 'run_name' not in globals():",
        "if 'workspace_root' not in globals():",
        "results_dir",
        "os.makedirs",
        "metrics_df.to_csv",
        "except Exception as e:",
    ]

    for keyword in required:
        assert keyword in cell_40_source, f"Missing required keyword: {keyword}"

    print("✅ Cell 40 content complete")

def test_no_orphaned_else():
    """Verify Cell 40 has no orphaned else blocks."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_40_source = ''.join(nb['cells'][40]['source'])

    # Parse and check for else blocks
    tree = ast.parse(cell_40_source)

    # If we can parse it, there are no orphaned else blocks
    # (AST parser would fail with SyntaxError if there were)
    print("✅ Cell 40 has no orphaned else blocks")

def test_variable_fallbacks():
    """Verify Cell 40 provides fallback values for missing variables."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_40_source = ''.join(nb['cells'][40]['source'])

    # Check for fallback assignments
    assert "run_name = 'training_run'" in cell_40_source, "Missing run_name fallback"
    assert "workspace_root = './workspace'" in cell_40_source, "Missing workspace_root fallback"

    print("✅ Cell 40 has proper variable fallbacks")

if __name__ == "__main__":
    test_cell_structure()
    test_cell_syntax()
    test_cell_content()
    test_no_orphaned_else()
    test_variable_fallbacks()
    print()
    print("🎉 All Cell 40 integration tests passed!")
