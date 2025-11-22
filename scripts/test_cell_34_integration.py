#!/usr/bin/env python3
"""Integration test for Cell 34 (Load Model Weights)."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

def test_cell_structure():
    """Verify Cell 34 has correct Jupyter structure."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_34 = nb['cells'][34]

    # Check required fields
    assert cell_34['cell_type'] == 'code', "Cell 34 must be code cell"
    assert 'execution_count' in cell_34, "Missing execution_count"
    assert 'outputs' in cell_34, "Missing outputs"
    assert 'metadata' in cell_34, "Missing metadata"
    assert cell_34['metadata'].get('cellView') == 'form', "Must be form cell"

    print("✅ Cell 34 structure valid")

def test_cell_content():
    """Verify Cell 34 has all required sections."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_34_source = ''.join(nb['cells'][34]['source'])

    # Check required sections
    required = [
        "LOAD MODEL WEIGHTS FROM CHECKPOINT",
        "if 'model' not in globals():",
        "list_checkpoints",
        "checkpoint_path",
        "torch.load",
        "load_state_dict",
        "model.eval()",
        "CHECKPOINT INFO",
        "MODEL INFO",
        "ARCHITECTURE PREVIEW",
        "NEXT STEPS",
        "RuntimeError",
        # New: intelligent checkpoint discovery
        "import os",
        "if 'ckpt_dir' in globals()",
        "checkpoint_search_paths",
        "os.path.exists",
    ]

    for keyword in required:
        assert keyword in cell_34_source, f"Missing required keyword: {keyword}"

    print("✅ Cell 34 content complete")

def test_cell_order():
    """Verify cells are in correct order."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Extract titles
    titles = []
    for i in [33, 34, 35]:
        title = nb['cells'][i]['source'][0]
        titles.append((i, title[:60]))

    # Verify order
    assert "Recover Training Results" in titles[0][1], "Cell 33 should be Recovery"
    assert "Load Model Weights" in titles[1][1], "Cell 34 should be Load Weights"
    assert "Extract Session Variables" in titles[2][1], "Cell 35 should be Extract"

    print("✅ Cell order correct")
    for i, title in titles:
        print(f"   Cell {i}: {title}")

if __name__ == "__main__":
    test_cell_structure()
    test_cell_content()
    test_cell_order()
    print()
    print("🎉 All integration tests passed!")
