#!/usr/bin/env python3
"""Integration test for Cell 8 (Import Infrastructure) - Function Validation."""

import json
import ast
from pathlib import Path

NOTEBOOK_PATH = Path('training.ipynb')

def test_cell_structure():
    """Verify Cell 8 has correct Jupyter structure."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8 = nb['cells'][8]

    # Check required fields
    assert cell_8['cell_type'] == 'code', "Cell 8 must be code cell"
    assert 'source' in cell_8, "Missing source"
    assert len(cell_8['source']) > 0, "Source is empty"

    print("✅ Cell 8 structure valid")

def test_cell_syntax():
    """Verify Cell 8 has valid Python syntax."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8_source = ''.join(nb['cells'][8]['source'])

    # Validate Python syntax with AST
    try:
        ast.parse(cell_8_source)
        print("✅ Cell 8 Python syntax valid")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error at line {e.lineno}: {e.msg}")

def test_import_statements():
    """Verify Cell 8 imports all required components."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8_source = ''.join(nb['cells'][8]['source'])

    # Required imports
    required_imports = [
        "from utils.training.engine.trainer import Trainer",
        "from utils.training.engine.loop import TrainingLoop, ValidationLoop",
        "from utils.training.training_config import TrainingConfig, TrainingConfigBuilder",
        "from utils.training.task_spec import TaskSpec",
        "from utils.training.metrics_tracker import MetricsTracker",
        "from utils.training.experiment_db import ExperimentDB",
        "from utils.training.training_core import TrainingCoordinator",
        "from utils.training.drift_metrics import compute_dataset_profile, compare_profiles",
        "from utils.training.dashboard import TrainingDashboard",
        "from utils.training.export_utilities import create_export_bundle",
        "from utils.adapters.model_adapter import UniversalModelAdapter, FlashAttentionWrapper",
        "from utils.training.engine.data import UniversalDataModule",
        "from utils.tokenization.data_module import SimpleDataModule",
    ]

    for import_stmt in required_imports:
        assert import_stmt in cell_8_source, f"Missing import: {import_stmt}"

    print("✅ Cell 8 has all required imports")

def test_validation_includes_functions():
    """Verify Cell 8 validates create_export_bundle and drift functions."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8_source = ''.join(nb['cells'][8]['source'])

    # Critical: These functions MUST be in required_classes validation
    # If they're not validated, import failures can be silently swallowed
    required_validations = [
        "'create_export_bundle': create_export_bundle",
        "'compute_dataset_profile': compute_dataset_profile",
        "'compare_profiles': compare_profiles",
    ]

    for validation in required_validations:
        assert validation in cell_8_source, (
            f"Missing validation: {validation}\n"
            f"Without validation, import failures will be silently swallowed!"
        )

    print("✅ Cell 8 validates create_export_bundle and drift functions")

def test_validation_dict_structure():
    """Verify required_classes dict includes all 17 components (14 classes + 3 functions)."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8_source = ''.join(nb['cells'][8]['source'])

    # Expected components in required_classes dict
    expected_components = [
        # Training engine (3)
        "'Trainer': Trainer",
        "'TrainingLoop': TrainingLoop",
        "'ValidationLoop': ValidationLoop",
        # Configuration (3)
        "'TrainingConfig': TrainingConfig",
        "'TrainingConfigBuilder': TrainingConfigBuilder",
        "'TaskSpec': TaskSpec",
        # Utilities (3)
        "'MetricsTracker': MetricsTracker",
        "'ExperimentDB': ExperimentDB",
        "'TrainingCoordinator': TrainingCoordinator",
        # Data modules (2)
        "'SimpleDataModule': SimpleDataModule",
        "'UniversalDataModule': UniversalDataModule",
        # Adapters (2)
        "'UniversalModelAdapter': UniversalModelAdapter",
        "'FlashAttentionWrapper': FlashAttentionWrapper",
        # Analysis (1)
        "'TrainingDashboard': TrainingDashboard",
        # v3.5 Export & v3.6 Drift (3) - CRITICAL
        "'create_export_bundle': create_export_bundle",
        "'compute_dataset_profile': compute_dataset_profile",
        "'compare_profiles': compare_profiles",
    ]

    for component in expected_components:
        assert component in cell_8_source, f"Missing component validation: {component}"

    print(f"✅ Cell 8 validates all {len(expected_components)} components")

def test_validation_message():
    """Verify validation success message mentions 17 components."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8_source = ''.join(nb['cells'][8]['source'])

    # Should show component count (not "classes" since we have functions too)
    assert "Validated: {len(required_classes)} components imported successfully" in cell_8_source, \
        "Validation message should mention 'components' (not 'classes') and use {len(required_classes)}"

    print("✅ Cell 8 validation message correct")

def test_export_feature_documentation():
    """Verify Cell 8 documents export bundle feature."""
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    cell_8_source = ''.join(nb['cells'][8]['source'])

    # Should mention export bundle in v3.5 features section
    assert "Export bundle generation" in cell_8_source or "export bundle" in cell_8_source, \
        "Cell 8 should document export bundle feature in v3.5 section"

    print("✅ Cell 8 documents export bundle feature")

if __name__ == "__main__":
    print("=" * 70)
    print("CELL 8 (IMPORT INFRASTRUCTURE) INTEGRATION TESTS")
    print("=" * 70)
    print()

    # Cell structure tests
    test_cell_structure()
    test_cell_syntax()
    print()

    # Import tests
    test_import_statements()
    print()

    # Validation tests (CRITICAL - prevents silent failures)
    test_validation_includes_functions()
    test_validation_dict_structure()
    test_validation_message()
    print()

    # Documentation tests
    test_export_feature_documentation()
    print()

    print("=" * 70)
    print("🎉 All Cell 8 integration tests passed!")
    print("=" * 70)
    print()
    print("✅ Function validation prevents silent import failures")
    print("✅ Cell 8 will fail-fast if create_export_bundle import fails")
    print("✅ Users will get clear error message instead of NameError in Cell 46")
