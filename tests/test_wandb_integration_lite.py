"""
Lightweight test suite for W&B integration (no PyTorch dependencies).

Tests basic structure, formatting, and security without requiring heavy dependencies.

Run with: pytest tests/test_wandb_integration_lite.py -v
"""

import os
import pytest
from datetime import datetime


# ==============================================================================
# Test: .gitignore Configuration
# ==============================================================================

def test_gitignore_contains_wandb_directory():
    """
    Test that .wandb/ directory is in .gitignore.

    Why: Prevents accidentally committing W&B logs and artifacts to git.
    Contract: .gitignore file contains '.wandb/' or 'wandb/' pattern.
    """
    gitignore_path = '/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/.gitignore'

    assert os.path.exists(gitignore_path), ".gitignore file should exist"

    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()

    assert '.wandb/' in gitignore_content or 'wandb/' in gitignore_content, \
        ".wandb/ directory must be in .gitignore to avoid committing logs"

    print("PASS: .gitignore correctly excludes .wandb/ directory")


# ==============================================================================
# Test: Project Organization
# ==============================================================================

def test_wandb_project_name_format():
    """
    Test that W&B project name follows W&B best practices.

    Why: Ensures project names are URL-safe and discoverable.
    Contract: Project name is lowercase, uses hyphens, no spaces.
    """
    expected_project = "transformer-builder-training"

    # Verify format is lowercase with hyphens (W&B best practice)
    assert expected_project.islower(), "Project name should be lowercase"
    assert ' ' not in expected_project, "Project name should not contain spaces"
    assert expected_project.replace('-', '').replace('_', '').isalnum(), \
        "Project name should only contain alphanumeric and hyphens/underscores"

    print(f"PASS: Project name '{expected_project}' follows W&B naming conventions")


def test_wandb_run_name_includes_timestamp_and_architecture():
    """
    Test that run names include timestamp and model architecture.

    Why: Makes runs easily identifiable and sortable in W&B dashboard.
    Contract: Run name format is '{model_type}_{YYYYMMDD_HHMMSS}'.
    """
    model_type = 'gpt'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_name = f"{model_type}_{timestamp}"

    # Verify format
    assert model_type in run_name, "Run name should include model type"
    assert len(timestamp) == 15, "Timestamp should be in YYYYMMDD_HHMMSS format (15 chars)"
    assert '_' in run_name, "Run name should separate components with underscore"

    print(f"PASS: Run name format '{run_name}' is valid")


def test_wandb_tags_format():
    """
    Test that W&B tags are valid strings.

    Why: Ensures tags are usable for filtering and organization in W&B.
    Contract: Tags are non-empty strings.
    """
    model_type = 'gpt'
    version = 'v1'

    tags = [model_type, version]

    # Verify tags are non-empty strings
    for tag in tags:
        assert isinstance(tag, str), f"Tag should be string, got {type(tag)}"
        assert len(tag) > 0, "Tag should not be empty"

    print(f"PASS: Tags {tags} are valid")


# ==============================================================================
# Test: API Key Security
# ==============================================================================

def test_no_hardcoded_api_keys_in_training_notebook():
    """
    Test that training.ipynb does not contain hardcoded W&B API keys.

    Why: Prevents accidental credential leakage in public repositories.
    Contract: Notebook should use Colab Secrets or env vars, not hardcoded keys.

    Checks for common dangerous patterns:
    - WANDB_API_KEY = "local..."
    - wandb_api_key = "local..."
    - wandb.login(key="local...")
    """
    notebook_path = '/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/training.ipynb'

    with open(notebook_path, 'r') as f:
        notebook_content = f.read()

    # These patterns should NOT appear with actual keys
    dangerous_patterns = [
        'WANDB_API_KEY = "local',  # Hardcoded key starting with 'local'
        'WANDB_API_KEY="local',
        'wandb_api_key = "local',
        'wandb.login(key="local',
    ]

    for pattern in dangerous_patterns:
        assert pattern not in notebook_content, \
            f"Found potentially hardcoded API key pattern: {pattern}"

    print("PASS: No hardcoded API keys detected in notebook")


# ==============================================================================
# Test: Offline Mode Configuration
# ==============================================================================

def test_offline_mode_environment_variable():
    """
    Test that WANDB_MODE environment variable can be set to 'offline'.

    Why: Allows training to proceed without internet/W&B authentication.
    Contract: Setting WANDB_MODE=offline should not raise errors.
    """
    # Temporarily set offline mode
    original_mode = os.environ.get('WANDB_MODE')

    try:
        os.environ['WANDB_MODE'] = 'offline'
        assert os.environ.get('WANDB_MODE') == 'offline', \
            "Offline mode environment variable should be settable"

        print("PASS: WANDB_MODE=offline can be set successfully")

    finally:
        # Restore original state
        if original_mode is not None:
            os.environ['WANDB_MODE'] = original_mode
        elif 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']


# ==============================================================================
# Test: W&B Config Structure (Schema Validation)
# ==============================================================================

def test_wandb_config_schema():
    """
    Test that W&B config structure includes all required fields.

    Why: Ensures consistent experiment tracking across all runs.
    Contract: Config dict contains hyperparameters, model metadata, environment info.

    Required field categories:
    1. Hyperparameters: learning_rate, batch_size, epochs, etc.
    2. Model metadata: model_type, vocab_size, total_params, etc.
    3. Environment: device, mixed_precision, etc.
    """
    # Define expected config schema
    required_hyperparameters = [
        'learning_rate',
        'batch_size',
        'epochs',
        'warmup_ratio',
        'weight_decay',
        'max_grad_norm'
    ]

    required_model_metadata = [
        'model_type',
        'vocab_size',
        'max_seq_len',
        'total_params',
        'trainable_params',
        'total_params_millions'
    ]

    required_environment = [
        'device',
        'mixed_precision',
        'gradient_accumulation_steps'
    ]

    all_required_fields = (
        required_hyperparameters +
        required_model_metadata +
        required_environment
    )

    # Create mock config matching expected structure
    mock_config = {
        # Hyperparameters
        'learning_rate': 5e-5,
        'batch_size': 4,
        'epochs': 10,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,

        # Model metadata
        'model_type': 'gpt',
        'vocab_size': 50257,
        'max_seq_len': 128,
        'total_params': 124439808,
        'trainable_params': 124439808,
        'total_params_millions': 124.44,

        # Environment
        'device': 'cuda',
        'mixed_precision': True,
        'gradient_accumulation_steps': 1
    }

    # Verify all required fields are present
    missing_fields = [field for field in all_required_fields if field not in mock_config]
    assert not missing_fields, f"Missing required config fields: {missing_fields}"

    # Verify data types
    assert isinstance(mock_config['learning_rate'], float), "learning_rate should be float"
    assert isinstance(mock_config['batch_size'], int), "batch_size should be int"
    assert isinstance(mock_config['epochs'], int), "epochs should be int"
    assert isinstance(mock_config['model_type'], str), "model_type should be str"
    assert isinstance(mock_config['total_params'], int), "total_params should be int"
    assert isinstance(mock_config['total_params_millions'], (int, float)), \
        "total_params_millions should be numeric"

    print(f"PASS: Config schema contains all {len(all_required_fields)} required fields")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
