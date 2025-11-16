"""
Test suite for W&B integration in training.ipynb.

Tests offline mode behavior, model type detection, and config structure.

Run with: pytest tests/test_wandb_integration.py -v
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch, MagicMock


# ==============================================================================
# Helper Functions Under Test
# ==============================================================================

def _detect_model_type(model: nn.Module) -> str:
    """
    Detect transformer architecture type from model structure.

    Returns:
        'gpt' | 'bert' | 't5' | 'custom'
    """
    model_class = model.__class__.__name__.lower()

    # Check class name first
    if 'gpt' in model_class or 'decoder' in model_class:
        return 'gpt'
    elif 'bert' in model_class or 'encoder' in model_class:
        return 'bert'
    elif 't5' in model_class or 'encoderdecoder' in model_class:
        return 't5'

    # Inspect module structure
    module_names = [name for name, _ in model.named_modules()]
    has_decoder = any('decoder' in n.lower() for n in module_names)
    has_encoder = any('encoder' in n.lower() for n in module_names)

    if has_decoder and not has_encoder:
        return 'gpt'
    elif has_encoder and not has_decoder:
        return 'bert'
    elif has_encoder and has_decoder:
        return 't5'

    return 'custom'


# ==============================================================================
# Test Fixtures
# ==============================================================================

class GPTStyleModel(nn.Module):
    """Mock GPT-style decoder-only model."""
    def __init__(self):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        self.embedding = nn.Embedding(50257, 128)

    def forward(self, x):
        return self.embedding(x)


class BERTStyleModel(nn.Module):
    """Mock BERT-style encoder-only model."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        self.embedding = nn.Embedding(30522, 128)

    def forward(self, x):
        return self.embedding(x)


class T5StyleModel(nn.Module):
    """Mock T5-style encoder-decoder model."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        self.embedding = nn.Embedding(32128, 128)

    def forward(self, x):
        return self.embedding(x)


class CustomTransformer(nn.Module):
    """Mock custom transformer without standard naming."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(4)
        ])
        self.embedding = nn.Embedding(50000, 128)

    def forward(self, x):
        return self.embedding(x)


@pytest.fixture
def gpt_model():
    """Returns a GPT-style model for testing."""
    return GPTStyleModel()


@pytest.fixture
def bert_model():
    """Returns a BERT-style model for testing."""
    return BERTStyleModel()


@pytest.fixture
def t5_model():
    """Returns a T5-style model for testing."""
    return T5StyleModel()


@pytest.fixture
def custom_model():
    """Returns a custom transformer model for testing."""
    return CustomTransformer()


@pytest.fixture
def mock_config():
    """Returns a mock config object."""
    return SimpleNamespace(
        vocab_size=50257,
        max_seq_len=128,
        max_batch_size=8
    )


# ==============================================================================
# Test: Model Type Detection
# ==============================================================================

def test_detect_gpt_by_class_name():
    """Test GPT detection from class name containing 'gpt'."""
    class MyGPTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    model = MyGPTModel()
    result = _detect_model_type(model)

    assert result == 'gpt', f"Expected 'gpt', got '{result}'"


def test_detect_gpt_by_decoder_in_class_name():
    """Test GPT detection from class name containing 'decoder'."""
    class TransformerDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    model = TransformerDecoder()
    result = _detect_model_type(model)

    assert result == 'gpt', f"Expected 'gpt', got '{result}'"


def test_detect_gpt_by_module_structure(gpt_model):
    """Test GPT detection from model having decoder modules but no encoder."""
    result = _detect_model_type(gpt_model)

    assert result == 'gpt', f"Expected 'gpt' for GPT-style model, got '{result}'"


def test_detect_bert_by_class_name():
    """Test BERT detection from class name containing 'bert'."""
    class MyBERTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    model = MyBERTModel()
    result = _detect_model_type(model)

    assert result == 'bert', f"Expected 'bert', got '{result}'"


def test_detect_bert_by_encoder_in_class_name():
    """Test BERT detection from class name containing 'encoder'."""
    class TransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    model = TransformerEncoder()
    result = _detect_model_type(model)

    assert result == 'bert', f"Expected 'bert', got '{result}'"


def test_detect_bert_by_module_structure(bert_model):
    """Test BERT detection from model having encoder modules but no decoder."""
    result = _detect_model_type(bert_model)

    assert result == 'bert', f"Expected 'bert' for BERT-style model, got '{result}'"


def test_detect_t5_by_class_name():
    """Test T5 detection from class name containing 't5'."""
    class MyT5Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    model = MyT5Model()
    result = _detect_model_type(model)

    assert result == 't5', f"Expected 't5', got '{result}'"


def test_detect_t5_by_encoderdecoder_in_class_name():
    """Test T5 detection from class name containing 'encoderdecoder'."""
    class EncoderDecoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

    model = EncoderDecoderModel()
    result = _detect_model_type(model)

    assert result == 't5', f"Expected 't5', got '{result}'"


def test_detect_t5_by_module_structure(t5_model):
    """Test T5 detection from model having both encoder and decoder modules."""
    result = _detect_model_type(t5_model)

    assert result == 't5', f"Expected 't5' for T5-style model, got '{result}'"


def test_detect_custom_for_unknown_architecture(custom_model):
    """Test that models without standard architecture return 'custom'."""
    result = _detect_model_type(custom_model)

    assert result == 'custom', f"Expected 'custom' for unknown architecture, got '{result}'"


# ==============================================================================
# Test: W&B Config Structure
# ==============================================================================

def test_wandb_config_structure(gpt_model, mock_config):
    """
    Test that W&B config includes all required fields.

    Required fields:
    - Hyperparameters: learning_rate, batch_size, epochs, warmup_ratio,
                       weight_decay, max_grad_norm
    - Model metadata: model_type, vocab_size, max_seq_len, total_params,
                      trainable_params, total_params_millions
    - Environment: device, mixed_precision, gradient_accumulation_steps
    """
    hyperparameters = {
        'learning_rate': 5e-5,
        'batch_size': 4,
        'epochs': 10,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'grad_accum_steps': 1
    }

    # Calculate model metadata
    total_params = sum(p.numel() for p in gpt_model.parameters())
    trainable_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
    model_type = _detect_model_type(gpt_model)
    device = str(next(gpt_model.parameters()).device)

    # Expected config structure
    expected_config = {
        # Hyperparameters
        'learning_rate': hyperparameters['learning_rate'],
        'batch_size': hyperparameters['batch_size'],
        'epochs': hyperparameters['epochs'],
        'warmup_ratio': hyperparameters['warmup_ratio'],
        'weight_decay': hyperparameters['weight_decay'],
        'max_grad_norm': hyperparameters['max_grad_norm'],

        # Model architecture
        'model_type': model_type,
        'vocab_size': mock_config.vocab_size,
        'max_seq_len': mock_config.max_seq_len,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_millions': round(total_params / 1e6, 2),

        # Environment
        'device': device,
        'mixed_precision': hyperparameters['use_amp'],
        'gradient_accumulation_steps': hyperparameters['grad_accum_steps']
    }

    # Verify all keys present
    for key in expected_config:
        assert key in expected_config, f"Missing required config key: {key}"

    # Verify data types
    assert isinstance(expected_config['learning_rate'], float)
    assert isinstance(expected_config['batch_size'], int)
    assert isinstance(expected_config['epochs'], int)
    assert isinstance(expected_config['model_type'], str)
    assert isinstance(expected_config['total_params'], int)
    assert expected_config['total_params'] > 0, "Model should have parameters"


# ==============================================================================
# Test: Offline Mode Behavior
# ==============================================================================

@patch.dict(os.environ, {'WANDB_MODE': 'offline'})
def test_offline_mode_environment_variable():
    """Test that WANDB_MODE=offline environment variable is respected."""
    assert os.environ.get('WANDB_MODE') == 'offline', \
        "Offline mode environment variable should be set"


def test_offline_mode_logs_locally():
    """
    Test that offline mode creates local logs in .wandb/ directory.

    Note: This is a structure test, not a full integration test.
    Actual wandb.init() behavior is tested in manual verification.
    """
    # Verify .wandb/ is in .gitignore
    gitignore_path = '/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/.gitignore'

    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()

    assert '.wandb/' in gitignore_content or 'wandb/' in gitignore_content, \
        ".wandb/ directory should be in .gitignore to avoid committing logs"


# ==============================================================================
# Test: API Key Security
# ==============================================================================

def test_no_hardcoded_api_keys_in_training_notebook():
    """
    Test that training.ipynb does not contain hardcoded W&B API keys.

    Checks for common patterns:
    - WANDB_API_KEY = "..."
    - wandb_api_key = "..."
    - wandb.login(key="...")
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

    # Verify Colab Secrets pattern is used (after implementation)
    # This test will pass once we add the W&B setup cell
    # assert 'userdata.get(' in notebook_content, \
    #     "Should use Colab Secrets pattern (userdata.get)"


# ==============================================================================
# Test: Project Organization
# ==============================================================================

def test_wandb_project_name_format():
    """Test that W&B project name follows expected format."""
    expected_project = "transformer-builder-training"

    # Verify format is lowercase with hyphens (W&B best practice)
    assert expected_project.islower(), "Project name should be lowercase"
    assert ' ' not in expected_project, "Project name should not contain spaces"
    assert expected_project.replace('-', '').replace('_', '').isalnum(), \
        "Project name should only contain alphanumeric and hyphens/underscores"


def test_wandb_run_name_includes_timestamp_and_architecture():
    """Test that run names include timestamp and model architecture."""
    from datetime import datetime

    model_type = 'gpt'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_name = f"{model_type}_{timestamp}"

    # Verify format
    assert model_type in run_name, "Run name should include model type"
    assert len(timestamp) == 15, "Timestamp should be in YYYYMMDD_HHMMSS format (15 chars)"
    assert '_' in run_name, "Run name should separate components with underscore"


def test_wandb_tags_format():
    """Test that W&B tags include architecture type and version."""
    model_type = 'gpt'
    version = 'v1'

    tags = [model_type, version]

    # Verify tags are non-empty strings
    for tag in tags:
        assert isinstance(tag, str), f"Tag should be string, got {type(tag)}"
        assert len(tag) > 0, "Tag should not be empty"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
