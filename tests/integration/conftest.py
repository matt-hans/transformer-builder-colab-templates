"""
Shared fixtures for integration tests.

Provides realistic models, datasets, and configurations for end-to-end testing.
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

from utils.training.task_spec import TaskSpec
from utils.training.training_config import TrainingConfig


# ============================================================================
# Model Fixtures (Small but realistic)
# ============================================================================

@pytest.fixture
def tiny_transformer_model():
    """Tiny transformer for fast integration tests (2 layers, d_model=64)."""
    class TinyTransformer(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64, num_layers=2, max_seq_len=32):
            super().__init__()
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_projection = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            B, T = input_ids.shape
            device = input_ids.device

            tok_emb = self.embedding(input_ids)
            pos_emb = self.pos_embedding(torch.arange(T, device=device))
            x = tok_emb + pos_emb.unsqueeze(0)

            x = self.transformer(x)
            logits = self.output_projection(x)
            return logits

    return TinyTransformer()


@pytest.fixture
def tiny_vision_model():
    """Tiny vision model for classification (2 conv layers)."""
    class TinyVisionModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.num_classes = num_classes
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, num_classes)

        def forward(self, pixel_values):
            x = torch.relu(self.conv1(pixel_values))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return TinyVisionModel()


@pytest.fixture
def tiny_classifier_model():
    """Tiny sequence classifier (encoder + classification head)."""
    class TinyClassifier(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64, num_classes=2):
            super().__init__()
            self.vocab_size = vocab_size
            self.num_classes = num_classes
            self.embedding = nn.Embedding(vocab_size, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(d_model, num_classes)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.encoder(x)
            # Mean pooling
            x = x.mean(dim=1)
            return self.classifier(x)

    return TinyClassifier()


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def tiny_config():
    """Config for tiny transformer model."""
    return SimpleNamespace(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        max_seq_len=32,
        pad_token_id=0
    )


@pytest.fixture
def tiny_vision_config():
    """Config for tiny vision model."""
    return SimpleNamespace(
        num_classes=10,
        image_size=32,
        num_channels=3
    )


@pytest.fixture
def tiny_classifier_config():
    """Config for tiny classifier model."""
    return SimpleNamespace(
        vocab_size=1000,
        d_model=64,
        num_classes=2,
        max_seq_len=32,
        pad_token_id=0
    )


# ============================================================================
# TaskSpec Fixtures
# ============================================================================

@pytest.fixture
def lm_task_spec():
    """Language modeling task spec."""
    return TaskSpec(
        name="tiny_lm",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "perplexity"]
    )


@pytest.fixture
def classification_task_spec():
    """Text classification task spec."""
    return TaskSpec(
        name="tiny_cls",
        task_type="text_classification",
        model_family="encoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"]
    )


@pytest.fixture
def vision_task_spec():
    """Vision classification task spec."""
    return TaskSpec(
        name="tiny_vision",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"]
    )


# ============================================================================
# TrainingConfig Fixtures
# ============================================================================

@pytest.fixture
def basic_training_config(tmp_path):
    """Basic training config for fast tests."""
    return TrainingConfig(
        # Fast training
        learning_rate=1e-3,
        batch_size=4,
        epochs=3,

        # Checkpointing
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_every_n_epochs=1,

        # Reproducibility
        random_seed=42,
        deterministic=False,  # Fast mode

        # Optimization
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,

        # Performance
        compile_mode=None,  # Disable for fast tests

        # Experiment
        run_name="integration_test",
        notes="Integration test run"
    )


@pytest.fixture
def production_training_config(tmp_path):
    """Production-like training config with all features enabled."""
    return TrainingConfig(
        # Training
        learning_rate=5e-5,
        batch_size=8,
        epochs=5,

        # Checkpointing
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_every_n_epochs=2,
        keep_last_n_checkpoints=2,

        # Early stopping
        early_stopping_patience=3,
        early_stopping_metric="val/loss",
        early_stopping_mode="min",

        # Reproducibility
        random_seed=42,
        deterministic=True,  # Bit-exact

        # Optimization
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,

        # Export
        export_bundle=True,
        export_formats=["pytorch"],
        export_dir=str(tmp_path / "exports"),

        # Experiment
        run_name="production_test",
        notes="Production integration test"
    )


# ============================================================================
# Dataset Fixtures (Synthetic)
# ============================================================================

@pytest.fixture
def synthetic_text_dataset():
    """Synthetic text dataset for testing."""
    class SyntheticTextDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, seq_len=32, vocab_size=1000):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate deterministic data based on index
            torch.manual_seed(idx)
            input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
            return {'input_ids': input_ids}

    return SyntheticTextDataset()


@pytest.fixture
def synthetic_vision_dataset():
    """Synthetic vision dataset for testing."""
    class SyntheticVisionDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, image_size=32, num_classes=10):
            self.num_samples = num_samples
            self.image_size = image_size
            self.num_classes = num_classes

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            pixel_values = torch.randn(3, self.image_size, self.image_size)
            labels = torch.tensor(idx % self.num_classes)
            return {'pixel_values': pixel_values, 'labels': labels}

    return SyntheticVisionDataset()


@pytest.fixture
def synthetic_classification_dataset():
    """Synthetic classification dataset for testing."""
    class SyntheticClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, seq_len=32, vocab_size=1000, num_classes=2):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size
            self.num_classes = num_classes

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            torch.manual_seed(idx)
            input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
            labels = torch.tensor(idx % self.num_classes)
            return {'input_ids': input_ids, 'labels': labels}

    return SyntheticClassificationDataset()


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def integration_tmp_dir():
    """Temporary directory for integration test artifacts."""
    tmp_dir = tempfile.mkdtemp(prefix="integration_test_")
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def checkpoint_dir(integration_tmp_dir):
    """Checkpoint directory."""
    checkpoint_path = integration_tmp_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


@pytest.fixture
def export_dir(integration_tmp_dir):
    """Export directory."""
    export_path = integration_tmp_dir / "exports"
    export_path.mkdir(parents=True, exist_ok=True)
    return export_path


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device (CPU or CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cuda_device():
    """Get CUDA device or skip test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ============================================================================
# Integration Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (slow, uses real data)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>30 seconds)"
    )
    config.addinivalue_line(
        "markers", "production: mark test as production workflow test"
    )
