"""
Test suite for ModelRegistry.

Tests cover:
- Model registration with validation
- Version management and semantic versioning
- Tag-based organization (production, staging, experimental)
- Model lineage tracking
- Export format tracking
- Query and filtering operations
- Metrics comparison
- Error handling and edge cases
- Integration with CheckpointManager

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn

from utils.training.model_registry import ModelRegistry, ModelRegistryEntry


# Test fixtures

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_registry.db"
        yield db_path


@pytest.fixture
def registry(temp_db):
    """Create ModelRegistry instance."""
    return ModelRegistry(temp_db)


@pytest.fixture
def temp_checkpoint():
    """Create temporary checkpoint file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pt"

        # Create dummy checkpoint
        model = nn.Linear(10, 10)
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

        yield checkpoint_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample model configuration."""
    return {
        "vocab_size": 50257,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_seq_len": 128
    }


# Test model registration

def test_register_model_basic(registry, temp_checkpoint, sample_config):
    """Test basic model registration."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42, "perplexity": 1.52}
    )

    assert model_id > 0

    # Verify model was saved
    model = registry.get_model(model_id=model_id)
    assert model is not None
    assert model['name'] == "test-model"
    assert model['version'] == "1.0.0"
    assert model['metrics']['val_loss'] == 0.42


def test_register_model_with_metadata(registry, temp_checkpoint, sample_config):
    """Test registration with full metadata."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="classification",
        config_hash=config_hash,
        metrics={"val_loss": 0.38, "accuracy": 0.92},
        export_formats=["onnx", "torchscript"],
        model_size_mb=256.5,
        memory_req_gb=4.0,
        training_run_id=42,
        metadata={"notes": "baseline model", "epochs": 10},
        tags=["baseline", "experimental"]
    )

    model = registry.get_model(model_id=model_id)
    assert model['model_size_mb'] == 256.5
    assert model['memory_req_gb'] == 4.0
    assert model['training_run_id'] == 42
    assert model['metadata']['notes'] == "baseline model"
    assert set(model['export_formats']) == {"onnx", "torchscript"}


def test_register_duplicate_version(registry, temp_checkpoint, sample_config):
    """Test that duplicate name+version raises error."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    # Register first model
    registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    # Attempt to register duplicate
    with pytest.raises(ValueError, match="already exists"):
        registry.register_model(
            name="test-model",
            version="1.0.0",
            checkpoint_path=temp_checkpoint,
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={"val_loss": 0.40}
        )


def test_register_nonexistent_checkpoint(registry, sample_config):
    """Test registration with nonexistent checkpoint."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    with pytest.raises(FileNotFoundError):
        registry.register_model(
            name="test-model",
            version="1.0.0",
            checkpoint_path="/nonexistent/path.pt",
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={"val_loss": 0.42}
        )


# Test model retrieval

def test_get_model_by_id(registry, temp_checkpoint, sample_config):
    """Test retrieval by model ID."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    model = registry.get_model(model_id=model_id)
    assert model is not None
    assert model['model_id'] == model_id
    assert model['name'] == "test-model"


def test_get_model_by_name_version(registry, temp_checkpoint, sample_config):
    """Test retrieval by name and version."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    model = registry.get_model(name="test-model", version="1.0.0")
    assert model is not None
    assert model['name'] == "test-model"
    assert model['version'] == "1.0.0"


def test_get_model_by_tag(registry, temp_checkpoint, sample_config):
    """Test retrieval by tag."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42},
        tags=["production"]
    )

    model = registry.get_model(tag="production")
    assert model is not None
    assert model['model_id'] == model_id


def test_get_nonexistent_model(registry):
    """Test retrieval of nonexistent model."""
    model = registry.get_model(model_id=999)
    assert model is None


# Test tag management

def test_promote_model(registry, temp_checkpoint, sample_config):
    """Test model promotion to tag."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    registry.promote_model(model_id, "production")

    model = registry.get_model(tag="production")
    assert model is not None
    assert model['model_id'] == model_id


def test_promote_model_removes_from_others(registry, temp_checkpoint, sample_config):
    """Test that promotion removes tag from other models."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    # Register two models
    model_id_1 = registry.register_model(
        name="model-v1",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.50},
        tags=["production"]
    )

    model_id_2 = registry.register_model(
        name="model-v2",
        version="2.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    # Promote second model (should remove tag from first)
    registry.promote_model(model_id_2, "production", remove_from_others=True)

    # Check that tag points to second model
    model = registry.get_model(tag="production")
    assert model['model_id'] == model_id_2


def test_promote_model_keep_others(registry, temp_checkpoint, sample_config):
    """Test promotion without removing tag from others."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id_1 = registry.register_model(
        name="model-v1",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.50},
        tags=["staging"]
    )

    model_id_2 = registry.register_model(
        name="model-v2",
        version="2.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    # Add staging tag without removing from others
    registry.promote_model(model_id_2, "staging", remove_from_others=False)

    # Both models should have staging tag
    models = registry.list_models(tag="staging")
    assert len(models) == 2


# Test model lifecycle

def test_retire_model(registry, temp_checkpoint, sample_config):
    """Test model retirement."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    registry.retire_model(model_id)

    model = registry.get_model(model_id=model_id)
    assert model['status'] == 'retired'


def test_delete_model(registry, temp_checkpoint, sample_config):
    """Test model deletion."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    registry.delete_model(model_id)

    model = registry.get_model(model_id=model_id)
    assert model is None


def test_delete_tagged_model_fails(registry, temp_checkpoint, sample_config):
    """Test that deleting tagged model without force fails."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42},
        tags=["production"]
    )

    with pytest.raises(ValueError, match="has .* tag"):
        registry.delete_model(model_id, force=False)


def test_delete_tagged_model_with_force(registry, temp_checkpoint, sample_config):
    """Test forced deletion of tagged model."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42},
        tags=["production"]
    )

    registry.delete_model(model_id, force=True)

    model = registry.get_model(model_id=model_id)
    assert model is None


# Test listing and filtering

def test_list_models(registry, temp_checkpoint, sample_config):
    """Test listing all models."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    # Register multiple models
    for i in range(3):
        registry.register_model(
            name=f"model-{i}",
            version="1.0.0",
            checkpoint_path=temp_checkpoint,
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={"val_loss": 0.40 + i * 0.05}
        )

    models = registry.list_models()
    assert len(models) >= 3


def test_list_models_by_task_type(registry, temp_checkpoint, sample_config):
    """Test filtering by task type."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    registry.register_model(
        name="lm-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    registry.register_model(
        name="cls-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="classification",
        config_hash=config_hash,
        metrics={"val_loss": 0.35}
    )

    lm_models = registry.list_models(task_type="language_modeling")
    assert len(lm_models) == 1
    assert lm_models.iloc[0]['name'] == "lm-model"


def test_list_models_by_tag(registry, temp_checkpoint, sample_config):
    """Test filtering by tag."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    registry.register_model(
        name="prod-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42},
        tags=["production"]
    )

    registry.register_model(
        name="exp-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.45},
        tags=["experimental"]
    )

    prod_models = registry.list_models(tag="production")
    assert len(prod_models) == 1
    assert prod_models.iloc[0]['name'] == "prod-model"


def test_list_models_by_status(registry, temp_checkpoint, sample_config):
    """Test filtering by status."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="old-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.50}
    )

    registry.register_model(
        name="new-model",
        version="2.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    # Retire old model
    registry.retire_model(model_id)

    active_models = registry.list_models(status="active")
    assert len(active_models) == 1
    assert active_models.iloc[0]['name'] == "new-model"


# Test comparison

def test_compare_models(registry, temp_checkpoint, sample_config):
    """Test model comparison."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_ids = []
    for i in range(3):
        model_id = registry.register_model(
            name=f"model-v{i+1}",
            version=f"{i+1}.0.0",
            checkpoint_path=temp_checkpoint,
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={
                "val_loss": 0.50 - i * 0.05,
                "perplexity": 1.65 - i * 0.05
            }
        )
        model_ids.append(model_id)

    comparison = registry.compare_models(model_ids)

    assert len(comparison) == 3
    assert 'val_loss' in comparison.columns
    assert 'perplexity' in comparison.columns
    assert comparison.iloc[0]['val_loss'] == 0.50
    assert comparison.iloc[2]['val_loss'] == 0.40


def test_compare_models_specific_metrics(registry, temp_checkpoint, sample_config):
    """Test comparison with specific metrics."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_ids = []
    for i in range(2):
        model_id = registry.register_model(
            name=f"model-{i}",
            version="1.0.0",
            checkpoint_path=temp_checkpoint,
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={
                "val_loss": 0.42,
                "train_loss": 0.40,
                "perplexity": 1.52
            }
        )
        model_ids.append(model_id)

    comparison = registry.compare_models(model_ids, metrics=["val_loss"])

    assert 'val_loss' in comparison.columns
    assert 'perplexity' not in comparison.columns


# Test lineage tracking

def test_model_lineage(registry, temp_checkpoint, sample_config):
    """Test model lineage tracking."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    # Create parent model
    parent_id = registry.register_model(
        name="base-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.50}
    )

    # Create child model
    child_id = registry.register_model(
        name="finetuned-model",
        version="1.1.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42},
        parent_model_id=parent_id
    )

    # Create grandchild model
    grandchild_id = registry.register_model(
        name="specialized-model",
        version="1.2.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.38},
        parent_model_id=child_id
    )

    lineage = registry.get_model_lineage(grandchild_id)

    assert len(lineage) == 3
    assert lineage[0]['name'] == "base-model"
    assert lineage[1]['name'] == "finetuned-model"
    assert lineage[2]['name'] == "specialized-model"


# Test export format tracking

def test_add_export_format(registry, temp_checkpoint, sample_config):
    """Test adding export format."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    # Add ONNX export
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        onnx_path.touch()

        registry.add_export_format(
            model_id=model_id,
            export_format="onnx",
            export_path=onnx_path,
            metadata={"opset_version": 14}
        )

    model = registry.get_model(model_id=model_id)
    assert "onnx" in model['export_formats']


def test_add_multiple_export_formats(registry, temp_checkpoint, sample_config):
    """Test adding multiple export formats."""
    config_hash = ModelRegistry.compute_config_hash(sample_config)

    model_id = registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Add multiple formats
        for fmt in ["onnx", "torchscript"]:
            path = Path(tmpdir) / f"model.{fmt}"
            path.touch()
            registry.add_export_format(model_id, fmt, path)

    model = registry.get_model(model_id=model_id)
    assert set(model['export_formats']) == {"onnx", "torchscript"}


# Test config hash

def test_compute_config_hash_deterministic():
    """Test that config hash is deterministic."""
    config = {"d_model": 768, "num_layers": 12, "vocab_size": 50257}

    hash1 = ModelRegistry.compute_config_hash(config)
    hash2 = ModelRegistry.compute_config_hash(config)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 produces 64 hex characters


def test_compute_config_hash_order_independent():
    """Test that config hash is order-independent."""
    config1 = {"d_model": 768, "num_layers": 12}
    config2 = {"num_layers": 12, "d_model": 768}

    hash1 = ModelRegistry.compute_config_hash(config1)
    hash2 = ModelRegistry.compute_config_hash(config2)

    assert hash1 == hash2


def test_compute_config_hash_sensitive_to_values():
    """Test that config hash changes with values."""
    config1 = {"d_model": 768}
    config2 = {"d_model": 512}

    hash1 = ModelRegistry.compute_config_hash(config1)
    hash2 = ModelRegistry.compute_config_hash(config2)

    assert hash1 != hash2


# Performance tests

def test_registry_query_performance(registry, temp_checkpoint, sample_config):
    """Test query performance (<10ms for queries)."""
    import time

    config_hash = ModelRegistry.compute_config_hash(sample_config)

    # Register 100 models
    model_ids = []
    for i in range(100):
        model_id = registry.register_model(
            name=f"model-{i}",
            version="1.0.0",
            checkpoint_path=temp_checkpoint,
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={"val_loss": 0.40 + i * 0.001}
        )
        model_ids.append(model_id)

    # Test query performance
    start = time.time()
    models = registry.list_models(limit=50)
    query_time = (time.time() - start) * 1000  # Convert to ms

    assert query_time < 10  # Should be <10ms
    assert len(models) == 50


def test_registry_write_performance(registry, temp_checkpoint, sample_config):
    """Test write performance (<100ms for writes)."""
    import time

    config_hash = ModelRegistry.compute_config_hash(sample_config)

    start = time.time()
    registry.register_model(
        name="test-model",
        version="1.0.0",
        checkpoint_path=temp_checkpoint,
        task_type="language_modeling",
        config_hash=config_hash,
        metrics={"val_loss": 0.42}
    )
    write_time = (time.time() - start) * 1000  # Convert to ms

    assert write_time < 100  # Should be <100ms


# Integration tests

def test_registry_with_checkpoint_manager(temp_checkpoint, sample_config):
    """Test integration with CheckpointManager."""
    from utils.training.engine.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create registry and checkpoint manager
        registry = ModelRegistry(Path(tmpdir) / "registry.db")
        checkpoint_mgr = CheckpointManager(
            checkpoint_dir=tmpdir,
            monitor="val_loss",
            mode="min"
        )

        # Simulate training checkpoint
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = checkpoint_mgr.save(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            metrics={"val_loss": 0.38, "train_loss": 0.42}
        )

        # Register model from checkpoint
        config_hash = ModelRegistry.compute_config_hash(sample_config)
        model_id = registry.register_model(
            name="trained-model",
            version="1.0.0",
            checkpoint_path=checkpoint_path,
            task_type="language_modeling",
            config_hash=config_hash,
            metrics={"val_loss": 0.38, "train_loss": 0.42}
        )

        # Verify registration
        model = registry.get_model(model_id=model_id)
        assert model is not None
        assert model['metrics']['val_loss'] == 0.38


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
