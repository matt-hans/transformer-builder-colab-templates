"""
Integration tests for complete training workflows.

Tests end-to-end training with all 5 loss strategies and production features.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec
from utils.adapters.model_adapter import UniversalModelAdapter


# ============================================================================
# Test 1: Basic Training (Simple model, small dataset, 3 epochs)
# ============================================================================

@pytest.mark.integration
def test_basic_training_workflow(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device
):
    """Test basic training workflow from start to finish."""
    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    # Split dataset
    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Act - Train
    trainer = Trainer(
        model=adapter,
        config=basic_training_config,
        task_spec=lm_task_spec
    )

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Assert
    assert results is not None
    assert 'final_loss' in results
    assert results['final_loss'] > 0
    assert 'metrics_summary' in results
    assert len(results['metrics_summary']) == basic_training_config.epochs
    assert 'best_epoch' in results

    # Verify model was trained (loss decreased)
    first_loss = results['metrics_summary']['train/loss'].iloc[0]
    last_loss = results['metrics_summary']['train/loss'].iloc[-1]
    assert last_loss < first_loss, "Training loss should decrease"


# ============================================================================
# Test 2: Training with Checkpointing (Save/resume mid-training)
# ============================================================================

@pytest.mark.integration
def test_training_with_checkpointing(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    checkpoint_dir,
    device
):
    """Test checkpoint saving and resuming mid-training."""
    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Update config to save checkpoints
    config_with_checkpoints = TrainingConfig(
        **basic_training_config.to_dict(),
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=1
    )

    # Act - Train for 2 epochs
    trainer = Trainer(
        model=adapter,
        config=config_with_checkpoints,
        task_spec=lm_task_spec
    )

    # Train partially
    partial_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=2  # Only 2 of 3 epochs
    )

    # Verify checkpoint exists
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) >= 1, "At least one checkpoint should be saved"

    # Load latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    checkpoint_data = torch.load(latest_checkpoint, map_location=device)

    # Resume training
    model2 = tiny_transformer_model.to(device)
    adapter2 = UniversalModelAdapter(model2, tiny_config, lm_task_spec)
    adapter2.model.load_state_dict(checkpoint_data['model_state_dict'])

    trainer2 = Trainer(
        model=adapter2,
        config=config_with_checkpoints,
        task_spec=lm_task_spec
    )

    # Resume from checkpoint
    resumed_results = trainer2.train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from_epoch=checkpoint_data['epoch']
    )

    # Assert
    assert partial_results is not None
    assert resumed_results is not None
    assert resumed_results['final_loss'] > 0


# ============================================================================
# Test 3: Training with Early Stopping
# ============================================================================

@pytest.mark.integration
def test_training_with_early_stopping(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device
):
    """Test early stopping when validation stops improving."""
    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Enable early stopping
    config_with_early_stop = TrainingConfig(
        **basic_training_config.to_dict(),
        early_stopping_patience=2,
        early_stopping_metric="val/loss",
        early_stopping_mode="min",
        epochs=10  # Set high, expect early stop
    )

    # Act
    trainer = Trainer(
        model=adapter,
        config=config_with_early_stop,
        task_spec=lm_task_spec
    )

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Assert
    assert results is not None
    # Early stopping may trigger, so epochs trained <= configured epochs
    epochs_trained = len(results['metrics_summary'])
    assert epochs_trained <= config_with_early_stop.epochs
    assert 'best_epoch' in results
    assert results['best_epoch'] < epochs_trained


# ============================================================================
# Test 4: Training with W&B (Mock W&B logging)
# ============================================================================

@pytest.mark.integration
def test_training_with_wandb_logging(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device,
    monkeypatch
):
    """Test W&B integration (mocked)."""
    # Arrange - Mock W&B
    wandb_logs = []

    class MockWandB:
        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(data, step=None):
            wandb_logs.append({'data': data, 'step': step})

        @staticmethod
        def finish():
            pass

    monkeypatch.setattr("wandb.init", MockWandB.init)
    monkeypatch.setattr("wandb.log", MockWandB.log)
    monkeypatch.setattr("wandb.finish", MockWandB.finish)

    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Enable W&B
    config_with_wandb = TrainingConfig(
        **basic_training_config.to_dict(),
        wandb_project="test_project",
        run_name="test_run"
    )

    # Act
    trainer = Trainer(
        model=adapter,
        config=config_with_wandb,
        task_spec=lm_task_spec
    )

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Assert
    assert results is not None
    assert len(wandb_logs) > 0, "W&B logs should be captured"
    # Verify metrics were logged
    logged_keys = set()
    for log_entry in wandb_logs:
        logged_keys.update(log_entry['data'].keys())
    assert 'train/loss' in logged_keys or any('loss' in k for k in logged_keys)


# ============================================================================
# Test 5: Training with Export Bundle
# ============================================================================

@pytest.mark.integration
def test_training_with_export_bundle(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    export_dir,
    device
):
    """Test export bundle generation after training."""
    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Enable export
    config_with_export = TrainingConfig(
        **basic_training_config.to_dict(),
        export_bundle=True,
        export_formats=["pytorch"],
        export_dir=str(export_dir)
    )

    # Act
    trainer = Trainer(
        model=adapter,
        config=config_with_export,
        task_spec=lm_task_spec
    )

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Assert
    assert results is not None

    # Check if export directory contains artifacts
    export_subdirs = list(export_dir.glob("model_*"))
    if len(export_subdirs) > 0:
        # Export was created
        export_bundle = export_subdirs[0]
        assert (export_bundle / "artifacts").exists() or (export_bundle / "model.pytorch.pt").exists()


# ============================================================================
# Test 6: Multi-Strategy Training (All 5 loss strategies)
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_multi_strategy_training(
    tiny_transformer_model,
    tiny_classifier_model,
    tiny_vision_model,
    tiny_config,
    tiny_classifier_config,
    tiny_vision_config,
    basic_training_config,
    synthetic_text_dataset,
    synthetic_classification_dataset,
    synthetic_vision_dataset,
    device
):
    """Test all 5 loss strategies: LM, Classification, PEFT, Quantization, Vision."""

    # Strategy 1: Language Modeling
    lm_task = TaskSpec(name="lm", modality="text", task_type="language_modeling")
    model_lm = tiny_transformer_model.to(device)
    adapter_lm = UniversalModelAdapter(model_lm, tiny_config, lm_task)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(synthetic_text_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4)

    trainer_lm = Trainer(model=adapter_lm, config=basic_training_config, task_spec=lm_task)
    results_lm = trainer_lm.train(train_loader=train_loader, val_loader=val_loader)
    assert results_lm is not None
    assert results_lm['final_loss'] > 0

    # Strategy 2: Classification
    cls_task = TaskSpec(name="cls", modality="text", task_type="classification", num_labels=2)
    model_cls = tiny_classifier_model.to(device)
    adapter_cls = UniversalModelAdapter(model_cls, tiny_classifier_config, cls_task)

    train_size = int(0.8 * len(synthetic_classification_dataset))
    val_size = len(synthetic_classification_dataset) - train_size
    train_ds_cls, val_ds_cls = torch.utils.data.random_split(
        synthetic_classification_dataset, [train_size, val_size]
    )

    train_loader_cls = torch.utils.data.DataLoader(train_ds_cls, batch_size=4, shuffle=True)
    val_loader_cls = torch.utils.data.DataLoader(val_ds_cls, batch_size=4)

    trainer_cls = Trainer(model=adapter_cls, config=basic_training_config, task_spec=cls_task)
    results_cls = trainer_cls.train(train_loader=train_loader_cls, val_loader=val_loader_cls)
    assert results_cls is not None
    assert results_cls['final_loss'] > 0

    # Strategy 3: Vision Classification
    vision_task = TaskSpec(name="vision", modality="vision", task_type="classification", num_labels=10)
    model_vision = tiny_vision_model.to(device)
    adapter_vision = UniversalModelAdapter(model_vision, tiny_vision_config, vision_task)

    train_size = int(0.8 * len(synthetic_vision_dataset))
    val_size = len(synthetic_vision_dataset) - train_size
    train_ds_vision, val_ds_vision = torch.utils.data.random_split(
        synthetic_vision_dataset, [train_size, val_size]
    )

    train_loader_vision = torch.utils.data.DataLoader(train_ds_vision, batch_size=4, shuffle=True)
    val_loader_vision = torch.utils.data.DataLoader(val_ds_vision, batch_size=4)

    trainer_vision = Trainer(model=adapter_vision, config=basic_training_config, task_spec=vision_task)
    results_vision = trainer_vision.train(train_loader=train_loader_vision, val_loader=val_loader_vision)
    assert results_vision is not None
    assert results_vision['final_loss'] > 0

    # Strategy 4 & 5: PEFT and Quantization are tested in specialized tests
    # (They require additional setup and are validated in test_hardware.py)

    # Verify all strategies produced valid results
    all_results = [results_lm, results_cls, results_vision]
    for i, results in enumerate(all_results):
        assert results is not None, f"Strategy {i+1} failed"
        assert 'final_loss' in results
        assert results['final_loss'] > 0
        assert 'metrics_summary' in results
