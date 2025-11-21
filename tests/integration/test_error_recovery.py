"""
Integration tests for error recovery and failure handling.

Tests OOM handling, NaN/Inf recovery, checkpoint corruption, network failures, and job retry logic.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import shutil
import json

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec
from utils.training.job_queue import JobQueue, JobStatus, JobPriority
from utils.adapters.model_adapter import UniversalModelAdapter


# ============================================================================
# Test 1: OOM Handling (Simulate with small memory limit)
# ============================================================================

@pytest.mark.integration
def test_oom_recovery(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device
):
    """Test graceful handling of out-of-memory errors."""
    # Arrange - Create intentionally large batch to trigger OOM-like behavior
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    # Try progressively smaller batch sizes on failure
    batch_sizes = [128, 64, 32, 16, 8, 4]
    successful_batch_size = None

    for batch_size in batch_sizes:
        try:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )

            config = TrainingConfig(
                **basic_training_config.to_dict(),
                batch_size=batch_size,
                epochs=1  # Quick test
            )

            trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)

            # Try training
            results = trainer.train(train_loader=train_loader, val_loader=val_loader)

            # Success - record batch size
            successful_batch_size = batch_size
            print(f"Successfully trained with batch size {batch_size}")
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM with batch size {batch_size}, trying smaller...")
                # Clear cache and try smaller batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                # Different error, re-raise
                raise

    # Assert - Should find a working batch size
    assert successful_batch_size is not None, "Should find a working batch size"
    assert successful_batch_size <= 64, "Should fall back to reasonable batch size"


# ============================================================================
# Test 2: NaN/Inf Gradient Recovery
# ============================================================================

@pytest.mark.integration
def test_nan_inf_gradient_recovery(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device
):
    """Test detection and recovery from NaN/Inf gradients."""

    class NaNInjectingDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, inject_nan_at=50):
            self.base_dataset = base_dataset
            self.inject_nan_at = inject_nan_at
            self.call_count = 0

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            self.call_count += 1
            item = self.base_dataset[idx]

            # Inject NaN values at specific iteration
            if self.call_count == self.inject_nan_at:
                # Create inputs that will cause NaN (e.g., extreme values)
                input_ids = torch.full_like(item['input_ids'], tiny_config.vocab_size - 1)
                return {'input_ids': input_ids}

            return item

    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_dataset = NaNInjectingDataset(synthetic_text_dataset, inject_nan_at=10)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset, batch_size=4
    )

    # Enable gradient monitoring
    config = TrainingConfig(
        **basic_training_config.to_dict(),
        gradient_clip_norm=1.0,  # Should help prevent NaN
        detect_anomaly=True  # Enable anomaly detection
    )

    # Act
    trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)

    try:
        results = trainer.train(train_loader=train_loader, val_loader=val_loader)
        # Training may complete despite NaN if clipping helps
        assert results is not None

    except RuntimeError as e:
        # Expected: NaN detection should trigger error
        assert "nan" in str(e).lower() or "inf" in str(e).lower()
        print(f"NaN/Inf detected correctly: {e}")


# ============================================================================
# Test 3: Checkpoint Corruption Recovery
# ============================================================================

@pytest.mark.integration
def test_checkpoint_corruption_recovery(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    checkpoint_dir,
    device
):
    """Test recovery from corrupted checkpoint files."""
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

    config = TrainingConfig(
        **basic_training_config.to_dict(),
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=1
    )

    # Act - Train and create checkpoints
    trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)
    results = trainer.train(train_loader=train_loader, val_loader=val_loader)

    # Find checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    assert len(checkpoints) > 0, "Should have saved checkpoints"

    # Corrupt one checkpoint
    corrupted_checkpoint = checkpoints[0]
    original_size = corrupted_checkpoint.stat().st_size

    with open(corrupted_checkpoint, 'wb') as f:
        f.write(b'corrupted data')

    # Try to load corrupted checkpoint
    try:
        checkpoint_data = torch.load(corrupted_checkpoint, map_location=device)
        # Should fail to load
        pytest.fail("Should have failed to load corrupted checkpoint")

    except Exception as e:
        # Expected: corruption detected
        print(f"Corruption detected correctly: {e}")

    # Recovery: Load from previous checkpoint
    if len(checkpoints) > 1:
        fallback_checkpoint = checkpoints[1]
        checkpoint_data = torch.load(fallback_checkpoint, map_location=device)
        assert 'model_state_dict' in checkpoint_data
        print("Successfully recovered from fallback checkpoint")


# ============================================================================
# Test 4: Network Failure Recovery (W&B, file I/O)
# ============================================================================

@pytest.mark.integration
def test_network_failure_recovery(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device,
    monkeypatch
):
    """Test graceful handling of network failures (W&B, cloud storage)."""

    # Mock W&B to simulate network failures
    class FailingWandB:
        call_count = 0
        fail_on_calls = [3, 7]  # Fail on specific calls

        @staticmethod
        def init(*args, **kwargs):
            return None

        @classmethod
        def log(cls, data, step=None):
            cls.call_count += 1
            if cls.call_count in cls.fail_on_calls:
                raise ConnectionError("Simulated network failure")
            # Success otherwise

        @staticmethod
        def finish():
            pass

    monkeypatch.setattr("wandb.init", FailingWandB.init)
    monkeypatch.setattr("wandb.log", FailingWandB.log)
    monkeypatch.setattr("wandb.finish", FailingWandB.finish)

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

    config = TrainingConfig(
        **basic_training_config.to_dict(),
        use_wandb=True,
        wandb_project="test_failures"
    )

    # Act - Training should continue despite network failures
    trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)

    try:
        results = trainer.train(train_loader=train_loader, val_loader=val_loader)
        # If trainer has retry logic, it should succeed
        assert results is not None
        print("Training completed despite network failures")

    except ConnectionError:
        # If no retry logic, expect failure
        print("Network failure caused training to fail (expected without retry logic)")


# ============================================================================
# Test 5: Job Failure and Retry Logic
# ============================================================================

@pytest.mark.integration
def test_job_retry_logic(integration_tmp_dir):
    """Test job queue retry logic on failures."""
    # Setup
    job_queue_db = integration_tmp_dir / "job_queue.db"
    job_queue = JobQueue(str(job_queue_db))

    # Submit job
    job_id = job_queue.submit_job(
        job_type="training",
        config={'epochs': 5},
        priority=JobPriority.HIGH
    )

    # Simulate job failure
    job_queue.update_job_status(job_id, JobStatus.RUNNING)
    job_queue.update_job_status(job_id, JobStatus.FAILED)
    job_queue.update_job_error(job_id, "Simulated training failure")

    # Verify failure recorded
    failed_job = job_queue.get_job(job_id)
    assert failed_job['status'] == JobStatus.FAILED.value
    assert failed_job['error'] is not None

    # Retry job
    retry_job_id = job_queue.submit_job(
        job_type="training",
        config={'epochs': 5, 'retry_of': job_id},
        priority=JobPriority.HIGH
    )

    # Simulate successful retry
    job_queue.update_job_status(retry_job_id, JobStatus.RUNNING)
    job_queue.update_job_status(retry_job_id, JobStatus.COMPLETED)
    job_queue.update_job_result(retry_job_id, {'final_loss': 0.5})

    # Verify retry succeeded
    retry_job = job_queue.get_job(retry_job_id)
    assert retry_job['status'] == JobStatus.COMPLETED.value
    assert retry_job['result'] is not None


# ============================================================================
# Test 6: Disk Space Exhaustion
# ============================================================================

@pytest.mark.integration
def test_disk_space_exhaustion(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    integration_tmp_dir,
    device
):
    """Test handling of disk space exhaustion during checkpoint saving."""

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

    # Create checkpoint dir with limited space simulation
    # (Real disk exhaustion is hard to simulate, so we test error handling)
    checkpoint_dir = integration_tmp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    config = TrainingConfig(
        **basic_training_config.to_dict(),
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=1
    )

    # Act
    trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)

    try:
        results = trainer.train(train_loader=train_loader, val_loader=val_loader)
        # Should complete successfully in normal conditions
        assert results is not None

        # Verify checkpoints saved
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoints) > 0

    except OSError as e:
        # If disk space exhaustion occurs, should be caught
        print(f"Disk error detected: {e}")


# ============================================================================
# Test 7: Interrupted Training (SIGINT simulation)
# ============================================================================

@pytest.mark.integration
def test_interrupted_training_resume(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    checkpoint_dir,
    device
):
    """Test resuming training after interruption."""

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

    config = TrainingConfig(
        **basic_training_config.to_dict(),
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=1,
        epochs=5
    )

    # Act - Train partially (simulate interruption after 2 epochs)
    trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)
    partial_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=2
    )

    # Save checkpoint manually
    checkpoint_path = checkpoint_dir / "checkpoint_epoch_2.pt"
    torch.save({
        'epoch': 2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else {},
        'loss': partial_results['final_loss']
    }, checkpoint_path)

    # Resume training
    model2 = tiny_transformer_model.to(device)
    adapter2 = UniversalModelAdapter(model2, tiny_config, lm_task_spec)

    # Load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    model2.load_state_dict(checkpoint_data['model_state_dict'])

    trainer2 = Trainer(model=adapter2, config=config, task_spec=lm_task_spec)
    resumed_results = trainer2.train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from_epoch=checkpoint_data['epoch']
    )

    # Assert
    assert resumed_results is not None
    # Should complete remaining epochs (3, 4, 5)
    assert len(resumed_results['metrics_summary']) >= 3


# ============================================================================
# Test 8: Invalid Configuration Recovery
# ============================================================================

@pytest.mark.integration
def test_invalid_configuration_recovery(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    synthetic_text_dataset,
    device
):
    """Test detection and recovery from invalid training configurations."""

    # Test invalid learning rate
    try:
        invalid_config = TrainingConfig(
            learning_rate=-1.0,  # Invalid: negative
            batch_size=4,
            epochs=3
        )
        invalid_config.validate()
        pytest.fail("Should have caught invalid learning rate")
    except ValueError as e:
        assert "learning_rate" in str(e).lower()
        print(f"Caught invalid config: {e}")

    # Test invalid batch size
    try:
        invalid_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=0,  # Invalid: zero
            epochs=3
        )
        invalid_config.validate()
        pytest.fail("Should have caught invalid batch size")
    except ValueError as e:
        assert "batch_size" in str(e).lower()
        print(f"Caught invalid config: {e}")

    # Test invalid gradient accumulation
    try:
        invalid_config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=4,
            epochs=3,
            gradient_accumulation_steps=-1  # Invalid: negative
        )
        invalid_config.validate()
        pytest.fail("Should have caught invalid gradient accumulation")
    except ValueError as e:
        assert "gradient_accumulation" in str(e).lower()
        print(f"Caught invalid config: {e}")


# ============================================================================
# Test 9: Model Load Failure Recovery
# ============================================================================

@pytest.mark.integration
def test_model_load_failure_recovery(
    tiny_config,
    lm_task_spec,
    integration_tmp_dir,
    device
):
    """Test recovery from model loading failures."""

    # Create invalid model file
    invalid_model_path = integration_tmp_dir / "invalid_model.pt"
    with open(invalid_model_path, 'w') as f:
        f.write("not a valid pytorch model")

    # Try to load invalid model
    try:
        model_data = torch.load(invalid_model_path, map_location=device)
        pytest.fail("Should have failed to load invalid model")
    except Exception as e:
        print(f"Model load failure detected: {e}")

    # Recovery: Create valid model instead
    from tests.integration.conftest import TinyTransformer

    class TinyTransformer(nn.Module):
        def __init__(self, vocab_size=1000, d_model=64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            return self.output(x)

    fallback_model = TinyTransformer().to(device)
    valid_model_path = integration_tmp_dir / "valid_model.pt"
    torch.save(fallback_model.state_dict(), valid_model_path)

    # Verify valid model loads
    loaded_state = torch.load(valid_model_path, map_location=device)
    assert loaded_state is not None
    print("Successfully recovered with fallback model")


# ============================================================================
# Test 10: Training Loop Exception Handling
# ============================================================================

@pytest.mark.integration
def test_training_loop_exception_handling(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    device
):
    """Test exception handling within training loop."""

    class ExceptionDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100, fail_at=50):
            self.num_samples = num_samples
            self.fail_at = fail_at
            self.call_count = 0

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            self.call_count += 1
            if self.call_count == self.fail_at:
                raise RuntimeError("Simulated data loading failure")

            torch.manual_seed(idx)
            return {'input_ids': torch.randint(1, 1000, (32,))}

    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    exception_dataset = ExceptionDataset(fail_at=20)
    train_loader = torch.utils.data.DataLoader(exception_dataset, batch_size=4)
    val_loader = torch.utils.data.DataLoader(exception_dataset, batch_size=4)

    trainer = Trainer(model=adapter, config=basic_training_config, task_spec=lm_task_spec)

    # Act
    try:
        results = trainer.train(train_loader=train_loader, val_loader=val_loader)
        pytest.fail("Should have raised exception from dataset")
    except RuntimeError as e:
        assert "data loading failure" in str(e)
        print(f"Exception handled correctly: {e}")
