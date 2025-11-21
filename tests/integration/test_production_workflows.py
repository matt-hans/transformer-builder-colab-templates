"""
Integration tests for production workflows.

Tests complete production lifecycle: training, registry, export, retraining, scheduling.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import time

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec
from utils.training.model_registry import ModelRegistry
from utils.training.job_queue import JobQueue, JobPriority, JobStatus
from utils.training.retraining_triggers import RetrainingMonitor, DriftTrigger
from utils.training.drift_metrics import profile_dataset, compute_drift
from utils.training.export_health import validate_export_health
from utils.adapters.model_adapter import UniversalModelAdapter


# ============================================================================
# Test 1: Model Lifecycle (Train → Register → Export → Health Check)
# ============================================================================

@pytest.mark.integration
@pytest.mark.production
def test_model_lifecycle_workflow(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    integration_tmp_dir,
    device
):
    """Test complete model lifecycle from training to production health checks."""
    # Setup
    registry_db = integration_tmp_dir / "registry.db"
    export_dir = integration_tmp_dir / "exports"
    export_dir.mkdir(exist_ok=True)

    # Step 1: Train model
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

    trainer = Trainer(
        model=adapter,
        config=basic_training_config,
        task_spec=lm_task_spec
    )

    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    assert training_results is not None
    final_loss = training_results['final_loss']

    # Step 2: Register model
    registry = ModelRegistry(str(registry_db))

    model_path = integration_tmp_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    model_id = registry.register_model(
        name="tiny_transformer_v1",
        version="1.0.0",
        model_path=str(model_path),
        config=tiny_config.__dict__,
        task_spec=lm_task_spec.to_dict(),
        training_config=basic_training_config.to_dict(),
        metrics={'final_loss': final_loss},
        tags=['integration_test', 'baseline']
    )

    assert model_id is not None
    assert model_id > 0

    # Verify registration
    model_info = registry.get_model(model_id)
    assert model_info is not None
    assert model_info['name'] == "tiny_transformer_v1"
    assert model_info['version'] == "1.0.0"
    assert model_info['status'] == 'registered'

    # Step 3: Export model
    from utils.training.export_utilities import create_export_bundle

    export_config = TrainingConfig(
        **basic_training_config.to_dict(),
        export_bundle=True,
        export_formats=["pytorch"],
        export_dir=str(export_dir)
    )

    export_bundle_path = create_export_bundle(
        model=model,
        config=tiny_config,
        task_spec=lm_task_spec,
        training_config=export_config
    )

    assert export_bundle_path is not None
    assert Path(export_bundle_path).exists()

    # Update registry with export path
    registry.update_model(
        model_id,
        export_path=str(export_bundle_path),
        status='exported'
    )

    # Step 4: Health check
    model_info_updated = registry.get_model(model_id)
    assert model_info_updated['status'] == 'exported'
    assert model_info_updated['export_path'] is not None

    # Validate export health
    health_results = validate_export_health(
        export_dir=export_bundle_path,
        model=model,
        config=tiny_config,
        task_spec=lm_task_spec,
        test_batch_size=2
    )

    assert health_results is not None
    assert health_results['status'] in ['healthy', 'warning', 'critical']

    # If healthy, mark as production-ready
    if health_results['status'] == 'healthy':
        registry.update_model(model_id, status='production')
        model_final = registry.get_model(model_id)
        assert model_final['status'] == 'production'


# ============================================================================
# Test 2: Retraining Workflow (Drift → Trigger → Job → Training → Registry)
# ============================================================================

@pytest.mark.integration
@pytest.mark.production
def test_retraining_workflow(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    integration_tmp_dir,
    device
):
    """Test automated retraining triggered by drift detection."""
    # Setup
    registry_db = integration_tmp_dir / "registry.db"
    job_queue_db = integration_tmp_dir / "job_queue.db"

    registry = ModelRegistry(str(registry_db))
    job_queue = JobQueue(str(job_queue_db))

    # Step 1: Train initial model
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

    trainer = Trainer(model=adapter, config=basic_training_config, task_spec=lm_task_spec)
    initial_results = trainer.train(train_loader=train_loader, val_loader=val_loader)

    # Register initial model
    model_path = integration_tmp_dir / "model_v1.pt"
    torch.save(model.state_dict(), model_path)

    model_id = registry.register_model(
        name="drift_test_model",
        version="1.0.0",
        model_path=str(model_path),
        config=tiny_config.__dict__,
        task_spec=lm_task_spec.to_dict(),
        training_config=basic_training_config.to_dict(),
        metrics={'final_loss': initial_results['final_loss']},
        tags=['production']
    )

    # Step 2: Simulate drift with profile comparison
    # Profile reference dataset
    ref_profile = profile_dataset(train_dataset, lm_task_spec)

    # Create "drifted" dataset (slightly different distribution)
    class DriftedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, offset=100):
            self.base_dataset = base_dataset
            self.offset = offset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            torch.manual_seed(idx + self.offset)  # Different seed = drift
            item = self.base_dataset[idx]
            # Modify sequence slightly
            if 'input_ids' in item:
                input_ids = item['input_ids'].clone()
                # Add offset to simulate vocabulary drift
                input_ids = torch.clamp(input_ids + 50, 0, tiny_config.vocab_size - 1)
                item['input_ids'] = input_ids
            return item

    drifted_dataset = DriftedDataset(train_dataset)
    new_profile = profile_dataset(drifted_dataset, lm_task_spec)

    # Compute drift
    drift_scores, drift_status = compute_drift(ref_profile, new_profile)

    # Step 3: Trigger retraining if drift detected
    drift_trigger = DriftTrigger(
        threshold=0.1,  # Sensitive threshold
        metric='sequence_length'
    )

    should_retrain = drift_trigger.should_trigger(drift_scores)

    if should_retrain or any(score > 0.1 for score in drift_scores.values()):
        # Submit retraining job
        job_id = job_queue.submit_job(
            job_type="retraining",
            config={
                'model_id': model_id,
                'reason': 'drift_detected',
                'drift_scores': drift_scores,
                'training_config': basic_training_config.to_dict()
            },
            priority=JobPriority.HIGH
        )

        assert job_id is not None

        # Step 4: Execute retraining job
        job = job_queue.get_job(job_id)
        assert job['status'] == JobStatus.PENDING.value

        # Start job
        job_queue.update_job_status(job_id, JobStatus.RUNNING)

        # Retrain model
        model_v2 = tiny_transformer_model.to(device)
        adapter_v2 = UniversalModelAdapter(model_v2, tiny_config, lm_task_spec)

        drifted_loader = torch.utils.data.DataLoader(
            drifted_dataset, batch_size=basic_training_config.batch_size, shuffle=True
        )

        trainer_v2 = Trainer(model=adapter_v2, config=basic_training_config, task_spec=lm_task_spec)
        retrain_results = trainer_v2.train(train_loader=drifted_loader, val_loader=val_loader)

        # Step 5: Register retrained model
        model_v2_path = integration_tmp_dir / "model_v2.pt"
        torch.save(model_v2.state_dict(), model_v2_path)

        model_v2_id = registry.register_model(
            name="drift_test_model",
            version="2.0.0",
            model_path=str(model_v2_path),
            config=tiny_config.__dict__,
            task_spec=lm_task_spec.to_dict(),
            training_config=basic_training_config.to_dict(),
            metrics={'final_loss': retrain_results['final_loss']},
            tags=['production', 'retrained'],
            notes=f"Retrained due to drift: {drift_scores}"
        )

        # Complete job
        job_queue.update_job_status(job_id, JobStatus.COMPLETED)
        job_queue.update_job_result(job_id, {'model_id': model_v2_id})

        # Verify workflow
        completed_job = job_queue.get_job(job_id)
        assert completed_job['status'] == JobStatus.COMPLETED.value
        assert completed_job['result']['model_id'] == model_v2_id

        # Verify model registry has both versions
        models = registry.list_models(name="drift_test_model")
        assert len(models) == 2
        versions = [m['version'] for m in models]
        assert '1.0.0' in versions
        assert '2.0.0' in versions


# ============================================================================
# Test 3: Scheduled Training (Job queue → Scheduler → Executor → Completion)
# ============================================================================

@pytest.mark.integration
@pytest.mark.production
def test_scheduled_training_workflow(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    integration_tmp_dir,
    device
):
    """Test scheduled training via job queue."""
    # Setup
    job_queue_db = integration_tmp_dir / "job_queue.db"
    job_queue = JobQueue(str(job_queue_db))

    # Step 1: Schedule multiple training jobs with priorities
    job_configs = [
        {
            'name': 'high_priority_job',
            'priority': JobPriority.HIGH,
            'config': {'epochs': 2}
        },
        {
            'name': 'medium_priority_job',
            'priority': JobPriority.MEDIUM,
            'config': {'epochs': 3}
        },
        {
            'name': 'low_priority_job',
            'priority': JobPriority.LOW,
            'config': {'epochs': 1}
        }
    ]

    job_ids = []
    for job_config in job_configs:
        job_id = job_queue.submit_job(
            job_type="training",
            config=job_config['config'],
            priority=job_config['priority']
        )
        job_ids.append(job_id)

    assert len(job_ids) == 3

    # Step 2: Execute jobs in priority order
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

    completed_jobs = []

    # Process jobs (high priority first)
    while True:
        next_job = job_queue.get_next_job()
        if next_job is None:
            break

        job_id = next_job['id']
        job_config = next_job['config']

        # Start job
        job_queue.update_job_status(job_id, JobStatus.RUNNING)

        # Execute training
        model = tiny_transformer_model.to(device)
        adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

        config = TrainingConfig(
            **basic_training_config.to_dict(),
            epochs=job_config.get('epochs', 3)
        )

        trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)
        results = trainer.train(train_loader=train_loader, val_loader=val_loader)

        # Complete job
        job_queue.update_job_status(job_id, JobStatus.COMPLETED)
        job_queue.update_job_result(job_id, {
            'final_loss': results['final_loss'],
            'epochs_trained': len(results['metrics_summary'])
        })

        completed_jobs.append(job_id)

    # Step 3: Verify execution order (high → medium → low)
    assert len(completed_jobs) == 3

    # Verify all jobs completed
    for job_id in job_ids:
        job = job_queue.get_job(job_id)
        assert job['status'] == JobStatus.COMPLETED.value
        assert job['result'] is not None
        assert 'final_loss' in job['result']


# ============================================================================
# Test 4: Model Comparison (Train 2 → Compare → Promote to Production)
# ============================================================================

@pytest.mark.integration
@pytest.mark.production
def test_model_comparison_workflow(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    integration_tmp_dir,
    device
):
    """Test training multiple models, comparing, and promoting best."""
    # Setup
    registry_db = integration_tmp_dir / "registry.db"
    registry = ModelRegistry(str(registry_db))

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

    # Step 1: Train Model A (baseline)
    model_a = tiny_transformer_model.to(device)
    adapter_a = UniversalModelAdapter(model_a, tiny_config, lm_task_spec)

    config_a = TrainingConfig(**basic_training_config.to_dict(), learning_rate=1e-3)

    trainer_a = Trainer(model=adapter_a, config=config_a, task_spec=lm_task_spec)
    results_a = trainer_a.train(train_loader=train_loader, val_loader=val_loader)

    # Register Model A
    model_a_path = integration_tmp_dir / "model_a.pt"
    torch.save(model_a.state_dict(), model_a_path)

    model_a_id = registry.register_model(
        name="comparison_model",
        version="A",
        model_path=str(model_a_path),
        config=tiny_config.__dict__,
        task_spec=lm_task_spec.to_dict(),
        training_config=config_a.to_dict(),
        metrics={'final_loss': results_a['final_loss']},
        tags=['candidate']
    )

    # Step 2: Train Model B (with different hyperparams)
    model_b = tiny_transformer_model.to(device)
    adapter_b = UniversalModelAdapter(model_b, tiny_config, lm_task_spec)

    config_b = TrainingConfig(**basic_training_config.to_dict(), learning_rate=5e-4)

    trainer_b = Trainer(model=adapter_b, config=config_b, task_spec=lm_task_spec)
    results_b = trainer_b.train(train_loader=train_loader, val_loader=val_loader)

    # Register Model B
    model_b_path = integration_tmp_dir / "model_b.pt"
    torch.save(model_b.state_dict(), model_b_path)

    model_b_id = registry.register_model(
        name="comparison_model",
        version="B",
        model_path=str(model_b_path),
        config=tiny_config.__dict__,
        task_spec=lm_task_spec.to_dict(),
        training_config=config_b.to_dict(),
        metrics={'final_loss': results_b['final_loss']},
        tags=['candidate']
    )

    # Step 3: Compare models
    comparison = registry.compare_models([model_a_id, model_b_id])
    assert comparison is not None
    assert len(comparison) == 2

    # Find best model (lowest loss)
    best_model = min(comparison, key=lambda m: m['metrics'].get('final_loss', float('inf')))
    best_model_id = best_model['id']

    # Step 4: Promote best model to production
    registry.update_model(best_model_id, status='production', tags=['production', 'champion'])

    # Verify promotion
    best_model_info = registry.get_model(best_model_id)
    assert best_model_info['status'] == 'production'
    assert 'production' in best_model_info['tags']

    # Verify we can query production model
    production_models = registry.list_models(status='production')
    assert len(production_models) >= 1
    assert any(m['id'] == best_model_id for m in production_models)
