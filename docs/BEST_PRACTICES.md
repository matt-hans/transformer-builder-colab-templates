# Best Practices: Training Engine v4.0+

**Version:** 4.0+
**Last Updated:** 2025-11-20
**Target Audience:** Practitioners, teams building production systems

---

## Configuration Best Practices

### 1. Use Presets, Not Raw Config

**❌ Anti-Pattern:**
```python
config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    # Forgetting crucial settings
)
```

**✅ Best Practice:**
```python
config = TrainingConfigBuilder.baseline().build()
# Pre-tuned settings based on best practices
# Includes: LR schedule, warmup, weight decay, grad clipping, etc.
```

### 2. Customize via Builder, Not Direct Instantiation

**❌ Anti-Pattern:**
```python
# Create new config from scratch
config = TrainingConfig(learning_rate=1e-4, batch_size=8, epochs=20)
# Easy to miss important settings
```

**✅ Best Practice:**
```python
config = (TrainingConfigBuilder.baseline()
    .with_training(learning_rate=1e-4, batch_size=8, epochs=20)
    .build()  # Validates automatically
)
```

### 3. Match Config to Task Complexity

**For Quick Prototyping (< 1 hour):**
```python
config = TrainingConfigBuilder.quick_prototype().build()
# 3 epochs, small batch, disables expensive features
```

**For Standard Experiments (1-24 hours):**
```python
config = TrainingConfigBuilder.baseline().build()
# 10 epochs, balanced settings, checkpointing enabled
```

**For Production Deployment (reproducibility critical):**
```python
config = TrainingConfigBuilder.production()
    .with_training(deterministic=True)
    .build()
# 20 epochs, full validation, export enabled, bit-exact reproducibility
```

**For Resource-Constrained (Colab free tier):**
```python
config = TrainingConfigBuilder.low_memory().build()
# 2 batch, 8x gradient accumulation, smaller sequences
```

### 4. Validate Configuration Early

**❌ Anti-Pattern:**
```python
config = TrainingConfig(learning_rate=-1, batch_size=0, epochs=0)
# Errors discovered hours into training
results = trainer.train(...)  # Fails at runtime
```

**✅ Best Practice:**
```python
config = TrainingConfigBuilder.baseline().build()
config.validate()  # Catches errors immediately

# If manually creating:
try:
    config = TrainingConfig(...)
    config.validate()
except ValueError as e:
    print(f"Config error: {e}")
    # Fix and retry
```

### 5. Save Configuration for Reproducibility

**❌ Anti-Pattern:**
```python
config = TrainingConfigBuilder.baseline().build()
trainer.train(...)  # Lost config!
# Cannot reproduce 6 months later
```

**✅ Best Practice:**
```python
config = TrainingConfigBuilder.baseline().build()
config_path = config.save()  # Auto-generated timestamped filename
print(f"Config saved: {config_path}")

trainer = Trainer(..., training_config=config)
results = trainer.train(...)
```

---

## Training Best Practices

### 1. Use Trainer Class for Reproducibility

**❌ Anti-Pattern:**
```python
# Manual loop: easy to miss crucial steps
for epoch in range(10):
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Forgot: validation, checkpointing, metrics logging, early stopping
```

**✅ Best Practice:**
```python
trainer = Trainer(
    model=model,
    config=model_config,
    training_config=config,
    task_spec=task_spec
)

results = trainer.train(train_data, val_data)
# Handles: validation, checkpointing, metrics, early stopping, LR scheduling
```

### 2. Specify Task Early

**❌ Anti-Pattern:**
```python
trainer = Trainer(model, config, training_config, None)  # task_spec=None
# Trainer infers wrong loss strategy
# Training converges to wrong objective
```

**✅ Best Practice:**
```python
task_spec = TaskSpec.language_modeling(name='wikitext')
# or
task_spec = TaskSpec.classification(name='imdb', num_classes=2)
# or
task_spec = TaskSpec.vision_tiny()

trainer = Trainer(..., task_spec=task_spec)
# Loss strategy auto-selected, verified correct
```

### 3. Use Deterministic Mode for Reproducible Experiments

**❌ Anti-Pattern:**
```python
# Non-deterministic GPU operations
config = TrainingConfigBuilder.baseline().build()

# Run 1: loss trajectory = [0.5, 0.3, 0.2, 0.15]
results1 = trainer.train(...)

# Run 2: loss trajectory = [0.49, 0.31, 0.19, 0.14]  # Different!
results2 = trainer.train(...)
# Cannot debug which change caused difference
```

**✅ Best Practice (for final experiments):**
```python
config = TrainingConfigBuilder.production()
    .with_training(deterministic=True)
    .build()

# Run 1: loss = [0.5, 0.3, 0.2, 0.15]
results1 = trainer.train(...)

# Run 2: loss = [0.5, 0.3, 0.2, 0.15]  # Identical!
results2 = trainer.train(...)
# Confident that changes to code caused difference, not randomness
```

**Note:** Deterministic mode is ~5-10% slower. Use for publication/A/B tests only.

### 4. Monitor Gradient Health

**❌ Anti-Pattern:**
```python
# No gradient checking
trainer = Trainer(...)
results = trainer.train(...)
# Exploding gradients silently corrupted 5 epochs of training
```

**✅ Best Practice:**
```python
monitor = GradientMonitor(
    check_interval=10,
    norm_threshold_warning=1.0,
    norm_threshold_critical=10.0
)

trainer = Trainer(
    model=model,
    config=model_config,
    training_config=config,
    task_spec=task_spec,
    gradient_monitor=monitor
)

results = trainer.train(train_data, val_data)

# Monitor detects gradient explosion, alerts via logging
```

### 5. Use Gradient Accumulation for Effective Batch Sizes

**❌ Anti-Pattern:**
```python
config = TrainingConfigBuilder.baseline()
    .with_optimizer(gradient_accumulation_steps=1)  # Effective batch = 4
    .build()

# Want effective batch = 32 but GPU memory only allows 4
# Solution: larger LR, longer warmup
```

**✅ Best Practice:**
```python
config = TrainingConfigBuilder.baseline()
    .with_optimizer(
        gradient_accumulation_steps=8  # Effective batch = 32
    )
    .build()

# Trainer automatically handles:
# - Accumulating gradients for 8 steps
# - Clipping accumulated gradients
# - Updating optimizer every 8 steps
# - W&B logging every accumulation cycle (75% reduction)
```

---

## Checkpoint & Recovery Best Practices

### 1. Enable Checkpoint Management

**❌ Anti-Pattern:**
```python
# Manual checkpointing
torch.save(model.state_dict(), 'model_latest.pt')
# Only latest saved, cannot resume, no metadata

# Training crashes at epoch 50
# Restart training from epoch 1 (wasted compute)
```

**✅ Best Practice:**
```python
config = TrainingConfigBuilder.baseline()
    .with_checkpointing(
        checkpoint_dir='./checkpoints',
        save_every_n_epochs=1
    )
    .build()

trainer = Trainer(..., training_config=config)
results = trainer.train(...)

# CheckpointManager:
# - Saves best checkpoint (val_loss metric)
# - Keeps last 5 checkpoints
# - Stores metadata (epoch, metrics, timestamp)
```

### 2. Resume from Checkpoint

**❌ Anti-Pattern:**
```python
# Manual resume
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# Lost optimizer state, epoch number, metric tracking
# Resume training with different optimization state
```

**✅ Best Practice:**
```python
checkpoint_manager = CheckpointManager('checkpoints')

# Load best checkpoint
best_path = checkpoint_manager.get_best_checkpoint()
metadata = checkpoint_manager.load_checkpoint(
    path=best_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

# Resume at epoch after best
start_epoch = metadata.epoch + 1

trainer = Trainer(...)
results = trainer.train(...)
```

### 3. Monitor Checkpoint Disk Usage

**❌ Anti-Pattern:**
```python
checkpoint_manager = CheckpointManager(
    'checkpoints',
    keep_best_k=1000,  # Unconstrained
    keep_last_n=1000
)
# After 100 epochs: 100GB disk used!
```

**✅ Best Practice:**
```python
checkpoint_manager = CheckpointManager(
    'checkpoints',
    keep_best_k=3,      # Keep top 3 by metric
    keep_last_n=5,      # Keep last 5 regardless
    save_interval_epochs=5  # Don't save every epoch
)
# After 100 epochs: ~500MB disk used (estimated)
```

---

## Metrics & Monitoring Best Practices

### 1. Log All Relevant Metrics

**❌ Anti-Pattern:**
```python
tracker = MetricsTracker(use_wandb=True)
tracker.log_epoch(epoch=5, train_metrics={'loss': 0.42})
# No validation metrics, learning rate, gradient info
# Cannot diagnose why validation diverged
```

**✅ Best Practice:**
```python
engine = MetricsEngine(use_wandb=True)

engine.log_epoch(
    epoch=5,
    train_metrics={
        'loss': 0.42,
        'accuracy': 0.85,
        'perplexity': 1.53
    },
    val_metrics={
        'loss': 0.38,
        'accuracy': 0.87,
        'perplexity': 1.46
    },
    learning_rate=1e-4,
    gradient_norm=0.5,
    epoch_duration=120.5
)
# W&B logs: loss, accuracy, perplexity, LR, gradients, timing
```

### 2. Use Drift Detection for Data Quality

**❌ Anti-Pattern:**
```python
# No drift monitoring
trainer.train(train_data, val_data)
# Model accuracy drops 20% after deployment
# Root cause: training data changed, model not retrained
```

**✅ Best Practice:**
```python
from utils.training.drift_metrics import profile_dataset

# Profile training data
ref_profile = profile_dataset(train_dataset, task_spec)

# During validation
current_profile = profile_dataset(val_dataset, task_spec)

engine.log_epoch(
    ...,
    reference_profile=ref_profile,
    current_profile=current_profile
)

# Engine detects drift, alerts if JS divergence > threshold
# Trigger retraining if drift critical
```

### 3. Set Up Alerts for Anomalies

**❌ Anti-Pattern:**
```python
# No alerting
tracker.log_epoch(epoch=5, train_metrics={'loss': 0.42})
# Loss spikes to 10.0 at epoch 50 (gradient explosion)
# Discovered after 50 epochs wasted
```

**✅ Best Practice:**
```python
engine = MetricsEngine(
    use_wandb=True,
    alert_config=AlertConfig(
        val_loss_spike_threshold=0.2,      # 20% increase = alert
        accuracy_drop_threshold=0.05,       # 5% drop = alert
        gradient_norm_threshold=10.0,
        patience_epochs=2
    )
)

engine.log_epoch(...)

if engine.has_alerts():
    for alert in engine.get_alerts():
        logging.error(f"Alert: {alert['message']}")
        # Email, Slack, or auto-stop training
```

---

## Export & Deployment Best Practices

### 1. Export Complete Bundle

**❌ Anti-Pattern:**
```python
# Export only model
torch.save(model.state_dict(), 'model.pt')

# Deployment:
# - What preprocessing was used?
# - What tokenizer?
# - What task specification?
# - Guess and deploy wrong model!
```

**✅ Best Practice:**
```python
export_dir = create_export_bundle(
    model=model,
    config=model_config,
    task_spec=task_spec,
    training_config=config,
    formats=['onnx', 'torchscript', 'pytorch'],
    output_dir='./exports'
)

# Complete bundle:
# - Model in 3 formats (ONNX, TorchScript, PyTorch)
# - Task spec (preprocessing, tokenizer)
# - Training config (hyperparameters)
# - Inference script (standalone)
# - Dockerfile (reproducible deployment)
# - README (usage documentation)
```

### 2. Use Model Registry for Versioning

**❌ Anti-Pattern:**
```python
# Manual versioning
torch.save(model.state_dict(), 'model_v1.pt')
torch.save(model.state_dict(), 'model_v1_fixed.pt')
torch.save(model.state_dict(), 'model_v1_prod.pt')
# Unclear which is production, which is best
```

**✅ Best Practice:**
```python
registry = ModelRegistry('models.db')

model_id = registry.register_model(
    name='transformer-gpt',
    version='1.0.0',
    checkpoint_path='checkpoints/epoch_10.pt',
    task_type='language_modeling',
    metrics={'val_loss': 0.38, 'perplexity': 1.46}
)

# Tag for deployment
registry.promote_model(model_id, 'production')
registry.promote_model(model_id, 'staging')

# Load production model
prod_model = registry.get_model(tag='production')

# Rollback if needed
old_model = registry.get_model(version='0.9.5')
```

### 3. Document Deployment Process

**❌ Anti-Pattern:**
```bash
# Vague deployment notes
# "Run model_v1 in Docker"
# - What's the exact command?
# - What resources needed?
# - How to monitor?
```

**✅ Best Practice:**
```markdown
# Deployment Guide

## Quick Start
```bash
docker build -t model:v1 exports/model_20251120_143022/
docker run -p 8080:8080 model:v1
```

## System Requirements
- GPU: NVIDIA T4 or better
- Memory: 8GB RAM, 4GB VRAM
- Throughput: 100 requests/second

## Monitoring
- Latency: exported as prometheus_latency_seconds
- Errors: check logs at /var/log/model.log
- Health check: GET /health

## Rollback
```bash
# If model v1 has issues, rollback to v0.9.5
docker pull model:v0.9.5
docker run -p 8080:8080 model:v0.9.5
```
```

---

## Hyperparameter Tuning Best Practices

### 1. Use Structured Sweep, Not Manual Trial-and-Error

**❌ Anti-Pattern:**
```python
# Manual tuning
config1 = TrainingConfig(learning_rate=1e-4, batch_size=4)
config2 = TrainingConfig(learning_rate=5e-5, batch_size=8)
config3 = TrainingConfig(learning_rate=1e-5, batch_size=16)
# Hard to track what worked, cannot parallelize
```

**✅ Best Practice:**
```python
import optuna

def objective(trial):
    config = TrainingConfigBuilder.baseline()
        .with_training(
            learning_rate=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_int('batch', 4, 32, step=4),
            epochs=10
        )
        .build()

    trainer = Trainer(model, model_config, config, task_spec)
    results = trainer.train(train_data, val_data)

    return results['best_val_loss']

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print(f"Best: {study.best_value:.4f}")
print(f"Params: {study.best_params}")
```

### 2. Log Hyperparameter Sweeps to ExperimentDB

**❌ Anti-Pattern:**
```python
# Run 20 trials, forget parameters
results_1 = trainer.train(...)
results_2 = trainer.train(...)
# Which config worked best? Unclear
```

**✅ Best Practice:**
```python
db = ExperimentDB('experiments.db')

for trial in range(20):
    config = ...  # Create config for this trial
    run_id = db.log_run(
        run_name=f'hparam-search-trial-{trial}',
        config=config.to_dict(),
        notes=f'LR={config.learning_rate}, BS={config.batch_size}'
    )

    trainer = Trainer(..., training_config=config)
    results = trainer.train(...)

    db.log_metric(run_id, 'val_loss', results['best_val_loss'])
    db.update_run_status(run_id, 'completed')

# Analysis
best = db.get_best_run('val_loss', mode='min')
print(f"Best: {best['run_name']}, loss={best['best_value']:.4f}")
print(f"Config: {best['config']}")
```

---

## Team Collaboration Best Practices

### 1. Use Model Registry for Shared Models

**❌ Anti-Pattern:**
```python
# Scattered model files
# "Use the model in /Users/alice/models/best.pt"
# - Where is it? What version? Does alice still have it?
```

**✅ Best Practice:**
```python
# Centralized registry accessible to team
registry = ModelRegistry('/shared/models.db')

# Alice: Register her best model
registry.register_model(
    name='transformer-gpt',
    version='1.0.0',
    checkpoint_path='./checkpoints/epoch_10.pt',
    metrics={'val_loss': 0.38}
)

# Bob: Load Alice's model
bob_registry = ModelRegistry('/shared/models.db')
model = bob_registry.get_model(name='transformer-gpt', version='1.0.0')
```

### 2. Share Configurations via Git

**❌ Anti-Pattern:**
```python
# "Use learning rate 5e-5"
# - Hard to track, easy to forget
```

**✅ Best Practice:**
```python
# Commit configs to git
config = TrainingConfigBuilder.baseline().build()
config_path = config.save('configs/baseline_v1.json')
# Commit configs/baseline_v1.json to git

# Team can reproduce exactly
config = TrainingConfig.load('configs/baseline_v1.json')
```

### 3. Document Experiment Results

**❌ Anti-Pattern:**
```
experiment_results.txt
"V1: loss 0.38, accuracy 0.87
V2: loss 0.35 (better!)
V3: tried new data, loss 0.4 (worse)"
# No metadata, reproduction info, or decision rationale
```

**✅ Best Practice:**
```python
db = ExperimentDB('experiments.db')

run_id = db.log_run(
    run_name='baseline-v2-larger-model',
    config=config.to_dict(),
    notes="""
    Testing 24-layer model (vs 12-layer baseline).
    Hypothesis: larger model should improve val accuracy.
    Previous best: val_loss=0.38 (12-layer)
    Expected: val_loss=0.35 (24-layer)
    """
)

# Log results
for epoch, loss in enumerate(losses):
    db.log_metric(run_id, 'val_loss', loss, epoch=epoch)

# Document decision
db.update_run_status(run_id, 'completed')
# Later: retrieve run and see full context
```

---

## Production Operations Best Practices

### 1. Automate Retraining via Job Queue

**❌ Anti-Pattern:**
```python
# Manual retraining
# "Run training script every Monday morning"
# - Easy to forget
# - Manual intervention required
# - Single point of failure
```

**✅ Best Practice:**
```python
scheduler = TrainingScheduler('jobs.db')

# Schedule daily retraining at 2am
scheduler.create_schedule(
    name='daily-retrain',
    job_type='retraining',
    config={'training_config': config.to_dict()},
    schedule_expr='0 2 * * *',  # Cron: daily at 2am UTC
    priority=3
)

# Worker process handles execution
executor = JobExecutor(manager, worker_id='retraining-worker')
executor.run_worker(max_jobs=10)  # Process up to 10 jobs
```

### 2. Set Up Retraining Triggers

**❌ Anti-Pattern:**
```python
# No automated retraining
# Model accuracy degrades over time
# Manual monitoring catches it 2 weeks later
```

**✅ Best Practice:**
```python
trigger = RetrainingTrigger(
    monitor_metric='val_loss',
    degradation_threshold=0.1,  # 10% worse = retrain
    min_days_between_retrains=7,
    drift_threshold=0.2
)

# After each validation
current_metrics = validate()
drift_score = compute_drift()

if trigger.should_retrain(
    current_metrics=current_metrics,
    reference_metrics=best_metrics,
    drift_score=drift_score
):
    # Auto-submit retraining job
    manager.submit_job(
        job_type='retraining',
        config={'training_config': config.to_dict()},
        priority=8  # Higher priority than scheduled jobs
    )
```

### 3. Monitor Model Performance in Production

**❌ Anti-Pattern:**
```python
# Deploy model, forget about it
# Latency increases 10x due to model complexity
# Error rate spikes due to distribution shift
# No one notices for weeks
```

**✅ Best Practice:**
```python
# Export includes monitoring config
export_dir = create_export_bundle(
    model=model,
    ...,
    monitoring_config={
        'latency_threshold_ms': 100,
        'error_rate_threshold': 0.01,
        'drift_threshold': 0.2,
        'alert_email': 'team@example.com'
    }
)

# Deployment collects metrics
# - Latency (p50, p95, p99)
# - Error rate
# - Model output distribution (drift)
# - User feedback scores

# Alerting
if latency_p95 > 100ms:
    alert("Model latency degradation detected")
if error_rate > 1%:
    alert("Spike in prediction errors")
if drift_score > 0.2:
    alert("Production data distribution changed - retrain recommended")
```

---

## Troubleshooting Best Practices

### Issue: Training Loss Not Decreasing

**❌ Debugging Blind:**
```python
# Run training, see loss flat
# Hypothesis: learning rate too low?
# Change learning rate, try again (wastes 1 hour)
```

**✅ Systematic Debugging:**
```python
# 1. Check gradient health
monitor = GradientMonitor()
health = monitor.check_step(model)
if health.has_vanishing_gradients:
    print(f"Gradients too small: {health.min_norm:.6f}")
    # Fix: increase learning rate
if health.has_exploding_gradients:
    print(f"Gradients too large: {health.max_norm:.2f}")
    # Fix: decrease learning rate or increase gradient clipping

# 2. Check loss strategy
strategy = get_loss_strategy(task_spec.task_type)
strategy.validate_inputs(logits, labels)

# 3. Check data
sample_batch = next(iter(train_loader))
print(f"Input shape: {sample_batch['input_ids'].shape}")
print(f"Label shape: {sample_batch['labels'].shape}")
# Verify shapes match model expectations

# 4. Quick diagnostic training
config = TrainingConfigBuilder.quick_prototype().build()
results = trainer.train(train_data[:100], val_data[:100])  # 100 samples only
# Should show loss decreasing or clear error message
```

### Issue: Model Not Fitting Training Data

**❌ Debugging Blind:**
```python
# Model validation accuracy same as random
# Hypothesis: model too small?
# Increase model size, retrain (wastes compute)
```

**✅ Systematic Debugging:**
```python
# 1. Check model capacity
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable_params}")

# 2. Verify model is actually training
before_params = [p.clone() for p in model.parameters()]
train_epoch(model, train_loader)
after_params = model.parameters()
param_changed = any((before - after).abs().sum() > 1e-6
                    for before, after in zip(before_params, after_params))
if not param_changed:
    print("ERROR: Model parameters not changing!")

# 3. Check for data leakage
# Verify train and val come from different splits
assert len(set(train_indices) & set(val_indices)) == 0

# 4. Train on tiny dataset
results = trainer.train(
    train_data=train_data[:10],  # Only 10 samples
    val_data=val_data[:10]
)
# Model should overfit and get ~100% accuracy
# If not: check loss function, model architecture, or data format
```

---

## Checklist: Before Production Deployment

- [ ] Configuration validated and saved to git
- [ ] All metrics logged (loss, accuracy, precision, recall)
- [ ] Gradient health checked (no vanishing/exploding)
- [ ] Model registered in model registry with metadata
- [ ] Best checkpoint identified and tested
- [ ] Export bundle created with all artifacts
- [ ] Inference script tested end-to-end
- [ ] Dockerfile builds and runs successfully
- [ ] Documentation complete (README, usage guide)
- [ ] Performance benchmarked (latency, throughput)
- [ ] Monitoring configured (alerts, metrics collection)
- [ ] Rollback plan documented
- [ ] Team reviewed and approved
- [ ] Retrain triggers configured
- [ ] Post-deployment monitoring active

---

**Last Updated:** 2025-11-20
**Maintainer:** MLOps & Best Practices Team
**License:** Apache 2.0
