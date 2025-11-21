# Architecture: Training Engine v4.0+

**Version:** 4.0+
**Last Updated:** 2025-11-20
**Target Audience:** Contributors, advanced users, architects

---

## Design Philosophy

The v4.0 training engine is built on **composition over inheritance** using proven design patterns:

### Core Principles

1. **Single Responsibility:** Each component has one reason to change
2. **Protocol-Based Design:** No hard dependencies on concrete implementations
3. **Pluggable Components:** Swap implementations without changing core
4. **Type Safety:** Full type hints and Protocol validation
5. **Testability:** Components can be unit tested in isolation
6. **Observability:** Built-in logging, monitoring, and metrics

### Design Patterns Used

| Pattern | Component | Benefit |
|---------|-----------|---------|
| **Strategy** | `LossStrategy` (5 implementations) | Task-specific loss without hardcoding |
| **Registry** | `LossStrategyRegistry`, `CollatorRegistry` | Extensible without modifying core |
| **Builder** | `TrainingConfigBuilder` | Fluent configuration API |
| **Factory** | `DataLoaderFactory` | Consistent DataLoader creation |
| **Facade** | `legacy_api.py` | Backward compatibility wrapper |
| **Protocol** | `TrainingHooks`, `DataModuleProtocol` | Framework-agnostic interfaces |
| **Decorator** | `GradientAccumulator`, `GradientMonitor` | Orthogonal concerns |
| **Observer** | `MetricsEngine` with callbacks | Decoupled metric handling |

---

## Architecture Overview

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Trainer (Orchestrator)                │
│  Delegates to specialized components, manages lifecycle  │
└──────────────┬──────────────────────────────────────────┘
               │
     ┌─────────┼─────────┬──────────┬─────────┬─────────┐
     │         │         │          │         │         │
     ▼         ▼         ▼          ▼         ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────┐ ┌──────┐ ┌──────┐ ┌────────┐
│Training │ │Validation│Loss  │ │Check-│ │Metrics│ │Gradient│
│ Loop    │ │ Loop    │Strat.│ │point │ │Engine │ │Monitor │
└─────────┘ └─────────┘ └─────┘ │Mgr   │ └──────┘ └────────┘
                                 └──────┘
                                    │
                              ┌─────┴──────┐
                              ▼            ▼
                         ┌────────┐  ┌──────────┐
                         │Model   │  │Optimizer │
                         │Registry│  │ Scheduler│
                         └────────┘  └──────────┘

┌──────────────────────────────────────────────────────────┐
│           Data Loading Layer                             │
│  ┌────────────────┐      ┌──────────────────────────────┐│
│  │  DataLoader    │      │  CollatorRegistry             ││
│  │  Factory       │─────▶│  - LanguageModelingCollator  ││
│  └────────────────┘      │  - ClassificationCollator    ││
│                          │  - VisionCollator            ││
│                          └──────────────────────────────┘│
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│           Configuration System                           │
│  ┌───────────────────┐        ┌──────────────────────────┐
│  │TrainingConfig     │        │TrainingConfigBuilder     │
│  │(Dataclass)        │        │(Fluent API)              │
│  ├───────────────────┤        ├──────────────────────────┤
│  │- Hyperparameters  │        │- Presets (5 built-in)    │
│  │- Model arch       │        │- Method chaining         │
│  │- Reproducibility  │        │- Auto-validation         │
│  │- Checkpointing    │        └──────────────────────────┘
│  └───────────────────┘
│
│  ┌───────────────┐     ┌─────────────┐
│  │ TaskSpec      │     │ DataModule  │
│  │ (Task config) │     │ (Data def)  │
│  └───────────────┘     └─────────────┘
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│           Production Features                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Model    │ │ Job      │ │Export    │ │Retraining  │  │
│  │Registry  │ │Queue &   │ │Bundle    │ │Triggers    │  │
│  │          │ │Scheduler │ │          │ │            │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│           Utilities & Integration                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │W&B       │ │Experiment│ │Dashboard │ │Seed        │  │
│  │Integration│ │DB        │ │& Plotting│ │Manager     │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Trainer (Orchestrator)

**Location:** `utils/training/engine/trainer.py`

**Responsibility:** Coordinate components, manage training lifecycle, enforce invariants

**Dependencies:** All Phase 0 components + config + metrics

**Key Design Decision:** Trainer delegates, doesn't implement. Every concern has a specialist.

```python
class Trainer:
    def __init__(self, model, config, training_config, task_spec, hooks=None):
        # Delegate to specialists
        self.checkpoint_manager = CheckpointManager(...)
        self.metrics_engine = MetricsEngine(...)
        self.loss_strategy = get_loss_strategy(task_spec.task_type)
        self.data_factory = DataLoaderFactory(task_spec)
        self.training_loop = TrainingLoop(...)
        self.validation_loop = ValidationLoop(...)

        self.hooks = hooks or DefaultHooks()

    def train(self, train_data, val_data):
        # Orchestrate: call components in right order
        for epoch in range(epochs):
            self.hooks.on_epoch_start(epoch)
            train_result = self.training_loop.run_epoch(...)
            val_result = self.validation_loop.run_epoch(...)
            self.checkpoint_manager.save_checkpoint(...)
            self.metrics_engine.log_epoch(...)
            self.hooks.on_epoch_end(epoch, metrics)
```

**Extension Points:**
- `TrainingHooks`: Custom behavior at lifecycle events
- Task-specific loss via `LossStrategy` registry
- Custom data loading via `DataLoaderFactory`

---

### 2. Loss Strategy Pattern

**Location:** `utils/training/engine/loss.py`

**Responsibility:** Compute task-specific loss with proper input validation

**Why a Pattern?** v3.x hardcoded `F.cross_entropy` for language modeling. This breaks for:
- Classification tasks (different output shape)
- Vision tasks (different input format)
- PEFT fine-tuning (gradient flow considerations)
- Quantized models (numerical sensitivity)

**Strategy Implementations:**

```
LossStrategy (Protocol)
├── LanguageModelingLoss     # Next-token prediction, excludes padding
├── ClassificationLoss       # Single-label classification, weighted
├── PEFTAwareLoss           # LoRA/adapter-aware gradients
├── QuantizationSafeLoss    # Safe for int8/int4 quantization
└── VisionLoss              # Image classification, detection, segmentation
```

**Type Safety:**

```python
# TypedDict for safe, documented inputs
class LossInputs(TypedDict, total=False):
    logits: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pad_token_id: Optional[int]
    pixel_values: Optional[torch.Tensor]
    class_weights: Optional[torch.Tensor]

# Protocol for extensibility
class LossStrategy(Protocol):
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: Optional[LossInputs] = None
    ) -> torch.Tensor:
        """Compute scalar loss."""
        ...

    def validate_inputs(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate shapes and types. Raise ValueError if invalid."""
        ...
```

**Registry Pattern:**

```python
# No modification needed to add custom strategy
registry = LossStrategyRegistry()
registry.register('my_task', CustomLoss())

# Automatically used by Trainer
strategy = registry.get('my_task')
```

---

### 3. Gradient Management (Monitor + Accumulator)

**Location:** `utils/training/engine/gradient_*.py`

**Responsibility:** Health checks and accumulation management (orthogonal concerns)

#### GradientMonitor
- Detects vanishing/exploding gradients
- Tracks norm statistics
- Plugs into training loop for early warning

#### GradientAccumulator
- Determines when to update optimizer
- Applies gradient clipping
- Maps effective steps for W&B logging

**Key Design: Separation of Concerns**

These are orthogonal (can be used independently):
- Monitor without accumulation (single batch per step)
- Accumulate without monitoring (if gradients are healthy)
- Combine both for maximum control

---

### 4. Training and Validation Loops

**Location:** `utils/training/engine/loop.py`

**Responsibility:** Execute one epoch with components plugged in

**Design Pattern: Strategy + Composition**

```python
class TrainingLoop:
    def __init__(
        self,
        gradient_accumulator: GradientAccumulator,
        gradient_monitor: GradientMonitor,
        loss_strategy: LossStrategy,
        gradient_clip_norm: float
    ):
        self.accumulator = gradient_accumulator
        self.monitor = gradient_monitor
        self.loss_strategy = loss_strategy
        self.clip_norm = gradient_clip_norm

    def run_epoch(self, epoch, model, loader, optimizer, scheduler, device, metrics_cb):
        # Pluggable components handle their concerns
        for batch_idx, batch in enumerate(loader):
            # Components compose: loss + accumulation + gradient checking
            logits = model(batch)
            loss = self.loss_strategy.compute_loss(logits, batch['labels'])

            loss.backward()

            if self.accumulator.should_update_optimizer(batch_idx):
                self.monitor.check_step(model)
                self.accumulator.clip_gradients(model)
                optimizer.step()
                optimizer.zero_grad()

            metrics_cb(loss.item(), batch_idx)
```

**Why This Design?**
- Each component testable independently
- Easy to swap implementations
- Clear data flow (loss → backward → check → clip → step)
- No hidden dependencies

---

### 5. MetricsEngine with Drift Detection

**Location:** `utils/training/engine/metrics.py`

**Responsibility:** Log metrics, detect drift, trigger alerts

**Feature: Drift Detection**

Monitors JS divergence between training and validation distributions:
- **Healthy:** JS divergence < 0.1 (normal training)
- **Warning:** 0.1-0.2 (monitor closely, data shift likely)
- **Critical:** > 0.2 (investigate immediately, retrain recommended)

```python
drift_metrics = engine.log_epoch(
    epoch=5,
    train_metrics={'loss': 0.42, 'accuracy': 0.85},
    val_metrics={'loss': 0.38, 'accuracy': 0.87},
    reference_profile=original_dataset_profile,
    current_profile=new_data_profile
)

# Drift metrics indicate data distribution change
if drift_metrics.status == 'critical':
    print(f"⚠️ Retrain recommended: {drift_metrics.affected_features}")
```

**Alert System:**

```python
alert_config = AlertConfig(
    val_loss_spike_threshold=0.2,    # 20% increase = alert
    accuracy_drop_threshold=0.05,     # 5% drop = alert
    gradient_norm_threshold=10.0,
    patience_epochs=2                 # Wait N epochs before escalating
)

if engine.has_alerts():
    for alert in engine.get_alerts():
        logging.warning(f"Alert: {alert['message']}")
```

**Integration with W&B:**

- Gradient accumulation aware (logs every N steps, not every batch)
- Confidence metrics (top-1, top-5, entropy)
- GPU metrics (memory, utilization, temperature)

---

### 6. Checkpoint Manager

**Location:** `utils/training/engine/checkpoint.py`

**Responsibility:** Atomic state persistence with retention policies

**Key Feature: Smart Retention**

```python
manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    keep_best_k=3,      # Keep top 3 by metric
    keep_last_n=5,      # Keep last 5 regardless of metric
    monitor='val_loss',
    mode='min'
)

# Automatic cleanup
for epoch in range(100):
    # ... training ...
    manager.save_checkpoint(epoch, metrics, model, optimizer)
    # Manager: keeps only best 3 + last 5
```

**Retention Invariants:**
- Always have a "best" checkpoint
- Always have recent checkpoints (resume-friendly)
- Bounded disk usage
- Metadata tracking

---

### 7. Data Loading Factory

**Location:** `utils/training/engine/data.py`

**Responsibility:** Create consistent DataLoaders with task-aware collation

**Design: Factory + Registry**

```python
# Factory creates loaders
factory = DataLoaderFactory(task_spec, seed=42)

# Registry holds collators
registry = factory.collator_registry
registry.register('custom_task', custom_collator)

# Create loader with optimal settings for task
loader = factory.create_loader(
    dataset=dataset,
    config=DataLoaderConfig(batch_size=32, shuffle=True)
)
```

**Built-in Collators:**
- `'language_modeling'` - Pads to seq_len, excludes padding from loss
- `'classification'` - Right-pads to max in batch
- `'vision'` - Normalizes images, handles variable sizes
- `'sequence_to_sequence'` - Pads input/output independently

---

## Configuration System

### TrainingConfig (Dataclass)

**Location:** `utils/training/training_config.py`

**Responsibility:** Immutable configuration snapshot for reproducibility

**Design: Dataclass + Validation**

```python
@dataclass
class TrainingConfig:
    # All fields have defaults
    learning_rate: float = 5e-5
    batch_size: int = 4
    epochs: int = 10
    # ... 20+ fields ...

    def validate(self) -> None:
        """Validate constraints."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        # ... more constraints ...

    def to_dict(self) -> Dict:
        """Serialize for JSON/W&B."""
        return asdict(self)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load from JSON."""
        data = json.load(open(path))
        return cls(**data)
```

**Why Dataclass?**
- Automatic `__init__`, `__repr__`, `__eq__`
- Type hints as documentation
- Serialization/deserialization simple
- Validation explicit (call `validate()`)

---

### TrainingConfigBuilder (Fluent API)

**Location:** `utils/training/training_config.py`

**Responsibility:** Convenient config creation with presets

**Design Pattern: Builder with Method Chaining**

```python
# Start with preset
config = (TrainingConfigBuilder.baseline()
    # Customize via method chaining
    .with_training(epochs=30, batch_size=8)
    .with_optimizer(gradient_accumulation_steps=4)
    .with_logging(run_name='exp-1')
    # Build triggers validation
    .build()
)
```

**Presets (5 Built-in):**
1. `quick_prototype()` - 3 epochs, 12M params (debugging)
2. `baseline()` - 10 epochs, 125M params (standard)
3. `production()` - 20 epochs, export enabled
4. `distributed()` - DDP/FSDP, 4 GPUs
5. `low_memory()` - Colab free tier (2 batch, 8x accumulation)

**Why Builder?**
- Prevents "constructor with 30 parameters"
- Chainable for readability
- Validation on build (not on init)
- Presets capture domain knowledge

---

## Production Features

### Model Registry

**Location:** `utils/training/model_registry.py`

**Responsibility:** Version control and metadata for trained models

**Design: SQLite-based, Queryable**

```python
registry = ModelRegistry('models.db')

# Register after training
model_id = registry.register_model(
    name='gpt-finetune',
    version='1.0.0',
    checkpoint_path='checkpoints/epoch_10.pt',
    metrics={'val_loss': 0.38, 'perplexity': 1.46},
    config_hash='abc123...',
    training_run_id=42
)

# Tag for deployment
registry.promote_model(model_id, 'production')

# Load production model
model = registry.get_model(tag='production')

# Compare models
df = registry.compare_models([1, 2, 3])
# SQL query: SELECT version, val_loss, created_at FROM models WHERE id IN (1,2,3)
```

**Why SQL?**
- Queryable (easy to find best model)
- Scalable (millions of versions)
- Transactional (atomic updates)
- Linkable (connect to ExperimentDB runs)

---

### Job Queue & Scheduler

**Location:** `utils/training/job_queue.py`

**Responsibility:** Orchestrate automated training workflows

**Design: SQLite Queue with Atomic Claiming**

```python
manager = JobManager('jobs.db')

# Submit training job
job_id = manager.submit_job(
    job_type='training',
    config={'training_config': config.to_dict()},
    priority=5,
    max_retries=3
)

# Scheduler: recurring jobs
scheduler = TrainingScheduler('jobs.db')
scheduler.create_schedule(
    name='daily-retrain',
    job_type='retraining',
    schedule_expr='0 2 * * *',  # Cron: daily at 2am
    priority=3
)

# Worker: execute jobs
executor = JobExecutor(manager, worker_id='worker-1')
executor.run_worker(max_jobs=10)

# Worker:
# 1. Poll for pending jobs (ORDER BY priority DESC)
# 2. Atomically claim job (CAS: compare-and-swap on worker_id)
# 3. Execute job
# 4. Update status (completed/failed)
# 5. Repeat
```

**Why SQLite Queue?**
- Single-node (Colab, local dev)
- Atomic claiming (no race conditions)
- Persistent (survive restarts)
- Simple (no message broker)

---

### Export Bundle

**Location:** `utils/training/export_utilities.py`

**Responsibility:** Generate production-ready deployment artifacts

**Generated Structure:**
```
exports/model_<timestamp>/
├── artifacts/
│   ├── model.onnx              # Cross-platform inference
│   ├── model.torchscript.pt    # TorchScript JIT
│   └── model.pytorch.pt        # PyTorch state dict
├── configs/
│   ├── task_spec.json
│   ├── training_config.json
│   └── torchserve_config.json  # TorchServe deployment
├── inference.py                # Standalone script
├── README.md                   # Quick-start
├── Dockerfile                  # Container
└── requirements.txt            # Dependencies
```

**Why Bundle Everything?**
- One export = complete deployment
- Configs linked (reproducibility)
- Multiple formats (flexibility)
- Scripts tested (no surprises)

---

## Phase Timeline

### Phase 0: Engine Architecture (v4.0.0)
- CheckpointManager
- LossStrategy + 5 implementations
- GradientMonitor / GradientAccumulator
- DataLoaderFactory + CollatorRegistry
- TrainingLoop / ValidationLoop
- MetricsEngine (basic)
- Trainer orchestrator

### Phase 1: Advanced Features (v4.1.0)
- MetricsEngine enhancements (drift, alerts, confidence)
- Dashboard & visualization
- Flash Attention support
- Distributed training guardrails

### Phase 2: Production Hardening (v4.2.0)
- ModelRegistry (versioning, tagging)
- JobQueue & Scheduler (automation)
- ExportBundle (deployment artifacts)
- RetrainingTriggers (automated workflows)

### Phase 3: Testing & Migration (v4.3.0)
- Complete API documentation
- Migration guides
- Legacy API wrappers
- Example notebooks

---

## Extension Guide

### Adding a Custom Loss Strategy

```python
# 1. Define strategy
class CustomLoss:
    def compute_loss(self, logits, labels, inputs=None):
        # Custom logic
        return F.cross_entropy(logits.view(-1, logits.size(-1)),
                              labels.view(-1),
                              ignore_index=inputs.get('pad_token_id', 0))

    def validate_inputs(self, logits, labels):
        assert logits.ndim == 3
        assert labels.ndim == 2

# 2. Register
from utils.training.engine import LossStrategyRegistry
registry = LossStrategyRegistry()
registry.register('my_loss', CustomLoss())

# 3. Use
strategy = registry.get('my_loss')
loss = strategy.compute_loss(logits, labels)
```

### Adding Custom Hooks

```python
# 1. Implement protocol
class CustomHooks:
    def on_epoch_start(self, epoch: int) -> None:
        print(f"Epoch {epoch} starting")

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, loss={loss:.4f}")

    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        print(f"Epoch {epoch}: {metrics}")

# 2. Pass to trainer
trainer = Trainer(..., hooks=CustomHooks())

# 3. Trainer calls hooks at right times
```

### Adding Custom Data Collator

```python
# 1. Define collator function
def my_collator(batch):
    # Custom batching logic
    return {
        'input_ids': torch.tensor([...]),
        'labels': torch.tensor([...])
    }

# 2. Register
factory = DataLoaderFactory(task_spec)
factory.collator_registry.register('my_task', my_collator)

# 3. Use
loader = factory.create_loader(dataset, is_eval=False)
```

---

## Performance Characteristics

### Overhead Analysis

| Component | Overhead per Step | Notes |
|-----------|-------------------|-------|
| Trainer orchestration | <1ms | Delegating only |
| LossStrategy dispatch | <1ms | Protocol call |
| GradientMonitor | <1ms | Only when check_interval hits |
| GradientAccumulator | <1ms | Step tracking only |
| MetricsEngine (no W&B) | <1ms | In-memory logging |
| MetricsEngine (W&B) | ~10ms | Network I/O (batch) |
| **Total** | **<5ms** | Negligible vs training time |

### Comparison: v3.x → v4.0

```
v3.x (monolithic):
- test_fine_tuning(): 100% of time in training
- Overhead: ~2% (minimal, tight loop)

v4.0 (modular):
- Trainer delegates to components
- Overhead: ~5% (slightly higher due to protocols, still <1ms/step)
- Benefit: Composability, extensibility, testing
```

### Memory Overhead

| Component | Memory |
|-----------|--------|
| CheckpointManager metadata | ~10MB |
| MetricsEngine tracking | ~5MB |
| GradientMonitor statistics | <1MB |
| ExperimentDB SQLite | ~2MB per 1K runs |
| **Total** | ~50MB |

---

## Testing Strategy

### Unit Tests (Component Isolation)

```
test_checkpoint_manager.py
  - Test save/load state dict
  - Test retention policies (keep_best_k, keep_last_n)
  - Test metadata tracking

test_loss_strategy.py
  - Test each strategy implementation
  - Test input validation
  - Test numerical stability

test_trainer.py
  - Test orchestration (calls components in order)
  - Test hook invocation
  - Test early stopping
```

### Integration Tests (Component Interaction)

```
test_training_loop.py
  - Train for 1 epoch
  - Verify checkpoint saved
  - Verify metrics logged
  - Verify model weights changed

test_full_pipeline.py
  - Train model end-to-end
  - Export to bundle
  - Register in model registry
  - Load and verify
```

### E2E Tests (Notebook Workflows)

```
test_colab_notebook.py
  - Load model from Gist
  - Train in Colab runtime
  - Check W&B logging
  - Verify export artifacts
```

---

## Backward Compatibility

### Compatibility Guarantee: v3.x Code Works

```python
# v3.x code continues to work via legacy_api module
from utils.training.legacy_api import test_fine_tuning

results = test_fine_tuning(
    model=model,
    config=config,
    n_epochs=10,
    use_wandb=True
)
```

### How It Works

```python
# legacy_api.py: Thin wrapper around new engine
def test_fine_tuning(model, config, n_epochs, **kwargs):
    # Translate old API to new components
    task_spec = infer_task_spec(config)
    trainer = Trainer(
        model=model,
        config=config,
        training_config=config,
        task_spec=task_spec
    )

    # Call new engine
    results = trainer.train(train_data, val_data)

    # Translate output back to old format
    return translate_results(results)
```

### Deprecation Timeline

| Version | Status |
|---------|--------|
| v4.0 | Legacy API works, no warnings |
| v4.2 | Legacy API works, deprecation warnings |
| v5.0 | Legacy API removed |

---

## Index of Components

### Engine Components (Phase 0)
- **CheckpointManager** (`utils/training/engine/checkpoint.py`) - State persistence
- **LossStrategy** (`utils/training/engine/loss.py`) - Task-specific loss (5 implementations)
- **GradientMonitor** (`utils/training/engine/gradient_monitor.py`) - Gradient health
- **GradientAccumulator** (`utils/training/engine/gradient_accumulator.py`) - Gradient accumulation
- **DataLoaderFactory** (`utils/training/engine/data.py`) - DataLoader creation
- **CollatorRegistry** (`utils/training/engine/data.py`) - Task-aware collation
- **TrainingLoop** (`utils/training/engine/loop.py`) - Epoch execution
- **ValidationLoop** (`utils/training/engine/loop.py`) - Validation epoch
- **MetricsEngine** (`utils/training/engine/metrics.py`) - Metrics + drift
- **Trainer** (`utils/training/engine/trainer.py`) - Orchestrator

### Configuration (Phase 0)
- **TrainingConfig** (`utils/training/training_config.py`) - Config dataclass
- **TrainingConfigBuilder** (`utils/training/training_config.py`) - Fluent API
- **TaskSpec** (`utils/training/task_spec.py`) - Task specification

### Production (Phase 2)
- **ModelRegistry** (`utils/training/model_registry.py`) - Model versioning
- **JobQueue** (`utils/training/job_queue.py`) - Job management
- **ExportBundle** (`utils/training/export_utilities.py`) - Deployment artifacts
- **RetrainingTriggers** (`utils/training/retraining_triggers.py`) - Automated workflows

### Utilities
- **MetricsTracker** (`utils/training/metrics_tracker.py`) - Legacy metrics (v3.x compatible)
- **ExperimentDB** (`utils/training/experiment_db.py`) - Local experiment tracking
- **SeedManager** (`utils/training/seed_manager.py`) - Reproducibility

### Backward Compatibility
- **legacy_api** (`utils/training/legacy_api.py`) - v3.x wrapper

---

**Last Updated:** 2025-11-20
**Maintainer:** Architecture & MLOps Teams
**License:** Apache 2.0
