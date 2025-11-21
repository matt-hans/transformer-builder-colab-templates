# Documentation Index - Training Engine v4.0+

**Version:** 4.0+
**Last Updated:** 2025-11-20
**Status:** Complete

---

## Quick Navigation

### Getting Started (New Users)
1. **[README.md](../README.md)** - High-level overview, quick start, repository structure
2. **[USAGE_GUIDE_COLAB_AND_CLI.md](./USAGE_GUIDE_COLAB_AND_CLI.md)** - Usage in Colab notebooks and CLI
3. **[BEST_PRACTICES.md](./BEST_PRACTICES.md)** - Practical patterns and anti-patterns

### API Reference (Developers)
1. **[API_REFERENCE.md](./API_REFERENCE.md)** - Complete API with examples (2000+ lines)
   - CheckpointManager
   - LossStrategy (5 implementations)
   - GradientMonitor & GradientAccumulator
   - DataLoaderFactory & CollatorRegistry
   - TrainingLoop & ValidationLoop
   - MetricsEngine with drift detection
   - Trainer orchestrator
   - Configuration system (TrainingConfig, TrainingConfigBuilder, TaskSpec)
   - Production features (ModelRegistry, JobQueue, ExportBundle)

### Architecture & Design (Contributors)
1. **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Design patterns, component interactions, extension guide
2. **[TYPE_SYSTEM.md](./TYPE_SYSTEM.md)** - Type hints, Protocol-based design

### Migration & Upgrades
1. **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** - Step-by-step upgrade from v3.x to v4.0+
   - Quick start (5 minutes)
   - Detailed scenarios
   - Breaking changes
   - Common issues & solutions
   - Backward compatibility guarantees

### Specialized Topics
1. **[DATA_LOADING_GUIDE.md](./DATA_LOADING_GUIDE.md)** - Data loading, collators, TaskSpec
2. **[METRICS_ENGINE_MIGRATION.md](./METRICS_ENGINE_MIGRATION.md)** - Metrics tracking, drift detection
3. **[JOB_QUEUE_GUIDE.md](./JOB_QUEUE_GUIDE.md)** - Job scheduling, automated workflows
4. **[MODEL_REGISTRY.md](./MODEL_REGISTRY.md)** - Model versioning, tagging, deployment
5. **[FLASH_ATTENTION_VALIDATION.md](./FLASH_ATTENTION_VALIDATION.md)** - Performance optimization
6. **[COLAB_TROUBLESHOOTING.md](./COLAB_TROUBLESHOOTING.md)** - Common issues in Colab
7. **[COLAB_VALIDATION_INSTRUCTIONS.md](./COLAB_VALIDATION_INSTRUCTIONS.md)** - Validation in Colab

---

## By User Role

### Data Scientists & ML Engineers

**For quick training:**
1. Read: [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Configuration patterns
2. Use: [API_REFERENCE.md](./API_REFERENCE.md) - Trainer API
3. Reference: [DATA_LOADING_GUIDE.md](./DATA_LOADING_GUIDE.md) - Data setup

**For hyperparameter tuning:**
1. Read: [USAGE_GUIDE_COLAB_AND_CLI.md](./USAGE_GUIDE_COLAB_AND_CLI.md) - Sweep setup
2. Reference: [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Tuning strategies
3. Look up: [API_REFERENCE.md](./API_REFERENCE.md#common-workflows) - Code examples

**For production deployment:**
1. Read: [MODEL_REGISTRY.md](./MODEL_REGISTRY.md) - Versioning
2. Read: [API_REFERENCE.md](./API_REFERENCE.md#exportbundle) - Export bundle
3. Reference: [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Production checklist

### ML Infrastructure / MLOps

**For job scheduling & automation:**
1. Read: [JOB_QUEUE_GUIDE.md](./JOB_QUEUE_GUIDE.md) - Full guide
2. Reference: [API_REFERENCE.md](./API_REFERENCE.md#jobqueue) - API
3. Read: [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Operations patterns

**For monitoring & alerts:**
1. Read: [METRICS_ENGINE_MIGRATION.md](./METRICS_ENGINE_MIGRATION.md) - Drift detection
2. Reference: [API_REFERENCE.md](./API_REFERENCE.md#metricsengine) - MetricsEngine API
3. Look up: [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Monitoring setup

### Software Engineers & Contributors

**For understanding architecture:**
1. Read: [ARCHITECTURE.md](./ARCHITECTURE.md) - Design patterns
2. Read: [TYPE_SYSTEM.md](./TYPE_SYSTEM.md) - Type safety
3. Reference: [API_REFERENCE.md](./API_REFERENCE.md) - Component APIs

**For extending the engine:**
1. Read: [ARCHITECTURE.md](./ARCHITECTURE.md#extension-guide) - Extension guide
2. Look up: [API_REFERENCE.md](./API_REFERENCE.md#common-workflows) - Examples
3. Check: [TYPE_SYSTEM.md](./TYPE_SYSTEM.md) - Protocol definitions

### New Team Members

**First Week:**
1. Read: [README.md](../README.md) - Overview
2. Read: [USAGE_GUIDE_COLAB_AND_CLI.md](./USAGE_GUIDE_COLAB_AND_CLI.md) - Hands-on usage
3. Skim: [ARCHITECTURE.md](./ARCHITECTURE.md) - Design overview

**Second Week:**
1. Deep dive: [API_REFERENCE.md](./API_REFERENCE.md) - Learn components
2. Read: [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Team patterns
3. Run: Example notebooks in `docs/examples/`

**Third Week & Beyond:**
1. Pick specialty: [JOB_QUEUE_GUIDE.md](./JOB_QUEUE_GUIDE.md), [MODEL_REGISTRY.md](./MODEL_REGISTRY.md), etc.
2. Review: [ARCHITECTURE.md](./ARCHITECTURE.md) - Deep design understanding
3. Contribute: Extensions following [Extension Guide](./ARCHITECTURE.md#extension-guide)

---

## By Topic

### Training Configuration
- **Quick Start:** [BEST_PRACTICES.md#configuration](./BEST_PRACTICES.md#configuration-best-practices)
- **Full API:** [API_REFERENCE.md#trainingconfig](./API_REFERENCE.md#trainingconfig)
- **Migration:** [MIGRATION_GUIDE.md#scenario-1](./MIGRATION_GUIDE.md#scenario-1-update-trainingconfig-recommended-start)

### Model Architecture & Task Specification
- **Quick Start:** [API_REFERENCE.md#taskspec](./API_REFERENCE.md#taskspec)
- **Examples:** [API_REFERENCE.md#common-workflows](./API_REFERENCE.md#common-workflows)
- **Guide:** [DATA_LOADING_GUIDE.md](./DATA_LOADING_GUIDE.md)

### Loss Functions
- **Overview:** [API_REFERENCE.md#lossstrategy](./API_REFERENCE.md#lossstrategy)
- **Custom Loss:** [ARCHITECTURE.md#adding-a-custom-loss-strategy](./ARCHITECTURE.md#adding-a-custom-loss-strategy)
- **Migration:** [MIGRATION_GUIDE.md#scenario-5](./MIGRATION_GUIDE.md#scenario-5-custom-loss-functions)

### Checkpointing & Recovery
- **Guide:** [API_REFERENCE.md#checkpointmanager](./API_REFERENCE.md#checkpointmanager)
- **Best Practices:** [BEST_PRACTICES.md#checkpoint](./BEST_PRACTICES.md#checkpoint--recovery-best-practices)
- **Migration:** [MIGRATION_GUIDE.md#scenario-3](./MIGRATION_GUIDE.md#scenario-3-checkpoint-resume)

### Metrics & Monitoring
- **Guide:** [METRICS_ENGINE_MIGRATION.md](./METRICS_ENGINE_MIGRATION.md)
- **API Reference:** [API_REFERENCE.md#metricsengine](./API_REFERENCE.md#metricsengine)
- **Best Practices:** [BEST_PRACTICES.md#metrics](./BEST_PRACTICES.md#metrics--monitoring-best-practices)

### Data Loading
- **Quick Start:** [USAGE_GUIDE_COLAB_AND_CLI.md](./USAGE_GUIDE_COLAB_AND_CLI.md)
- **Full Guide:** [DATA_LOADING_GUIDE.md](./DATA_LOADING_GUIDE.md)
- **API:** [API_REFERENCE.md#data-loading](./API_REFERENCE.md#data-loading)

### Gradient Management
- **API Reference:** [API_REFERENCE.md#gradientmonitor](./API_REFERENCE.md#gradientmonitor)
- **Accumulation Guide:** [docs/gradient_accumulation_guide.md](./gradient_accumulation_guide.md)
- **Best Practices:** [BEST_PRACTICES.md#training](./BEST_PRACTICES.md#training-best-practices)

### Experiment Tracking
- **W&B Integration:** [METRICS_ENGINE_MIGRATION.md](./METRICS_ENGINE_MIGRATION.md)
- **Local Tracking:** [API_REFERENCE.md#experimentdb](./API_REFERENCE.md#experimentdb)
- **Best Practices:** [BEST_PRACTICES.md#team-collaboration](./BEST_PRACTICES.md#team-collaboration-best-practices)

### Model Versioning & Registry
- **Full Guide:** [MODEL_REGISTRY.md](./MODEL_REGISTRY.md)
- **API Reference:** [API_REFERENCE.md#modelregistry](./API_REFERENCE.md#modelregistry)
- **Migration:** [MIGRATION_GUIDE.md#scenario-7](./MIGRATION_GUIDE.md#scenario-7-model-registry--versioning)

### Export & Deployment
- **API Reference:** [API_REFERENCE.md#exportbundle](./API_REFERENCE.md#export-bundle)
- **Best Practices:** [BEST_PRACTICES.md#export](./BEST_PRACTICES.md#export--deployment-best-practices)
- **Migration:** [MIGRATION_GUIDE.md#scenario-6](./MIGRATION_GUIDE.md#scenario-6-export--deployment)

### Job Scheduling & Automation
- **Full Guide:** [JOB_QUEUE_GUIDE.md](./JOB_QUEUE_GUIDE.md)
- **API Reference:** [API_REFERENCE.md#jobqueue](./API_REFERENCE.md#jobqueue--scheduler)
- **Best Practices:** [BEST_PRACTICES.md#operations](./BEST_PRACTICES.md#production-operations-best-practices)

### Hyperparameter Tuning
- **Best Practices:** [BEST_PRACTICES.md#tuning](./BEST_PRACTICES.md#hyperparameter-tuning-best-practices)
- **Examples:** [API_REFERENCE.md#common-workflows](./API_REFERENCE.md#workflow-5-scheduled-retraining)

### Colab-Specific
- **Setup Guide:** [USAGE_GUIDE_COLAB_AND_CLI.md](./USAGE_GUIDE_COLAB_AND_CLI.md)
- **Troubleshooting:** [COLAB_TROUBLESHOOTING.md](./COLAB_TROUBLESHOOTING.md)
- **Validation:** [COLAB_VALIDATION_INSTRUCTIONS.md](./COLAB_VALIDATION_INSTRUCTIONS.md)
- **Validation Report:** [COLAB_VALIDATION_REPORT.md](./COLAB_VALIDATION_REPORT.md)

---

## API Reference by Component

### Phase 0: Core Engine
| Component | Responsibility | Location |
|-----------|-----------------|----------|
| **CheckpointManager** | State persistence | [API](./API_REFERENCE.md#checkpointmanager) |
| **LossStrategy** | Task-specific loss | [API](./API_REFERENCE.md#lossstrategy) |
| **GradientMonitor** | Gradient health checks | [API](./API_REFERENCE.md#gradientmonitor) |
| **GradientAccumulator** | Gradient accumulation | [API](./API_REFERENCE.md#gradientaccumulator) |
| **DataLoaderFactory** | DataLoader creation | [API](./API_REFERENCE.md#dataloaderfactory) |
| **CollatorRegistry** | Data collation strategies | [API](./API_REFERENCE.md#collatorregistry) |
| **TrainingLoop** | Training epoch execution | [API](./API_REFERENCE.md#training-and-validation-loops) |
| **ValidationLoop** | Validation epoch execution | [API](./API_REFERENCE.md#training-and-validation-loops) |
| **MetricsEngine** | Metrics + drift detection | [API](./API_REFERENCE.md#metricsengine) |
| **Trainer** | Orchestrator | [API](./API_REFERENCE.md#trainer) |

### Configuration
| Component | Responsibility | Location |
|-----------|-----------------|----------|
| **TrainingConfig** | Config dataclass | [API](./API_REFERENCE.md#trainingconfig) |
| **TrainingConfigBuilder** | Fluent API | [API](./API_REFERENCE.md#trainingconfigbuilder) |
| **TaskSpec** | Task specification | [API](./API_REFERENCE.md#taskspec) |

### Phase 2: Production
| Component | Responsibility | Location |
|-----------|-----------------|----------|
| **ModelRegistry** | Model versioning | [API](./API_REFERENCE.md#modelregistry) & [Guide](./MODEL_REGISTRY.md) |
| **JobQueue** | Job management | [API](./API_REFERENCE.md#jobqueue) & [Guide](./JOB_QUEUE_GUIDE.md) |
| **JobScheduler** | Recurring jobs | [API](./API_REFERENCE.md#jobqueue) & [Guide](./JOB_QUEUE_GUIDE.md) |
| **ExportBundle** | Deployment artifacts | [API](./API_REFERENCE.md#export-bundle) |
| **RetrainingTriggers** | Auto-retrain logic | [API](./API_REFERENCE.md#retraining-triggers) |

### Utilities
| Component | Responsibility | Location |
|-----------|-----------------|----------|
| **MetricsTracker** | Legacy metrics (v3.x) | [API](./API_REFERENCE.md#metricstracker) |
| **ExperimentDB** | Local experiment tracking | [API](./API_REFERENCE.md#experimentdb) |
| **SeedManager** | Reproducibility | [API](./API_REFERENCE.md#seedmanager) |

---

## Code Examples

### Basic Training
```python
from utils.training.engine import Trainer
from utils.training.training_config import TrainingConfigBuilder
from utils.training.task_spec import TaskSpec

config = TrainingConfigBuilder.baseline().build()
task_spec = TaskSpec.language_modeling(name='wikitext')
trainer = Trainer(model, model_config, config, task_spec)
results = trainer.train(train_data, val_data)
```
**See:** [API_REFERENCE.md#workflow-1](./API_REFERENCE.md#workflow-1-simple-training-loop)

### Custom Loss Strategy
**See:** [API_REFERENCE.md#workflow-2](./API_REFERENCE.md#workflow-2-custom-loss-strategy)

### Model Registry & Versioning
**See:** [API_REFERENCE.md#workflow-3](./API_REFERENCE.md#workflow-3-model-registry--versioning)

### Export Bundle
**See:** [API_REFERENCE.md#workflow-4](./API_REFERENCE.md#workflow-4-export-bundle)

### Scheduled Retraining
**See:** [API_REFERENCE.md#workflow-5](./API_REFERENCE.md#workflow-5-scheduled-retraining)

---

## Document Status

| Document | Status | Lines | Last Updated |
|----------|--------|-------|--------------|
| [README.md](../README.md) | Updated | 100+ | 2025-11-20 |
| [USAGE_GUIDE_COLAB_AND_CLI.md](./USAGE_GUIDE_COLAB_AND_CLI.md) | Updated | 500+ | 2025-11-20 |
| [API_REFERENCE.md](./API_REFERENCE.md) | New | 2000+ | 2025-11-20 |
| [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) | Updated | 800+ | 2025-11-20 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | New | 600+ | 2025-11-20 |
| [BEST_PRACTICES.md](./BEST_PRACTICES.md) | New | 600+ | 2025-11-20 |
| [INDEX.md](./INDEX.md) | New | 300+ | 2025-11-20 |
| [TYPE_SYSTEM.md](./TYPE_SYSTEM.md) | Existing | 200+ | 2025-11-20 |
| [DATA_LOADING_GUIDE.md](./DATA_LOADING_GUIDE.md) | Existing | 400+ | 2025-11-20 |
| [METRICS_ENGINE_MIGRATION.md](./METRICS_ENGINE_MIGRATION.md) | Existing | 300+ | 2025-11-20 |
| [JOB_QUEUE_GUIDE.md](./JOB_QUEUE_GUIDE.md) | Existing | 500+ | 2025-11-20 |
| [MODEL_REGISTRY.md](./MODEL_REGISTRY.md) | Existing | 400+ | 2025-11-20 |
| [COLAB_TROUBLESHOOTING.md](./COLAB_TROUBLESHOOTING.md) | Existing | 600+ | 2025-11-20 |

**Total Documentation:** 7000+ lines covering all aspects of the training engine.

---

## Related Resources

### Example Notebooks
- `template.ipynb` - Model validation and testing
- `training.ipynb` - Training utilities (Tier 3)
- `docs/examples/` - Additional examples (see directory)

### Community
- Issues: GitHub Issues (bug reports, feature requests)
- Discussions: GitHub Discussions (questions, ideas)
- Contributing: See CONTRIBUTING.md

### External References
- PyTorch Documentation: https://pytorch.org/docs/
- Weights & Biases: https://docs.wandb.ai/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/

---

## Version History

| Version | Release Date | Key Features | Migration |
|---------|--------------|--------------|-----------|
| v4.0 | 2025-11-20 | Modular engine, CheckpointManager, LossStrategy | From v3.x |
| v4.1 | Planned | MetricsEngine enhancements, Flash Attention | Backward compatible |
| v4.2 | Planned | ModelRegistry, JobQueue, ExportBundle | Backward compatible |
| v4.3 | Planned | Complete documentation, legacy_api | Backward compatible |

---

**Last Updated:** 2025-11-20
**Status:** Complete
**Maintainer:** MLOps & Documentation Teams
**License:** Apache 2.0
