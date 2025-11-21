# Training Pipeline Refactoring Plan v4.0

**Status:** Planning Phase
**Start Date:** 2025-01-20
**Estimated Completion:** 2025-03-24 (9 weeks)
**Document Version:** 1.0

---

## Executive Summary

### Project Goal
Transform the monolithic `utils/tier3_training_utilities.py` (1765 lines) into a production-grade, modular training engine capable of handling diverse AI architectures (Vision, Text, Classification) with robust error handling, comprehensive monitoring, and production deployment readiness.

### Critical Success Metrics
- ✅ Checkpoint/resume support for Colab 12-hour timeout
- ✅ 80%+ test coverage on all engine modules
- ✅ Type-safe (mypy --strict compliance)
- ✅ Backward compatible facade until v4.0
- ✅ Production-ready export bundles with health checks

### Current Problems Identified

**From ML Engineer Analysis:**
- Export bundles incomplete (no model registry, no health checks, static Dockerfile)
- LossStrategy missing edge cases (LoRA/PEFT, quantization, distributed training)
- Flash Attention not validated for all architectures (Vision Transformers, Longformer)
- Gradient accumulation conflicts with PyTorch Lightning's `accumulate_grad_batches`
- MetricsTracker missing drift detection and prediction confidence tracking

**From MLOps Engineer Analysis:**
- **CRITICAL:** Zero checkpointing in 1765-line training loop (data loss on Colab timeout)
- Refactor plan missing checkpoint.py specification
- No automated retraining triggers (drift-based, performance-based)
- No model registry for artifact versioning (only timestamp-based directories)
- No job queue/scheduler for multi-experiment workflows
- Requirements.txt sync validation missing in CI

**From Python Architect Analysis:**
- Factory function fragile (runtime errors on typos) → Use Registry Pattern
- `test_fine_tuning()` is God Function (30+ parameters) → Use Builder Pattern
- LossStrategy signature incomplete → Use Protocol + TypedDict
- No gradient health checks (NaN/Inf silent failures) → Add GradientMonitor
- Mixed abstraction levels in training loop → Delegate to specialized modules
- Primitive obsession in output parsing → Add ModelOutput dataclass

---

## Architecture Overview

### Current Structure (Monolithic)
```
utils/
└── tier3_training_utilities.py (1765 lines)
    ├── test_fine_tuning() (200+ lines)
    ├── test_hyperparameter_search()
    ├── test_benchmark_comparison()
    ├── 20+ helper functions
    └── Hardcoded Causal LM logic
```

**Problems:**
- Tightly coupled training logic
- Hardcoded token shifting breaks non-LM tasks
- No checkpointing (Colab timeout = data loss)
- Mixed abstraction levels (high-level + tensor ops in same function)
- 30+ function parameters (God Function antipattern)

### Target Structure (Modular)
```
utils/training/engine/
├── __init__.py              # Public API exports
├── trainer.py               # High-level orchestration (Trainer class)
├── loop.py                  # Epoch execution logic
├── data.py                  # DataLoader creation, collation strategies
├── loss.py                  # Task-aware loss computation (Strategy Pattern)
├── metrics.py               # Metrics tracking with drift detection
├── visualization.py         # Dashboard and plotting
└── checkpoint.py            # State management (save/load/resume)
```

**Benefits:**
- Single Responsibility Principle (each module <500 lines)
- Task-agnostic via Strategy Pattern
- Testable (dependency injection, mocking)
- Type-safe (mypy --strict compliance)
- Production-ready (checkpointing, monitoring, export bundles)

---

## Design Patterns

### 1. Strategy Pattern (Loss Computation)
**Problem:** Hardcoded Causal LM logic breaks Vision/Classification tasks

**Solution:**
```python
from typing import Protocol, TypedDict

class LossInputs(TypedDict, total=False):
    logits: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pad_token_id: Optional[int]

class LossStrategy(Protocol):
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        ...

class CausalLMLoss:
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        # Shift tokens for next-token prediction
        logits = inputs['logits'][:, :-1, :].contiguous()
        labels = inputs['labels'][:, 1:].contiguous()
        return F.cross_entropy(logits.view(-1, vocab), labels.view(-1))

class ClassificationLoss:
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        # No shifting for classification
        return F.cross_entropy(inputs['logits'], inputs['labels'])
```

### 2. Registry Pattern (Strategy Lookup)
**Problem:** Factory function has runtime errors on typos

**Solution:**
```python
class LossStrategyRegistry:
    _strategies: Dict[str, Type[LossStrategy]] = {}

    @classmethod
    def register(cls, task_type: str):
        def decorator(strategy_cls):
            cls._strategies[task_type] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, task_spec: TaskSpec) -> LossStrategy:
        if task_spec.task_type not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unknown task_type '{task_spec.task_type}'. Available: {available}")
        return cls._strategies[task_spec.task_type]()

# Usage
@LossStrategyRegistry.register("language_modeling")
class CausalLMLoss:
    ...
```

### 3. Builder Pattern (Configuration)
**Problem:** 30+ parameter God Function

**Solution:**
```python
@dataclass
class TrainingPipelineConfig:
    # Core training
    n_epochs: int = 5
    learning_rate: float = 5e-5
    batch_size: int = 4

    # Optimization
    gradient_accumulation_steps: int = 1
    compile_mode: Optional[str] = None

    # Export
    export_bundle: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["pytorch"])

    def validate(self):
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        # ... more validation

# Simplified API (3 parameters instead of 30+)
def test_fine_tuning(
    model: nn.Module,
    config: SimpleNamespace,
    training_config: TrainingPipelineConfig
) -> Dict[str, Any]:
    training_config.validate()
    trainer = Trainer(model, config, training_config)
    return trainer.train()
```

### 4. Protocol + TypedDict (Type Safety)
**Problem:** No type safety for model outputs (tuple/dict parsing)

**Solution:**
```python
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None

    @classmethod
    def from_raw(cls, output: Any) -> 'ModelOutput':
        """Parse raw model output into structured format."""
        if isinstance(output, torch.Tensor):
            return cls(logits=output)
        elif isinstance(output, tuple):
            return cls(logits=output[0], loss=output[1] if len(output) > 1 else None)
        elif isinstance(output, dict):
            return cls(logits=output['logits'], loss=output.get('loss'))
        # ... handle HuggingFace ModelOutput
        raise TypeError(f"Cannot parse output type: {type(output)}")
```

---

## Phase 1: Critical Blockers (Week 1-2)

### Task P0-1: Design and Implement Checkpoint System
**Agent:** `python-development:python-pro`
**Effort:** 3 days
**Dependencies:** None

**Description:**
Create `utils/training/engine/checkpoint.py` with comprehensive state management for training resume. Must handle model state, optimizer, scheduler, epoch counters, RNG state, and best metrics tracking.

**Why This is P0:**
The current tier3_training_utilities.py (1765 lines) has **zero checkpointing**. This is a critical failure mode for Colab's 12-hour timeout. Users lose all training progress if disconnected.

**Acceptance Criteria:**
- [ ] `CheckpointManager` class with `save()`, `load()`, `list_checkpoints()`, `get_best()` methods
- [ ] Supports both epoch-based and step-based checkpointing
- [ ] Preserves all RNG states (Python, NumPy, PyTorch, CUDA) for reproducibility
- [ ] Configurable retention policy (keep_best_k=3, keep_last_n=5)
- [ ] Validates checkpoint integrity (detects corrupted files)
- [ ] Protocol for custom state injection (LossStrategy state, MetricsTracker state)
- [ ] Unit tests with mocked model/optimizer: test save→load→resume equivalence
- [ ] Type hints with `TypedDict` for checkpoint structure
- [ ] Google Drive backup support for Colab persistence
- [ ] Documentation with usage examples

**Test Scenarios:**
1. Save checkpoint after epoch 5 → Load → Verify epoch counter = 6
2. Save with optimizer state → Load → Verify learning rate preserved
3. Corrupt checkpoint file → Load → Raises clear error with recovery steps
4. Multiple checkpoints → `get_best()` → Returns checkpoint with lowest val_loss
5. Save with custom state (loss strategy config) → Load → Custom state restored
6. RNG state preservation: train 10 steps, checkpoint, resume → identical random outputs

**Technical Implementation:**
```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import torch

@dataclass
class CheckpointMetadata:
    epoch: int
    global_step: int
    best_metric: float
    timestamp: str
    git_commit: Optional[str] = None

class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_k: int = 3,
        keep_last_n: int = 5,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_best_k = keep_best_k
        self.keep_last_n = keep_last_n
        self.monitor = monitor
        self.mode = mode
        self.checkpoints: List[CheckpointMetadata] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        custom_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save checkpoint with full training state."""
        # ... implementation

    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint and return state dict."""
        # ... implementation

    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint based on monitor metric."""
        # ... implementation
```

**Dependencies:**
- None (foundational module)

**Risks & Mitigations:**
- **Risk:** Checkpoint files become too large (>500MB) for Colab disk
  - **Mitigation:** Compression, selective state saving, Drive backup
- **Risk:** Checkpoint corruption from interrupted save
  - **Mitigation:** Atomic write (save to .tmp, rename on success)

---

### Task P0-2: Implement LossStrategy Protocol with Type Safety
**Agent:** `python-development:python-pro`
**Effort:** 2 days
**Dependencies:** None

**Description:**
Refactor `utils/training/engine/loss.py` with Protocol-based LossStrategy interface. Add implementations for language modeling, classification, PEFT/LoRA, quantized models, and distributed training edge cases.

**Why This is P0:**
Current factory function (`get_loss_strategy(task_spec)`) has runtime errors on typos. Missing PEFT/quantization support blocks production use. Type safety prevents silent failures.

**Acceptance Criteria:**
- [ ] `LossStrategy` Protocol with `compute_loss(batch, model_output, config)` signature
- [ ] `TypedDict` for `LossBatch` (input_ids, labels, attention_mask, pixel_values)
- [ ] `ModelOutputProtocol` dataclass replacing tuple/dict parsing
- [ ] Implementations: `LanguageModelingLoss`, `ClassificationLoss`, `PEFTAwareLoss`, `QuantizationSafeLoss`
- [ ] Edge case handling: missing labels, attention mask padding exclusion, logit shape mismatches
- [ ] Registry pattern for strategy lookup (replaces fragile factory function)
- [ ] Unit tests with synthetic batches: verify padding exclusion, PEFT parameter freezing
- [ ] Type-safe: passes mypy --strict
- [ ] Documentation with usage examples for each strategy
- [ ] Performance: <5ms overhead per loss computation

**Test Scenarios:**
1. LanguageModelingLoss with padding → Verify padding tokens excluded from loss
2. ClassificationLoss with imbalanced dataset → Apply class weights correctly
3. PEFTAwareLoss with frozen base model → Verify gradients only on adapter parameters
4. QuantizationSafeLoss with 4-bit model → Dequantize before loss computation
5. Missing attention_mask → Raises clear error with fix suggestion
6. Logit shape mismatch (batch=4, but labels=8) → Raises clear error

**Technical Implementation:**
```python
from typing import Protocol, TypedDict, Optional
import torch

class LossInputs(TypedDict, total=False):
    logits: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pad_token_id: Optional[int]
    class_weights: Optional[torch.Tensor]

class LossStrategy(Protocol):
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        """Compute task-specific loss."""
        ...

@LossStrategyRegistry.register("language_modeling")
class LanguageModelingLoss:
    def compute_loss(self, inputs: LossInputs) -> torch.Tensor:
        logits = inputs['logits']
        labels = inputs['labels']
        pad_token_id = inputs.get('pad_token_id', 0)

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id
        )
```

**Dependencies:**
- None (foundational module)

**Risks & Mitigations:**
- **Risk:** Protocol overhead slows down training
  - **Mitigation:** Benchmark to ensure <5ms overhead, use @torch.jit.script if needed
- **Risk:** Missing strategy for new task type causes runtime error
  - **Mitigation:** Registry validates at registration, provides clear error with available strategies

---

### Task P0-3: Extract Gradient Monitoring and Health Checks
**Agent:** `python-development:python-pro`
**Effort:** 1.5 days
**Dependencies:** Task P0-2 (needs ModelOutputProtocol)

**Description:**
Create `utils/training/engine/gradient_monitor.py` to detect NaN/Inf gradients, gradient vanishing/explosion, and parameter update health. Integrates with training loop for early warning system.

**Why This is P0:**
Silent NaN failures corrupt training runs. Current code has no gradient health checks. Early detection saves hours of debugging.

**Acceptance Criteria:**
- [ ] `GradientMonitor` class with `check_gradients(model)` method
- [ ] Returns `GradientHealth` dataclass: has_nan, has_inf, max_norm, min_norm, affected_layers
- [ ] Configurable thresholds for vanishing (<1e-7) and explosion (>10.0)
- [ ] Integration hook for training loop: log warnings and optionally halt training
- [ ] Gradient histogram logging to W&B (optional, disabled by default)
- [ ] Unit tests with synthetic gradients: inject NaN, test layer-wise detection
- [ ] Performance optimized: <5ms overhead per check
- [ ] Consecutive failure tracking (halt after 3 NaN gradients in a row)
- [ ] Documentation with troubleshooting guide
- [ ] Type-safe: passes mypy --strict

**Test Scenarios:**
1. Inject NaN gradient in layer 'fc.weight' → Detects NaN, logs affected layer
2. 3 consecutive NaN gradients → Raises RuntimeError with remediation steps
3. Gradient norm 1e-9 (vanishing) → Logs warning, suggests LR increase
4. Gradient norm 50.0 (explosion) → Logs warning, suggests LR decrease
5. Healthy gradients (norm 0.1-1.0) → No warnings, minimal overhead
6. Large model (1B params) → Check completes in <50ms

**Technical Implementation:**
```python
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn

@dataclass
class GradientHealth:
    has_nan: bool
    has_inf: bool
    max_norm: float
    min_norm: float
    affected_layers: List[str]
    is_healthy: bool

class GradientMonitor:
    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        explosion_threshold: float = 10.0,
        max_consecutive_failures: int = 3
    ):
        self.vanishing_threshold = vanishing_threshold
        self.explosion_threshold = explosion_threshold
        self.max_consecutive_failures = max_consecutive_failures
        self.nan_count = 0
        self.inf_count = 0

    def check_gradients(self, model: nn.Module) -> GradientHealth:
        """Check gradient health and detect anomalies."""
        has_nan = False
        has_inf = False
        max_norm = 0.0
        min_norm = float('inf')
        affected_layers = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()

            if torch.isnan(grad).any():
                has_nan = True
                affected_layers.append(f"{name} (NaN)")

            if torch.isinf(grad).any():
                has_inf = True
                affected_layers.append(f"{name} (Inf)")

            norm = grad.norm().item()
            max_norm = max(max_norm, norm)
            min_norm = min(min_norm, norm)

        # Track consecutive failures
        if has_nan:
            self.nan_count += 1
        else:
            self.nan_count = 0

        if self.nan_count >= self.max_consecutive_failures:
            raise RuntimeError(
                f"Training unstable: {self.nan_count} consecutive NaN gradients. "
                "Try: (1) Lower learning rate, (2) Check loss computation, "
                "(3) Enable gradient clipping"
            )

        is_healthy = not (has_nan or has_inf)

        return GradientHealth(
            has_nan=has_nan,
            has_inf=has_inf,
            max_norm=max_norm,
            min_norm=min_norm,
            affected_layers=affected_layers,
            is_healthy=is_healthy
        )
```

**Dependencies:**
- Task P0-2 (ModelOutputProtocol for loss computation integration)

**Risks & Mitigations:**
- **Risk:** Overhead slows training (checking every batch)
  - **Mitigation:** Optimize checks, make frequency configurable (every N batches)
- **Risk:** False positives on valid gradients
  - **Mitigation:** Configurable thresholds, log warnings instead of errors

---

### Task P0-4: Fix Gradient Accumulation Conflict with PyTorch Lightning
**Agent:** `machine-learning-ops:ml-engineer`
**Effort:** 2 days
**Dependencies:** Task P0-3 (gradient monitoring integration)

**Description:**
Resolve conflict between manual gradient accumulation in `tier3_training_utilities.py` and PyTorch Lightning's `accumulate_grad_batches`. Ensure MetricsTracker step counting aligns with actual optimizer updates.

**Why This is P0:**
Current implementation causes incorrect step counts in W&B logging. Manual accumulation + Lightning accumulation = double accumulation (2x effective batch size). Critical for production workflows.

**Acceptance Criteria:**
- [ ] Detection logic: if `Trainer` is Lightning instance, delegate to `accumulate_grad_batches`
- [ ] Manual accumulation loop refactored to avoid double-accumulation
- [ ] MetricsTracker.effective_step calculation validated against Lightning's global_step
- [ ] Integration test: Lightning trainer + manual accumulation → verify step count alignment
- [ ] Documentation: when to use manual vs Lightning accumulation
- [ ] Backward compatible: existing code without Lightning works unchanged
- [ ] Unit tests: verify step counting with accumulation=1,4,8
- [ ] W&B logging validation: verify metrics logged at correct steps
- [ ] Performance: no overhead when accumulation=1
- [ ] Type-safe: passes mypy --strict

**Test Scenarios:**
1. Manual accumulation (steps=4) → Optimizer updates every 4 batches, correct step count
2. Lightning trainer (accumulate_grad_batches=4) → No double accumulation
3. Both manual + Lightning → Raises clear error with resolution steps
4. MetricsTracker with accumulation=4 → effective_step = step // 4
5. W&B logging with accumulation → Commits only at accumulation boundaries
6. Single GPU vs multi-GPU → Step counts consistent

**Technical Implementation:**
```python
from typing import Optional
import pytorch_lightning as pl

class GradientAccumulator:
    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_steps: int = 1,
        trainer: Optional[pl.Trainer] = None
    ):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.trainer = trainer
        self.current_step = 0

        # Detect conflict
        if trainer is not None and trainer.accumulate_grad_batches > 1:
            if accumulation_steps > 1:
                raise ValueError(
                    "Cannot use both manual accumulation_steps and Lightning's "
                    "accumulate_grad_batches. Choose one: "
                    "(1) Manual: set accumulation_steps, trainer=None, "
                    "(2) Lightning: set accumulate_grad_batches, accumulation_steps=1"
                )

    def step(self, loss: torch.Tensor) -> bool:
        """Accumulate gradients and step optimizer when ready."""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self.current_step += 1

        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True  # Optimizer stepped

        return False  # Still accumulating
```

**Dependencies:**
- Task P0-3 (GradientMonitor integration for health checks)

**Risks & Mitigations:**
- **Risk:** Breaking change for existing Lightning users
  - **Mitigation:** Detection + clear error message + documentation
- **Risk:** Step count mismatch in distributed training
  - **Mitigation:** Test with DDP, verify global_step synchronization

---

### Task P0-5: Requirements Sync Validation in CI
**Agent:** `machine-learning-ops:mlops-engineer`
**Effort:** 1 day
**Dependencies:** None

**Description:**
Add GitHub Actions workflow to validate `requirements.txt`, `requirements-training.txt`, and `requirements-colab-v3.4.0.txt` are in sync (where applicable) and that all imports in codebase are declared.

**Why This is P0:**
Requirements drift causes "works on my machine" bugs. Manual sync is error-prone. CI validation prevents dependency conflicts and missing imports.

**Acceptance Criteria:**
- [ ] CI job: `.github/workflows/validate-requirements.yml`
- [ ] Script: `scripts/check_requirements_sync.py`
- [ ] Checks: training.txt ⊆ colab.txt (training section only), all imports in `utils/` declared
- [ ] Detects drift: version mismatches, missing dependencies
- [ ] Fails PR if validation fails (blocking)
- [ ] Exemption list for stdlib and intentionally omitted packages
- [ ] Runs on push to `main` and all PRs
- [ ] Documentation: how to update requirements properly
- [ ] Exit code 0 on success, 1 on failure (CI integration)
- [ ] Clear error messages with fix suggestions

**Test Scenarios:**
1. `torch==2.0.0` in requirements.txt, `torch>=1.9.0` in colab → Pass (compatible)
2. `torch==1.8.0` in requirements.txt, `torch>=2.0.0` in colab → Fail (incompatible)
3. `import numpy` in code, missing from requirements.txt → Fail
4. `import sys` (stdlib) in code, missing from requirements → Pass (exempted)
5. All requirements in sync → Pass, exit 0
6. Version mismatch → Fail with diff output and fix command

**Technical Implementation:**
```yaml
# .github/workflows/validate-requirements.yml
name: Validate Requirements Sync

on:
  push:
    branches: [main]
  pull_request:

jobs:
  check-requirements:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install packaging
      - name: Run requirements sync check
        run: python scripts/check_requirements_sync.py
```

```python
# scripts/check_requirements_sync.py
from packaging import version
import sys

def check_version_compatibility():
    # Parse requirements files
    reqs_exact = parse_requirements('requirements.txt')
    reqs_colab = parse_requirements('requirements-colab-v3.4.0.txt')

    # Validate compatibility
    for pkg, ver_exact in reqs_exact.items():
        if pkg in reqs_colab:
            ver_min = reqs_colab[pkg]
            if not version.parse(ver_exact) >= version.parse(ver_min):
                print(f"❌ Version conflict: {pkg}")
                print(f"   requirements.txt: {ver_exact}")
                print(f"   requirements-colab: >={ver_min}")
                sys.exit(1)

    print("✅ All requirements in sync")
    sys.exit(0)

if __name__ == '__main__':
    check_version_compatibility()
```

**Dependencies:**
- None (standalone CI validation)

**Risks & Mitigations:**
- **Risk:** False positives block legitimate PRs
  - **Mitigation:** Exemption list, clear error messages with override instructions
- **Risk:** Slow CI runtime
  - **Mitigation:** Cache dependencies, optimize parsing

---

## Phase 2: Core Refactoring (Week 3-4)

### Task P1-1: Extract DataLoader and Collation Logic
**Agent:** `python-development:python-pro`
**Effort:** 2.5 days
**Dependencies:** Task P0-1 (checkpoint state for DataLoader epoch position)

**Description:**
Create `utils/training/engine/data.py` with modular data loading and collation. Separate concerns: dataset loading, batch collation, worker management.

**Acceptance Criteria:**
- [ ] `DataModuleProtocol` interface with `train_dataloader()`, `val_dataloader()` methods
- [ ] `CollatorRegistry` for task-specific collation (text, vision, multimodal)
- [ ] Worker seeding integrated with SeedManager for reproducibility
- [ ] Prefetching and pin_memory optimizations configurable
- [ ] VisionDataCollator with normalization strategy (ImageNet, CIFAR-10, custom)
- [ ] Unit tests: verify collation output shapes, worker seeding reproducibility
- [ ] Integration with TaskSpec: auto-detect collator from modality
- [ ] Documentation with examples for custom collators
- [ ] Performance: DataLoader overhead <2% of training time
- [ ] Type-safe: passes mypy --strict

---

### Task P1-2: Refactor Metrics Tracking with Drift Detection
**Agent:** `machine-learning-ops:ml-engineer`
**Effort:** 3 days
**Dependencies:** Task P0-1 (checkpoint includes metrics state)

**Description:**
Enhance `utils/training/engine/metrics.py` (extracted from MetricsTracker) with drift detection, prediction confidence tracking, and model performance degradation alerts.

**Acceptance Criteria:**
- [ ] `MetricsEngine` class separated from visualization concerns
- [ ] Drift detection: JS divergence between train/val distributions
- [ ] Confidence tracking: log prediction probabilities (top-1, top-5, entropy)
- [ ] Performance alerts: configurable thresholds for accuracy drop, loss spike
- [ ] Integration with ExperimentDB: log drift scores per epoch
- [ ] Unit tests: synthetic drift scenario, alert trigger verification
- [ ] W&B custom charts for confidence histograms
- [ ] Documentation with alert configuration guide
- [ ] Performance: metrics overhead <1% of training time
- [ ] Type-safe: passes mypy --strict

---

### Task P1-3: Create Training Loop Orchestrator
**Agent:** `backend-development:backend-architect`
**Effort:** 2 days
**Dependencies:** Task P0-1 (CheckpointManager), Task P1-2 (MetricsEngine)

**Description:**
Design `utils/training/engine/trainer.py` as high-level orchestrator. Delegates to Loop, Checkpoint, Metrics, Visualization. Clear separation of concerns.

**Acceptance Criteria:**
- [ ] `Trainer` class with single-responsibility: workflow orchestration
- [ ] Delegates epoch execution to `TrainingLoop`
- [ ] Delegates checkpointing to `CheckpointManager`
- [ ] Delegates metrics to `MetricsEngine`
- [ ] Configurable via `TrainingConfig` (no 30+ parameter constructor)
- [ ] Supports hooks: on_epoch_start, on_batch_end, on_validation_end
- [ ] Integration test: end-to-end training with mocked components
- [ ] Type-safe: passes mypy --strict
- [ ] Documentation with architecture diagram
- [ ] Backward compatible: works with existing configs

---

### Task P1-4: Extract Training Loop Execution
**Agent:** `python-development:python-pro`
**Effort:** 3 days
**Dependencies:** Task P0-2 (LossStrategy), Task P0-3 (GradientMonitor), Task P0-4 (accumulation fix)

**Description:**
Create `utils/training/engine/loop.py` with single responsibility: execute one epoch of training. Delegates batch processing, gradient handling, loss computation.

**Acceptance Criteria:**
- [ ] `TrainingLoop` class with `train_epoch(model, dataloader, optimizer, loss_strategy)` method
- [ ] Returns `EpochResult` dataclass: loss, metrics, duration
- [ ] Integrates GradientMonitor for health checks
- [ ] Integrates gradient accumulation (manual + Lightning compatibility)
- [ ] Supports mixed precision training (torch.amp)
- [ ] Handles exceptions: OOM, NaN loss, keyboard interrupt
- [ ] Unit tests: synthetic batches, verify gradient flow
- [ ] Progress bar integration (tqdm or Lightning's ProgressBar)
- [ ] Documentation with usage examples
- [ ] Type-safe: passes mypy --strict

---

### Task P1-5: Implement Model Output Dataclass
**Agent:** `python-development:python-pro`
**Effort:** 1.5 days
**Dependencies:** Task P0-2 (LossStrategy uses ModelOutput)

**Description:**
Replace tuple/dict parsing with `ModelOutput` dataclass for type-safe output handling. Supports HuggingFace, custom models, vision transformers.

**Acceptance Criteria:**
- [ ] `ModelOutput` dataclass with `logits`, `loss`, `hidden_states`, `attentions` fields
- [ ] Factory method: `ModelOutput.from_raw(output)` handles tensors, tuples, dicts, HF objects
- [ ] Protocol: `ModelOutputProtocol` for custom models
- [ ] Validation: shape checking, device consistency
- [ ] Unit tests: parse all output formats, verify type safety
- [ ] Integration with LossStrategy: pass ModelOutput instead of raw tensors
- [ ] Backward compatible: existing tuple unpacking still works (deprecation warning)
- [ ] Documentation with examples for custom model outputs
- [ ] Performance: parsing overhead <1ms
- [ ] Type-safe: passes mypy --strict

---

### Task P1-6: Registry Pattern for Strategy Lookup
**Agent:** `python-development:python-pro`
**Effort:** 2 days
**Dependencies:** Task P0-2 (LossStrategy), Task P1-1 (CollatorRegistry)

**Description:**
Replace fragile factory functions with Registry pattern for LossStrategy, CollatorStrategy, ExportStrategy lookups.

**Acceptance Criteria:**
- [ ] `StrategyRegistry` generic class with `register()`, `get()`, `list_available()` methods
- [ ] Decorator: `@registry.register("name")` for strategy classes
- [ ] Type-safe: `Registry[T]` with Protocol constraint
- [ ] Error handling: clear error on missing strategy, suggestions for typos (Levenshtein distance)
- [ ] Unit tests: registration, lookup, typo handling
- [ ] Refactor loss.py, data.py, export_utilities.py to use registry
- [ ] Documentation: how to register custom strategies
- [ ] Performance: lookup overhead <0.1ms
- [ ] Thread-safe for multi-GPU training
- [ ] Type-safe: passes mypy --strict

---

## Phase 3: Production Hardening (Week 5-6)

### Task P2-1: Complete Export Bundle with Health Checks
**Agent:** `machine-learning-ops:ml-engineer`
**Effort:** 4 days
**Dependencies:** Task P1-5 (ModelOutput for inference), Task P2-2 (model registry)

**Description:**
Enhance `utils/training/export_utilities.py` with health check endpoints, dynamic Dockerfile generation, model registry integration, and TorchServe validation.

**Acceptance Criteria:**
- [ ] `inference.py` includes `/health` and `/ready` endpoints
- [ ] Dockerfile dynamically generated based on model requirements (GPU, quantization)
- [ ] TorchServe config validation before MAR creation
- [ ] Model registry integration: upload to MLflow, W&B Artifacts, or S3
- [ ] Export validation: load and test inference before bundle finalization
- [ ] Unit tests: export all formats, validate generated files
- [ ] Integration test: deploy bundle to local TorchServe instance
- [ ] Documentation: deployment guide for Docker, TorchServe, K8s
- [ ] Security: no credentials in generated artifacts
- [ ] Multi-format support: ONNX, TorchScript, PyTorch state dict

---

### Task P2-2: Implement Model Registry for Artifact Versioning
**Agent:** `machine-learning-ops:mlops-engineer`
**Effort:** 3 days
**Dependencies:** Task P0-1 (CheckpointManager provides best model)

**Description:**
Create `utils/training/model_registry.py` with versioned artifact storage. Supports MLflow, W&B Artifacts, S3, and local filesystem.

**Acceptance Criteria:**
- [ ] `ModelRegistry` interface with `register()`, `get()`, `list_versions()`, `promote()` methods
- [ ] Implementations: `MLflowRegistry`, `WandBRegistry`, `S3Registry`, `LocalRegistry`
- [ ] Metadata: experiment ID, metrics, config, Git commit hash
- [ ] Promotion workflow: dev → staging → production
- [ ] Version comparison: side-by-side metric comparison
- [ ] Unit tests: mock registry operations, version lifecycle
- [ ] Integration with CheckpointManager: auto-register best checkpoints
- [ ] Documentation: registry setup for each backend
- [ ] CLI tool: `python -m utils.training.model_registry promote v1.2.0 prod`
- [ ] Type-safe: passes mypy --strict

---

### Task P2-3: Flash Attention Validation for All Architectures
**Agent:** `machine-learning-ops:ml-engineer`
**Effort:** 3 days
**Dependencies:** None (enhances existing Flash Attention)

**Description:**
Validate Flash Attention (SDPA) support for Vision Transformers, Longformer, encoder-only, decoder-only architectures. Add fallback detection and testing.

**Acceptance Criteria:**
- [ ] Architecture detection: identify ViT, Longformer, BERT, GPT variants
- [ ] Compatibility matrix: test SDPA on all architectures with synthetic inputs
- [ ] Fallback logic: disable SDPA for incompatible layers (sliding window attention)
- [ ] Performance benchmarks: measure speedup for each architecture (T4, A100)
- [ ] Unit tests: verify correctness (SDPA output == standard attention output)
- [ ] Documentation: compatibility table, expected speedups
- [ ] Integration test: train ViT with SDPA enabled, validate accuracy
- [ ] Warning messages: clear logs when SDPA disabled with reason
- [ ] Backward compatible: existing models work unchanged
- [ ] Type-safe: passes mypy --strict

---

### Task P2-4: Automated Retraining Triggers
**Agent:** `machine-learning-ops:mlops-engineer`
**Effort:** 2.5 days
**Dependencies:** Task P1-2 (drift detection), Task P2-2 (model registry)

**Description:**
Create `utils/training/retraining_triggers.py` with drift-based and performance-based retraining automation. Integrates with job scheduler.

**Acceptance Criteria:**
- [ ] `RetrainingTrigger` interface with `should_retrain(metrics, drift_scores)` method
- [ ] Implementations: `DriftTrigger` (JS distance threshold), `PerformanceTrigger` (accuracy drop)
- [ ] Configurable thresholds: drift > 0.2, accuracy drop > 5%
- [ ] Job queue integration: submit retraining job to Ray, Celery, or simple cron
- [ ] Notification system: email/Slack alerts on trigger activation
- [ ] Unit tests: synthetic drift/performance scenarios
- [ ] Integration test: trigger → job submission → training execution
- [ ] Documentation: setup guide for each notification backend
- [ ] Dry-run mode: log triggers without executing retraining
- [ ] Type-safe: passes mypy --strict

---

### Task P2-5: Job Queue and Scheduler Integration
**Agent:** `machine-learning-ops:mlops-engineer`
**Effort:** 3 days
**Dependencies:** Task P2-4 (retraining triggers submit jobs)

**Description:**
Create `utils/training/scheduler.py` for multi-experiment workflows. Supports Ray, Celery, or simple process-based scheduling.

**Acceptance Criteria:**
- [ ] `ExperimentScheduler` class with `submit()`, `cancel()`, `list_jobs()` methods
- [ ] Implementations: `RayScheduler`, `CeleryScheduler`, `ProcessScheduler`
- [ ] Priority queue: P0 (urgent) → P1 (normal) → P2 (background)
- [ ] Resource management: GPU allocation, memory limits
- [ ] Job persistence: survive scheduler restarts
- [ ] Status tracking: queued, running, completed, failed
- [ ] Unit tests: job lifecycle, priority ordering
- [ ] Integration test: submit 10 jobs, verify execution order
- [ ] CLI tool: `python -m utils.training.scheduler status`
- [ ] Documentation: setup for Ray, Celery, process-based

---

### Task P2-6: Builder Pattern for Training Configuration
**Agent:** `python-development:python-pro`
**Effort:** 2 days
**Dependencies:** Task P1-3 (Trainer uses TrainingConfig)

**Description:**
Replace 30+ parameter `test_fine_tuning()` function with fluent Builder pattern. Improves readability and reduces parameter errors.

**Acceptance Criteria:**
- [ ] `TrainingConfigBuilder` class with fluent API: `.with_learning_rate()`, `.with_batch_size()`, `.build()`
- [ ] Validation: required parameters, value ranges, mutually exclusive options
- [ ] Presets: `.from_preset("baseline")`, `.from_preset("production")`
- [ ] Backward compatibility: existing function signature still works (calls builder internally)
- [ ] Unit tests: build valid config, catch invalid combinations
- [ ] Documentation: examples of common configurations
- [ ] Type-safe: passes mypy --strict
- [ ] Serializable: `.to_dict()`, `.from_dict()` methods
- [ ] CLI integration: parse config from JSON/YAML files
- [ ] Performance: builder overhead <1ms

---

## Phase 4: Testing & Migration (Week 7-8)

### Task P3-1: Comprehensive Unit Test Suite
**Agent:** `task-developer`
**Effort:** 4 days
**Dependencies:** All P0, P1, P2 tasks (tests modules created earlier)

**Description:**
Achieve 80%+ test coverage for all `utils/training/engine/` modules. Focus on edge cases, error handling, and integration points.

**Acceptance Criteria:**
- [ ] Coverage report: >80% line coverage, >70% branch coverage
- [ ] Tests for all engine modules: checkpoint, loss, loop, trainer, data, metrics
- [ ] Edge cases: OOM, NaN loss, missing labels, corrupted checkpoints
- [ ] Mocked dependencies: model, optimizer, dataloader
- [ ] Fast: entire suite runs in <60 seconds
- [ ] CI integration: run on all PRs, block merge if coverage drops
- [ ] Coverage badge in README.md
- [ ] Documentation: testing guide for contributors
- [ ] Fixtures: reusable test models, datasets, configs
- [ ] Parametrized tests for multiple scenarios

---

### Task P3-2: Integration Tests for End-to-End Training
**Agent:** `task-developer`
**Effort:** 3 days
**Dependencies:** Task P3-1 (unit tests pass)

**Description:**
Create integration tests that validate complete training workflows: checkpoint → resume, export → deploy, drift → retrain.

**Acceptance Criteria:**
- [ ] Test: train 2 epochs → save checkpoint → resume → verify epoch 3 loss matches
- [ ] Test: train → export bundle → load in TorchServe → inference
- [ ] Test: inject drift → trigger retraining → verify new model deployed
- [ ] Test: Lightning integration → gradient accumulation → verify step counts
- [ ] Test: multi-GPU (if available) → verify distributed training correctness
- [ ] Uses real (but tiny) models: 2-layer transformer, <1M parameters
- [ ] Runs in CI: <5 minutes total
- [ ] Skips GPU tests if CUDA unavailable
- [ ] Documentation: integration testing guide
- [ ] Fixtures: reusable workflows for common scenarios

---

### Task P3-3: Type Safety Validation (mypy --strict)
**Agent:** `python-development:python-pro`
**Effort:** 2 days
**Dependencies:** All P0, P1, P2 tasks (validates modules)

**Description:**
Ensure all `utils/training/engine/` modules pass mypy --strict. Add missing type hints, fix Protocol violations, resolve Any types.

**Acceptance Criteria:**
- [ ] CI job: `.github/workflows/mypy-strict-check.yml`
- [ ] Zero mypy errors in strict mode for engine modules
- [ ] Type stubs for third-party libraries (if needed)
- [ ] Documentation: type annotation guidelines
- [ ] Pre-commit hook: run mypy on changed files
- [ ] Exemption list: legacy modules outside engine/ (temporary)
- [ ] Blocks PR merge if new type errors introduced
- [ ] Performance: mypy check <30 seconds
- [ ] Coverage: 100% of public functions have type hints
- [ ] Documentation: common mypy errors and fixes

---

### Task P3-4: Backward Compatibility Facade Migration
**Agent:** `backend-development:backend-architect`
**Effort:** 2 days
**Dependencies:** Task P1-3 (Trainer), Task P2-6 (Builder)

**Description:**
Update `tier3_training_utilities.py` to be thin facade over new engine modules. Maintain 100% API compatibility for existing notebooks.

**Acceptance Criteria:**
- [ ] `test_fine_tuning()` internally delegates to `Trainer` class
- [ ] All existing parameters mapped to `TrainingConfig` or `TrainingConfigBuilder`
- [ ] Deprecation warnings: suggest new API, link to migration guide
- [ ] Integration test: run template.ipynb with facade → verify results identical
- [ ] Performance: <2% overhead from facade layer
- [ ] Documentation: migration guide (old API → new API examples)
- [ ] Planned removal: v4.0 (6 months)
- [ ] Unit tests: verify facade behavior matches original
- [ ] Error messages: suggest new API when facade used
- [ ] Backward compatible: all existing notebooks work unchanged

---

### Task P3-5: Colab Notebook Validation
**Agent:** `task-developer`
**Effort:** 2 days
**Dependencies:** Task P3-4 (facade migration)

**Description:**
Validate all Colab notebooks work with refactored engine. Test in fresh Colab runtime with default Python/CUDA versions.

**Acceptance Criteria:**
- [ ] `template.ipynb` runs end-to-end in Colab without errors
- [ ] `training.ipynb` runs with new engine (not facade)
- [ ] Checkpoint/resume tested: interrupt training, restart runtime, resume
- [ ] Export bundle tested: generate, download, validate locally
- [ ] W&B integration tested: metrics logged correctly
- [ ] 12-hour timeout scenario: save checkpoints every 30 min, resume multiple times
- [ ] Documentation: Colab-specific troubleshooting
- [ ] Test matrix: Python 3.10, 3.11, 3.12
- [ ] Test matrix: PyTorch 2.0, 2.1, 2.2
- [ ] Performance: no regressions vs baseline

---

### Task P3-6: Documentation and Migration Guide
**Agent:** `task-developer`
**Effort:** 3 days
**Dependencies:** All previous tasks (documents final implementation)

**Description:**
Create comprehensive documentation for new engine architecture. Include migration guide, API reference, troubleshooting, and performance tuning.

**Acceptance Criteria:**
- [ ] `docs/TRAINING_ENGINE_ARCHITECTURE.md`: module overview, design decisions
- [ ] `docs/MIGRATION_GUIDE_V4.md`: old API → new API examples, FAQ
- [ ] `docs/API_REFERENCE.md`: auto-generated from docstrings
- [ ] `docs/TROUBLESHOOTING.md`: common errors, solutions, performance tips
- [ ] `docs/PERFORMANCE_TUNING.md`: Flash Attention, compile, mixed precision, profiling
- [ ] Docstrings: all public classes/methods with examples
- [ ] README.md: updated with new architecture overview
- [ ] Video tutorial: 10-minute walkthrough (optional)
- [ ] Code examples: runnable scripts for common workflows
- [ ] Changelog: complete v4.0 release notes

---

### Task P3-7: Performance Benchmarking and Optimization
**Agent:** `machine-learning-ops:ml-engineer`
**Effort:** 3 days
**Dependencies:** Task P3-1 (unit tests ensure correctness)

**Description:**
Benchmark refactored engine against baseline. Identify bottlenecks, optimize hot paths, validate speedup claims.

**Acceptance Criteria:**
- [ ] Benchmark suite: `scripts/benchmark_training.py`
- [ ] Metrics: throughput (samples/sec), memory usage, checkpoint overhead
- [ ] Scenarios: baseline, Flash Attention, torch.compile, mixed precision, combined
- [ ] Report: speedup matrix (architecture × optimization)
- [ ] Validation: Flash Attention 2-4x claim verified on T4/A100
- [ ] Profiling: cProfile + PyTorch Profiler analysis
- [ ] Optimization: eliminate top 3 bottlenecks
- [ ] CI: regression tests for throughput (fail if >5% slowdown)
- [ ] Documentation: performance tuning guide
- [ ] Visualization: speedup charts, memory usage graphs

---

## Phase 5: Security and Quality Gates (Week 9)

### Task P3-8: Security Audit and Secret Scanning
**Agent:** `verify-security`
**Effort:** 2 days
**Dependencies:** Task P2-1 (export bundles), Task P0-1 (checkpoints)

**Description:**
Audit export bundles, checkpoint files, and logs for accidental secret leakage. Enhance pre-commit hook with context-aware scanning.

**Acceptance Criteria:**
- [ ] Audit: scan all generated artifacts for API keys, tokens, credentials
- [ ] Enhanced pre-commit hook: detect secrets in config files, checkpoints, logs
- [ ] Exclusion patterns: exclude .pt, .pth, .ckpt from Git by default
- [ ] Documentation: secure credential management (environment variables, secret managers)
- [ ] CI: automated secret scanning on all PRs (GitHub Secret Scanning API)
- [ ] Incident response: guide for revoking leaked credentials
- [ ] Test: inject fake secret, verify detection
- [ ] Integration: GitHub Advanced Security alerts
- [ ] Documentation: security best practices
- [ ] Compliance: GDPR, SOC 2 considerations

---

### Task P3-9: Final Integration and Smoke Tests
**Agent:** `verify-quality`
**Effort:** 2 days
**Dependencies:** All previous tasks

**Description:**
Run comprehensive smoke tests across all workflows. Validate backward compatibility, performance, security, and documentation accuracy.

**Acceptance Criteria:**
- [ ] Smoke test: train 1 epoch with all features enabled (Flash Attention, compile, checkpointing, export)
- [ ] Backward compatibility: facade API produces identical results to new engine
- [ ] Performance: new engine ≥ baseline throughput (no regressions)
- [ ] Security: no secrets in generated artifacts
- [ ] Documentation: all code examples run without errors
- [ ] CI: green build on main branch
- [ ] Release checklist: version bump, changelog, migration guide
- [ ] Deployment: update production models with new engine
- [ ] Monitoring: track post-release metrics (error rates, throughput)
- [ ] Rollback plan: revert procedure if issues found

---

## Timeline and Dependencies

### Critical Path
```
P0-1 (Checkpoint) → P1-1 (DataLoader) → P1-3 (Trainer) → P3-4 (Facade) → P3-5 (Colab) → P3-9 (Smoke Tests)
                  → P1-2 (Metrics)    ↗
                  → P2-2 (Registry)   ↗

P0-2 (Loss) → P1-4 (Loop) → P1-3 (Trainer)
           → P1-5 (ModelOutput) ↗

P0-3 (GradMon) → P0-4 (AccumFix) → P1-4 (Loop)
```

### Weekly Breakdown
- **Week 1:** P0-1, P0-2, P0-3 (parallel)
- **Week 2:** P0-4, P0-5, P1-1 (parallel)
- **Week 3:** P1-2, P1-3, P1-4 (parallel)
- **Week 4:** P1-5, P1-6, P2-1 (parallel)
- **Week 5:** P2-2, P2-3, P2-4 (parallel)
- **Week 6:** P2-5, P2-6, P3-1 (parallel)
- **Week 7:** P3-2, P3-3, P3-4 (parallel)
- **Week 8:** P3-5, P3-6, P3-7 (parallel)
- **Week 9:** P3-8, P3-9 (sequential)

---

## Agent Workload Distribution

### python-development:python-pro (10 tasks, 21 days)
- P0-1: Checkpoint System (3d)
- P0-2: LossStrategy Protocol (2d)
- P0-3: Gradient Monitoring (1.5d)
- P1-1: DataLoader & Collation (2.5d)
- P1-4: Training Loop (3d)
- P1-5: ModelOutput Dataclass (1.5d)
- P1-6: Registry Pattern (2d)
- P2-6: Builder Pattern (2d)
- P3-3: Type Safety (2d)

### machine-learning-ops:ml-engineer (5 tasks, 15 days)
- P0-4: Gradient Accumulation Fix (2d)
- P1-2: Metrics with Drift (3d)
- P2-1: Complete Export Bundles (4d)
- P2-3: Flash Attention Validation (3d)
- P3-7: Performance Benchmarking (3d)

### machine-learning-ops:mlops-engineer (4 tasks, 9.5 days)
- P0-5: Requirements CI (1d)
- P2-2: Model Registry (3d)
- P2-4: Retraining Triggers (2.5d)
- P2-5: Job Scheduler (3d)

### backend-development:backend-architect (2 tasks, 4 days)
- P1-3: Training Orchestrator (2d)
- P3-4: Facade Migration (2d)

### task-developer (4 tasks, 14 days)
- P3-1: Unit Test Suite (4d)
- P3-2: Integration Tests (3d)
- P3-5: Colab Validation (2d)
- P3-6: Documentation (3d)

### verify-* agents (2 tasks, 4 days)
- P3-8: Security Audit (2d)
- P3-9: Smoke Tests (2d)

---

## Risk Management

### High-Risk Items

**Risk 1: Checkpoint System Complexity**
- **Impact:** High (blocks Colab timeout recovery)
- **Probability:** Medium (complex state management)
- **Mitigation:** Incremental implementation, extensive testing, rollback plan

**Risk 2: Backward Compatibility Breaks**
- **Impact:** High (existing notebooks fail)
- **Probability:** Medium (facade complexity)
- **Mitigation:** Comprehensive integration tests, gradual rollout, deprecation warnings

**Risk 3: Performance Regression**
- **Impact:** Medium (slower training)
- **Probability:** Low (modular design overhead)
- **Mitigation:** Continuous benchmarking, profiling, optimization in P3-7

**Risk 4: Type Safety Overhead**
- **Impact:** Low (development velocity)
- **Probability:** Medium (mypy strict mode)
- **Mitigation:** Incremental adoption, exemption list, documentation

### Contingency Plans

**If Phase 1 delayed:**
- Extend timeline by 1 week, adjust Phase 2 start
- Prioritize P0-1 (checkpoint) over P0-5 (CI)

**If integration tests fail (P3-2):**
- Pause Phase 5, allocate 3 days for debugging
- Rollback breaking changes, re-run Phase 4

**If performance regression detected (P3-7):**
- Allocate 2 additional days for optimization
- Use profiling to identify bottlenecks
- Consider selective rollback of slow modules

---

## Success Criteria

### Functional Requirements
- ✅ Checkpoint/resume works in Colab (12-hour timeout recovery)
- ✅ Training completes with all task types (LM, Classification, Vision)
- ✅ Export bundles deploy successfully to TorchServe
- ✅ Drift detection triggers retraining alerts
- ✅ All existing notebooks work unchanged (backward compatibility)

### Non-Functional Requirements
- ✅ 80%+ test coverage on engine modules
- ✅ Type-safe (mypy --strict passes)
- ✅ Performance: ≥ baseline throughput (no regressions)
- ✅ Documentation: complete migration guide
- ✅ Security: no secrets in generated artifacts

### Release Criteria for v4.0
- [ ] All P0 tasks complete (critical blockers)
- [ ] All P1 tasks complete (core refactoring)
- [ ] All P3 tasks complete (testing & migration)
- [ ] CI green on main branch
- [ ] Smoke tests pass
- [ ] Documentation complete
- [ ] Migration guide published
- [ ] Announcement blog post drafted

---

## Next Steps

1. **Review and Approve Plan** (This document)
2. **Create GitHub Project Board** with all tasks
3. **Kickoff Phase 1** (Week 1):
   - Assign agents to P0-1, P0-2, P0-3
   - Set up daily standup meetings
   - Begin parallel implementation
4. **Weekly Check-ins**:
   - Review progress against timeline
   - Unblock dependencies
   - Adjust priorities as needed
5. **Phase Gates**:
   - Gate 1 (End of Week 2): All P0 tasks complete
   - Gate 2 (End of Week 4): All P1 tasks complete
   - Gate 3 (End of Week 6): All P2 tasks complete
   - Gate 4 (End of Week 8): All P3 tasks complete
   - Gate 5 (End of Week 9): Release v4.0

---

## Appendix A: Expert Agent Reports

### ML Engineer Report (Summary)
- Export bundles incomplete: no model registry, static Dockerfile
- Flash Attention untested on Vision Transformers, Longformer
- Gradient accumulation conflicts with PyTorch Lightning
- MetricsTracker missing drift detection, confidence tracking
- Production checklist: 12 must-have items before deployment

### MLOps Engineer Report (Summary)
- **CRITICAL:** No checkpointing in 1765-line training loop
- Refactor plan missing checkpoint.py specification
- No automated retraining triggers (drift/performance)
- Requirements.txt sync validation missing in CI
- Migration path: 5-phase rollout with backward compatibility

### Python Architect Report (Summary)
- God Function antipattern: 30+ parameters in `test_fine_tuning()`
- Registry Pattern prevents runtime errors (replaces factory function)
- Type safety with Protocol + TypedDict (mypy --strict compliance)
- Gradient health checks: detect NaN/Inf early
- Testing strategy: 80%+ coverage, integration tests, mypy validation

---

## Appendix B: Key Design Decisions

### Decision 1: Protocol vs ABC for LossStrategy
**Choice:** Protocol
**Rationale:** Duck typing, easier mocking, no inheritance required
**Trade-off:** Less explicit than ABC, requires mypy for validation

### Decision 2: Registry vs Factory for Strategy Lookup
**Choice:** Registry Pattern
**Rationale:** Compile-time validation, clear error messages, extensible
**Trade-off:** More boilerplate (decorators), but prevents runtime errors

### Decision 3: Facade vs Immediate Migration
**Choice:** Backward compatibility facade until v4.0
**Rationale:** Zero downtime, gradual user migration, risk mitigation
**Trade-off:** Maintains legacy code for 6 months, deprecation warnings

### Decision 4: mypy --strict vs Gradual Typing
**Choice:** mypy --strict for engine modules only
**Rationale:** Type safety for new code, exempts legacy modules
**Trade-off:** Development velocity slowdown, but prevents bugs

### Decision 5: Checkpoint Format (PyTorch vs Safetensors)
**Choice:** PyTorch native (.pt files) for now
**Rationale:** Mature, well-tested, Colab compatible
**Trade-off:** Consider Safetensors in future for security/speed

---

**Document Status:** Complete and ready for review
**Last Updated:** 2025-01-20
**Next Review:** After Phase 1 completion (Week 2)