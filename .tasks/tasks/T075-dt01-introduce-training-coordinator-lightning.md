---
id: T075
enhancement_id: DT-01
title: Introduce TrainingCoordinator with PyTorch Lightning Integration
status: pending
priority: 2
agent: backend
dependencies: []
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [distributed, training, lightning, architecture, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - docs/ARCHITECTURE_OVERVIEW_v4.0.0.md

est_tokens: 16000
actual_tokens: null
---

## Description

Refactor existing training loop into a `TrainingCoordinator` class that wraps PyTorch Lightning's Trainer for distributed training (DDP/FSDP) support. Maintains backward compatibility with simple CPU/single-GPU workflows while enabling multi-GPU scaling.

Coordinator abstracts Lightning complexity, providing simple `.train(model, datamodule, task_spec)` API. Falls back to vanilla PyTorch loop if Lightning not available.

## Business Context

**User Story**: As a researcher with multi-GPU hardware, I want to enable DDP training with one config change, so I can scale experiments without rewriting training code.

**Why This Matters**: Enables large-scale training; reduces time-to-results by 4-8x on multi-GPU

**What It Unblocks**: DT-02 (DDP/FSDP configs), DT-03 (checkpointing), DT-04 (docs)

**Priority Justification**: Priority 2 - High value for power users but not blocking core features

## Acceptance Criteria

- [ ] `utils/training/training_core.py` created with `TrainingCoordinator` class
- [ ] Constructor accepts strategy, precision, devices, num_nodes parameters
- [ ] `.train(model, datamodule, task_spec, experiment_db)` method executes training
- [ ] Falls back to vanilla PyTorch loop if Lightning not installed (graceful degradation)
- [ ] Existing single-GPU training continues to work via coordinator with default args
- [ ] Unit test: simple LM training for 1 epoch using coordinator
- [ ] Type hints for all coordinator methods
- [ ] Docstrings explain Lightning parameter mapping
- [ ] No breaking changes to existing training.ipynb cells

## Test Scenarios

**Test Case 1: Single-GPU Training via Coordinator**
- Given: TrainingCoordinator(strategy="auto", devices=1)
- When: coordinator.train(model, dataloader, task_spec)
- Then: Training completes on single GPU, identical to pre-coordinator behavior

**Test Case 2: Fallback to Vanilla PyTorch**
- Given: Lightning not installed
- When: TrainingCoordinator instantiated
- Then: Uses vanilla PyTorch loop, logs warning about Lightning

**Test Case 3: Constructor Parameter Validation**
- Given: TrainingCoordinator(strategy="ddp", devices="auto")
- When: Trainer instantiated
- Then: Lightning Trainer created with strategy="ddp"

**Test Case 4: Backward Compatibility**
- Given: Existing training loop in tier3_training_utilities.py
- When: Refactored to use coordinator
- Then: Same results, no regressions

## Technical Implementation

```python
# utils/training/training_core.py
class TrainingCoordinator:
    """Unified training interface supporting Lightning DDP/FSDP and vanilla PyTorch."""

    def __init__(
        self,
        strategy: str = "auto",
        precision: str = "bf16-mixed",
        devices: int | str | list[int] | None = None,
        num_nodes: int = 1,
        enable_checkpointing: bool = True,
        log_every_n_steps: int = 10,
    ):
        """
        Args:
            strategy: "auto", "ddp", "fsdp_native", or None (vanilla PyTorch)
            precision: "bf16-mixed", "16-mixed", or "32"
            devices: Number of devices or list of device IDs
            num_nodes: Number of nodes for multi-node training
        """
        self.use_lightning = strategy in ["auto", "ddp", "fsdp_native", "ddp_spawn"]

        if self.use_lightning:
            try:
                import pytorch_lightning as pl
                self._trainer = pl.Trainer(
                    strategy=strategy,
                    precision=precision,
                    devices=devices,
                    num_nodes=num_nodes,
                    enable_checkpointing=enable_checkpointing,
                    log_every_n_steps=log_every_n_steps,
                )
            except ImportError:
                logger.warning("PyTorch Lightning not installed, falling back to vanilla PyTorch")
                self.use_lightning = False

    def train(
        self,
        model: nn.Module,
        datamodule_or_loaders: Any,
        task_spec: TaskSpec,
        experiment_db: Optional[ExperimentDB] = None,
        n_epochs: int = 10,
    ) -> dict:
        """Execute training using Lightning or vanilla PyTorch."""
        if self.use_lightning:
            return self._train_lightning(model, datamodule_or_loaders, n_epochs)
        else:
            return self._train_vanilla(model, datamodule_or_loaders, n_epochs)

    def _train_lightning(self, model, datamodule, n_epochs):
        """Lightning training path (DDP/FSDP)."""
        self._trainer.fit(model, datamodule, max_epochs=n_epochs)
        return {"status": "completed", "epochs": n_epochs}

    def _train_vanilla(self, model, dataloader, n_epochs):
        """Vanilla PyTorch training path (single-GPU/CPU)."""
        # Existing training loop from tier3_training_utilities
        for epoch in range(n_epochs):
            for batch in dataloader:
                # Forward, backward, optimizer step
                ...
        return {"status": "completed", "epochs": n_epochs}
```

## Dependencies

**Hard Dependencies**: None - Can start immediately

**External Dependencies**:
- pytorch-lightning >= 2.4.0 (optional, lazy import)

## Design Decisions

**Decision 1: Coordinator wraps Lightning, not inherits**
- **Rationale**: Composition over inheritance; easier to fall back to vanilla
- **Trade-offs**: More boilerplate, but clearer separation of concerns

**Decision 2: Fallback to vanilla PyTorch if Lightning missing**
- **Rationale**: Don't force Lightning dependency for simple use cases
- **Trade-offs**: Two code paths to maintain, but better UX

**Decision 3: Strategy parameter instead of boolean flags**
- **Rationale**: Lightning uses strategy strings; direct mapping
- **Trade-offs**: Less intuitive than `use_ddp=True`, but consistent with Lightning API

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Lightning version incompatibility | H - Breaks distributed training | M | Pin Lightning >= 2.4.0 in requirements-training.txt; test with CI |
| Vanilla fallback diverges from Lightning | M - Inconsistent behavior | M | Comprehensive integration tests for both paths |
| Coordinator adds complexity for simple use cases | M - Steeper learning curve | L | Provide simple defaults; document with examples |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** First distributed training task (DT-01 from enhancement1.0.md)
**Dependencies:** None - foundation task
**Estimated Complexity:** Complex (refactoring existing training loop + Lightning integration)

## Completion Checklist

- [ ] TrainingCoordinator class created
- [ ] Constructor with Lightning parameters
- [ ] .train() method with Lightning and vanilla paths
- [ ] Fallback logic implemented
- [ ] Unit test with simple LM training
- [ ] Backward compatibility verified
- [ ] All 9 acceptance criteria met
- [ ] All 4 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 3 risks mitigated

**Definition of Done:** TrainingCoordinator refactors existing training loop, supports Lightning DDP/FSDP, falls back to vanilla PyTorch gracefully, maintains backward compatibility.
