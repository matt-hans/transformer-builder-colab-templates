---
id: T076
enhancement_id: DT-02
title: Surface DDP/FSDP Options Through Config and CLI
status: pending
priority: 2
agent: backend
dependencies: [T075]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [distributed, configuration, ddp, fsdp, enhancement1.0]

context_refs:
  - context/project.md

docs_refs:
  - docs/USAGE_GUIDE_COLAB_AND_CLI.md

est_tokens: 9000
actual_tokens: null
---

## Description

Extend `TrainingConfig` with distributed training fields (strategy, devices, num_nodes, accumulate_grad_batches, precision) and wire into `cli/run_training.py`. Creates example DDP config enabling multi-GPU training via JSON configuration.

## Business Context

**User Story**: As a power user with 4 GPUs, I want to set `"strategy": "ddp", "devices": 4` in my config JSON to enable data-parallel training.

**Why This Matters**: Config-driven distributed training; no code changes needed

**What It Unblocks**: DT-03 (checkpointing), production multi-GPU workflows

**Priority Justification**: Priority 2 - Completes distributed training foundation

## Acceptance Criteria

- [ ] TrainingConfig extended with fields: strategy, devices, num_nodes, accumulate_grad_batches, precision
- [ ] `cli/run_training.py` parses distributed config and passes to TrainingCoordinator
- [ ] `configs/example_train_ddp.json` created with DDP settings
- [ ] Command `python -m cli.run_training --config configs/example_train_ddp.json` launches DDP (if multi-GPU available)
- [ ] Safe defaults: strategy="auto", devices="auto" (single GPU on Colab)
- [ ] Type hints for new TrainingConfig fields
- [ ] Documentation section: "Distributed Training" in USAGE_GUIDE
- [ ] Works for both text and vision tasks

## Test Scenarios

**Test Case 1: DDP Config Parsing**
- Given: config with strategy="ddp", devices=2
- When: cli/run_training.py loads config
- Then: TrainingCoordinator instantiated with correct Lightning params

**Test Case 2: Single-GPU Fallback**
- Given: User specifies devices=4 but only 1 GPU available
- When: Training starts
- Then: Warning logged, falls back to devices=1

**Test Case 3: Gradient Accumulation**
- Given: accumulate_grad_batches=4 in config
- When: Training runs
- Then: Effective batch size = batch_size * 4

**Test Case 4: Mixed Precision**
- Given: precision="bf16-mixed"
- When: Training on GPU with bf16 support
- Then: Uses bfloat16 training for 2x speedup

## Technical Implementation

```python
# utils/training/training_config.py
@dataclass
class TrainingConfig:
    # Existing fields
    learning_rate: float = 5e-5
    batch_size: int = 4
    epochs: int = 10

    # NEW: Distributed training fields
    strategy: str | None = "auto"  # "auto", "ddp", "fsdp_native", None
    devices: int | str | list[int] | None = "auto"
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    precision: str = "bf16-mixed"  # "bf16-mixed", "16-mixed", "32"
```

**configs/example_train_ddp.json:**
```json
{
  "task_name": "lm_tiny",
  "learning_rate": 5e-5,
  "batch_size": 4,
  "epochs": 10,
  "strategy": "ddp",
  "devices": 2,
  "precision": "bf16-mixed",
  "accumulate_grad_batches": 2
}
```

## Dependencies

**Hard Dependencies**:
- [T075] TrainingCoordinator - Consumes strategy/devices config

## Design Decisions

**Decision 1: Default strategy="auto" not "ddp"**
- **Rationale**: Auto detects single vs multi-GPU; safer default
- **Trade-offs**: Less explicit, but prevents errors on single-GPU systems

**Decision 2: devices="auto" instead of devices=None**
- **Rationale**: Lightning convention; uses all available GPUs
- **Trade-offs**: May use all GPUs unintentionally (document clearly)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| User sets devices=4 with 1 GPU available | M - Confusing error | M | Add validation; warn and clamp to available devices |
| FSDP not supported on older GPUs | M - Training fails | M | Document FSDP requirements (A100/H100); default to DDP |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Second distributed training task (DT-02 from enhancement1.0.md)
**Dependencies:** T075 (TrainingCoordinator)
**Estimated Complexity:** Standard (config extension + CLI integration)

## Completion Checklist

- [ ] TrainingConfig fields added
- [ ] cli/run_training.py integration
- [ ] example_train_ddp.json created
- [ ] Documentation updated
- [ ] All 8 acceptance criteria met
- [ ] All 4 test scenarios validated
- [ ] 2 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** Distributed training configurable via JSON, CLI launches DDP/FSDP, example config works on multi-GPU hardware.
