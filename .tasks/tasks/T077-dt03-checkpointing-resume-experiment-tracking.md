---
id: T077
enhancement_id: DT-03
title: Robust Checkpointing, Resume Logic, and Experiment Tracking Integration
status: pending
priority: 2
agent: backend
dependencies: [T075, T076]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [distributed, checkpoints, experiment-tracking, enhancement1.0]

context_refs:
  - context/project.md

est_tokens: 12000
actual_tokens: null
---

## Description

Standardize checkpoint directories (`checkpoints/{run_name}/{epoch}`), integrate Lightning callbacks with ExperimentDB for metadata tracking, implement resume-from-checkpoint logic in TrainingCoordinator.

## Business Context

**User Story**: As a researcher, I want training to auto-resume from the last checkpoint if Colab disconnects, so I don't lose 8 hours of training progress.

**Why This Matters**: Session resilience; prevents wasted compute on Colab 12-hour timeout

**What It Unblocks**: Multi-day training runs, production workflows

**Priority Justification**: Priority 2 - Critical for long-running distributed training

## Acceptance Criteria

- [ ] Checkpoint directory structure: `checkpoints/{run_name}/epoch_{N}/`
- [ ] Lightning ModelCheckpoint callback integrated, saves top-k best models
- [ ] ExperimentDB updated with checkpoint paths: `run_id, checkpoint_path, val_metric, epoch`
- [ ] `resume_from_checkpoint` parameter in TrainingConfig
- [ ] TrainingCoordinator.train() resumes from checkpoint if specified
- [ ] Works for DDP/FSDP and vanilla PyTorch paths
- [ ] Metadata persisted: optimizer state, epoch number, RNG state
- [ ] Unit test: train 3 epochs, checkpoint, resume for 2 more, verify continuity

## Test Scenarios

**Test Case 1: Auto-Checkpoint Every Epoch**
- Given: Training for 5 epochs
- When: Each epoch completes
- Then: Checkpoint saved to `checkpoints/run_42/epoch_1/`, ..., epoch_5/

**Test Case 2: Resume from Checkpoint**
- Given: Training interrupted at epoch 3, checkpoint exists
- When: TrainingConfig(resume_from_checkpoint="checkpoints/run_42/epoch_3")
- Then: Training resumes from epoch 4, optimizer state restored

**Test Case 3: ExperimentDB Integration**
- Given: Checkpoint saved at epoch 5 with val_loss=0.38
- When: Checkpoint callback fires
- Then: ExperimentDB.log_artifact(run_id, "checkpoint", path, metadata={"val_loss": 0.38, "epoch": 5})

**Test Case 4: Top-K Best Models**
- Given: ModelCheckpoint with save_top_k=3
- When: Training for 10 epochs
- Then: Only 3 best checkpoints retained (lowest val_loss)

## Technical Implementation

```python
# utils/training/training_core.py
class TrainingCoordinator:
    def train(self, model, datamodule, task_spec, experiment_db, resume_from=None):
        if self.use_lightning:
            # Setup Lightning callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=f"checkpoints/{run_name}",
                filename="epoch_{epoch}",
                save_top_k=3,
                monitor="val_loss",
            )
            self._trainer.callbacks.append(checkpoint_callback)

            if resume_from:
                self._trainer.fit(model, datamodule, ckpt_path=resume_from)
            else:
                self._trainer.fit(model, datamodule)

            # Log best checkpoint to ExperimentDB
            if experiment_db:
                experiment_db.log_artifact(
                    run_id, "checkpoint", checkpoint_callback.best_model_path
                )
```

## Dependencies

**Hard Dependencies**:
- [T075] TrainingCoordinator
- [T076] Config integration

## Design Decisions

**Decision 1: Checkpoint every epoch, not every N steps**
- **Rationale**: Simpler logic; epochs are natural checkpoints
- **Trade-offs**: Less granular resume (may lose 1 epoch progress)

**Decision 2: Save top-k=3 best models, delete rest**
- **Rationale**: Disk space limited on Colab; 3 is sufficient for model selection
- **Trade-offs**: Can't resume from arbitrary epoch

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Checkpoint corrupted (Colab crashes mid-save) | H - Cannot resume | L | Atomic save (write to temp, then rename); validate after load |
| ExperimentDB out of sync with checkpoints | M - Metadata mismatch | M | Transactional updates; rollback on error |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Third distributed training task (DT-03 from enhancement1.0.md)
**Dependencies:** T075 (Coordinator), T076 (Config)
**Estimated Complexity:** Standard (callbacks + resume logic)

## Completion Checklist

- [ ] Checkpoint directory structure implemented
- [ ] Lightning callbacks integrated
- [ ] ExperimentDB logging
- [ ] Resume logic working
- [ ] All 8 acceptance criteria met
- [ ] All 4 test scenarios validated
- [ ] 2 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** Checkpoints saved reliably, resume works for interrupted training, ExperimentDB tracks checkpoint metadata.
