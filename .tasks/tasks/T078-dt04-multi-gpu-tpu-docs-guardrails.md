---
id: T078
enhancement_id: DT-04
title: Multi-GPU/TPU Documentation and Configuration Guardrails
status: pending
priority: 3
agent: fullstack
dependencies: [T075, T076, T077]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [distributed, documentation, guardrails, enhancement1.0]

context_refs:
  - context/project.md

docs_refs:
  - docs/USAGE_GUIDE_COLAB_AND_CLI.md
  - docs/ML_ENGINEERING_RISK_ANALYSIS.md

est_tokens: 7000
actual_tokens: null
---

## Description

Document distributed training strategies (auto/ddp/fsdp_native) with hardware setup notes, add runtime validation guardrails to prevent misconfiguration errors, create safe default configs.

## Business Context

**User Story**: As a newcomer to distributed training, I want clear docs explaining when to use DDP vs FSDP, so I don't waste time debugging config errors.

**Why This Matters**: Reduces user frustration; prevents common misconfigurations

**What It Unblocks**: Production distributed training adoption

**Priority Justification**: Priority 3 - Important for usability but not blocking features

## Acceptance Criteria

- [ ] "Distributed Training Guide" section in USAGE_GUIDE_COLAB_AND_CLI.md
- [ ] Explains strategies: "auto" (single-GPU), "ddp" (multi-GPU), "fsdp_native" (large models)
- [ ] Hardware setup notes: local multi-GPU vs Colab single-GPU
- [ ] Safe default configs documented
- [ ] Runtime checks: if strategy="ddp" but devices=1, warn and suggest auto
- [ ] Copy-pasteable example configs for 1-GPU, 2-GPU, 4-GPU setups
- [ ] Troubleshooting section: common errors and fixes
- [ ] Optional: ML_ENGINEERING_RISK_ANALYSIS.md updated with distributed training risks

## Test Scenarios

**Test Case 1: Misconfiguration Warning**
- Given: strategy="ddp", devices=1
- When: TrainingCoordinator instantiated
- Then: Warning logged: "DDP requires multiple devices, falling back to single-GPU training"

**Test Case 2: Documentation Walkthrough**
- Given: User follows distributed training guide
- When: They run example DDP config
- Then: Training succeeds on multi-GPU hardware

**Test Case 3: Safe Default Config**
- Given: User doesn't specify strategy
- When: Config parsed with defaults
- Then: strategy="auto", devices="auto" (works on any hardware)

## Technical Implementation

**Documentation Structure:**

```markdown
## Distributed Training Guide

### Strategies

- **auto**: Single-GPU or CPU (default, safest)
- **ddp**: Data-parallel across multiple GPUs (4-8x speedup)
- **fsdp_native**: Fully-sharded data-parallel (large models >1B params)

### Hardware Requirements

- DDP: 2+ GPUs on same node
- FSDP: A100/H100 GPUs with NVLink (BF16 support)

### Example Configs

**Single-GPU (Colab Free):**
```json
{"strategy": "auto", "devices": 1, "precision": "16-mixed"}
```

**4-GPU Local Workstation:**
```json
{"strategy": "ddp", "devices": 4, "precision": "bf16-mixed"}
```

### Troubleshooting

- **Error: "DDP requires multiple processes"**: Check devices > 1
- **FSDP OOM**: Reduce batch_size, enable activation checkpointing
```

**Runtime Validation:**
```python
def validate_distributed_config(config: TrainingConfig):
    if config.strategy == "ddp" and config.devices <= 1:
        logger.warning("DDP requires multiple devices, using single-GPU instead")
        config.strategy = "auto"
```

## Dependencies

**Hard Dependencies**:
- [T075] TrainingCoordinator
- [T076] Config integration
- [T077] Checkpointing

## Design Decisions

**Decision 1: Document safe defaults, not advanced features**
- **Rationale**: Most users have single-GPU; prioritize common case
- **Trade-offs**: Advanced users must read Lightning docs for edge cases

**Decision 2: Warn on misconfiguration, don't error**
- **Rationale**: Helpful instead of blocking; training still proceeds
- **Trade-offs**: May hide real issues (but logged clearly)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Docs become outdated as Lightning evolves | M - Incorrect guidance | M | Version docs to Lightning 2.4.0; note compatibility range |
| Users attempt FSDP on unsupported GPUs | M - Cryptic errors | M | Add GPU compatibility check; warn if not A100/H100 |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Fourth distributed training task (DT-04 from enhancement1.0.md)
**Dependencies:** T075, T076, T077
**Estimated Complexity:** Simple (documentation + validation logic)

## Completion Checklist

- [ ] Distributed training guide written
- [ ] Example configs created
- [ ] Runtime validation added
- [ ] Troubleshooting section complete
- [ ] All 8 acceptance criteria met
- [ ] All 3 test scenarios validated
- [ ] 2 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** Comprehensive distributed training docs with examples, runtime guardrails prevent common misconfigurations, copy-pasteable configs for 1/2/4 GPU setups.
