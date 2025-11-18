---
id: T064
title: Add GPU Metrics Tracking (Memory, Utilization, Temperature)
status: pending
priority: 3
agent: backend
dependencies: [T051]
blocked_by: []
created: 2025-11-16T12:00:00Z
updated: 2025-11-16T12:00:00Z
tags: [metrics, gpu, phase5, enhancement, monitoring]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - utils/tier3_training_utilities.py
  - utils/training/metrics_tracker.py

est_tokens: 9000
actual_tokens: null
---

## Description

Add GPU metrics tracking to MetricsTracker: memory usage (allocated/reserved), GPU utilization %, and temperature. Logs to W&B for real-time monitoring of hardware health during training.

**Requires T051's `log_scalar()` method** to log per-batch GPU metrics.

Current state: No GPU monitoring. Users don't know if they're hitting 12GB Colab limit until OOM error crashes training.

Target state: `_log_gpu_metrics(tracker, step)` function samples GPU state every epoch, logs:
- `gpu/memory_allocated_mb`: Currently allocated memory
- `gpu/memory_reserved_mb`: Reserved by PyTorch allocator
- `gpu/utilization_percent`: GPU compute utilization
- `gpu/temperature_celsius`: GPU temperature (if available)

## Business Context

**Why This Matters:** Colab free tier has 12GB GPU limit. Users hitting 11.8GB don't realize they're at risk of OOM. GPU metrics enable proactive optimization (reduce batch size, enable gradient checkpointing).

**Priority:** P3 - Monitoring enhancement. Nice-to-have for production training.

## Acceptance Criteria

- [ ] `_log_gpu_metrics(tracker, step)` function created using `torch.cuda` APIs
- [ ] Logs memory: `torch.cuda.memory_allocated() / 1024**2` (MB)
- [ ] Logs memory: `torch.cuda.memory_reserved() / 1024**2` (MB)
- [ ] Logs utilization via `nvidia-smi` or `pynvml` library (if available)
- [ ] Logs temperature via `nvidia-smi` (if available)
- [ ] Integrated into `_run_training_epoch()` - called once per epoch
- [ ] Validation: W&B dashboard shows GPU metrics plots
- [ ] Validation: Memory usage increases with batch size (sanity check)
- [ ] Gracefully handles CPU-only training (no errors if CUDA unavailable)

## Test Scenarios

**Test Case 1:** GPU memory logging
- Given: Train model on GPU
- When: Check W&B "gpu/memory_allocated_mb" plot
- Then: Shows memory usage increasing during forward pass, decreasing after backward

**Test Case 2:** CPU fallback
- Given: Train on CPU (no CUDA)
- When: GPU metrics logging called
- Then: Skips silently or logs zeros, no errors raised

**Test Case 3:** Memory limit detection
- Given: Training approaches 12GB limit
- When: Monitor GPU metrics
- Then: W&B plot shows memory approaching limit, user can intervene before OOM

## Technical Implementation

```python
import torch
from typing import Optional

def _log_gpu_metrics(tracker: MetricsTracker, step: int) -> None:
    """Log GPU metrics to tracker (requires T051's log_scalar method)."""
    if not torch.cuda.is_available():
        return  # CPU training, skip

    # Memory metrics
    mem_allocated_mb = torch.cuda.memory_allocated() / 1024**2
    mem_reserved_mb = torch.cuda.memory_reserved() / 1024**2

    tracker.log_scalar('gpu/memory_allocated_mb', mem_allocated_mb, step=step)
    tracker.log_scalar('gpu/memory_reserved_mb', mem_reserved_mb, step=step)

    # GPU utilization (optional - requires nvidia-smi or pynvml)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        tracker.log_scalar('gpu/utilization_percent', utilization, step=step)
        tracker.log_scalar('gpu/temperature_celsius', temperature, step=step)
    except (ImportError, Exception):
        pass  # pynvml not available or error accessing GPU, skip gracefully


# Integrate into training loop
def _run_training_epoch(model, dataloader, optimizer, device, tracker=None, epoch=0, ...):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # ... training step ...

        # Log GPU metrics once per epoch (first batch)
        if tracker and batch_idx == 0:
            _log_gpu_metrics(tracker, step=epoch)

    return total_loss / len(dataloader)
```

## Dependencies

**Hard Dependencies:**
- [T051] Add log_scalar() method - **Required** to log GPU metrics

**Soft Dependencies:**
- pynvml library (optional for utilization/temperature) - install via pip

**Blocks:** None

## Design Decisions

**Decision 1:** Log once per epoch (not per batch)
- **Rationale:** GPU metrics change slowly. Per-batch logging excessive.
- **Trade-offs:** Pro: Efficient. Con: Less granular (acceptable)

**Decision 2:** pynvml optional (graceful degradation)
- **Rationale:** Memory metrics work with torch.cuda only. Utilization/temp nice-to-have.
- **Trade-offs:** Pro: Works without pynvml. Con: Missing some metrics (acceptable)

**Decision 3:** Log in MB (not GB or bytes)
- **Rationale:** Colab limit is 12GB = 12,000MB. MB provides good resolution without decimals.
- **Trade-offs:** Pro: Human-readable. Con: Not SI units (acceptable, common in ML)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| pynvml import fails on Colab | Low | Medium | Graceful fallback - memory metrics still work. Document in CLAUDE.md: "Install pynvml for full metrics: !pip install pynvml" |
| Metrics overhead slows training | Low | Very Low | Sampling once per epoch negligible overhead. Profile if concern arises. |
| Multi-GPU not supported | Low | Low | Current code uses device 0. For multi-GPU, iterate over devices and log per-GPU metrics. Add in future if needed. |

## Progress Log

### 2025-11-16T12:00:00Z - Task Created

**Created By:** task-creator agent
**Reason:** Phase 5, Task 1 of 2. GPU monitoring enhancement. Requires T051's log_scalar() method.
**Dependencies:** [T051] log_scalar() method
**Estimated Complexity:** Standard (3 hours - GPU API integration + testing)

## Completion Checklist

- [ ] _log_gpu_metrics() function created
- [ ] Logs memory allocated/reserved via torch.cuda
- [ ] Logs utilization/temperature via pynvml (optional)
- [ ] Integrated into training loop (once per epoch)
- [ ] W&B dashboard shows GPU metrics
- [ ] CPU training works without errors

**Definition of Done:** GPU metrics logged to W&B, memory/utilization visible in dashboard, graceful CPU fallback.
