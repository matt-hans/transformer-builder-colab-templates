---
id: T072
enhancement_id: EX-02
title: Implement Tier 4 Export Validation Tests
status: pending
priority: 2
agent: backend
dependencies: [T071]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [export, tier4, validation, testing, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md

est_tokens: 15000
actual_tokens: null
---

## Description

Create a comprehensive Tier 4 testing suite that validates exported models for shape compatibility, numerical parity with PyTorch, and inference latency. This adds `tier4_export_validation.py` module with shape checks, precision validation (max_abs_diff, relative_error), and latency microbenchmarks for ONNX/TorchScript artifacts.

## Business Context

**User Story**: As a deployment engineer, I want automated tests that verify my exported ONNX model produces identical outputs to PyTorch, so that I can confidently deploy to production.

**Why This Matters**: Prevents silent accuracy regressions in deployed models; catches export bugs before production

**What It Unblocks**: EX-03 (CLI export), EX-04 (serving examples), production deployment workflows

**Priority Justification**: Priority 2 - Critical for deployment confidence but not blocking core features

## Acceptance Criteria

- [ ] `utils/training/tier4_export_validation.py` module created
- [ ] `run_tier4_export_validation(model, task_spec, export_dir)` function validates all exported formats
- [ ] Shape check: exported model input/output shapes match PyTorch model
- [ ] Numerical parity: max_abs_diff < 1e-4 for FP32, < 1e-2 for quantized models (configurable thresholds)
- [ ] Latency benchmark: measures inference time over 100 iterations, reports mean/std
- [ ] Returns structured dict: `{"status": "ok"|"warn"|"fail", "formats": {...}, "parity_report": {...}}`
- [ ] Exposed via `utils/test_functions.py` for backward compatibility
- [ ] Works with both text LM and vision classification models
- [ ] Unit test validates parity calculation with synthetic data
- [ ] Integration test runs Tier 4 on LM stub + SimpleCNN

## Test Scenarios

**Test Case 1: ONNX Numerical Parity - Pass**
- Given: LM exported to ONNX, PyTorch model
- When: Tier 4 parity test with 10 random batches
- Then: max_abs_diff < 1e-4, status="ok"

**Test Case 2: TorchScript Shape Check**
- Given: Vision model with output [B, 10], TorchScript export
- When: Shape validation on batch_size=1 and batch_size=4
- Then: Both match expected [B, 10] shape, status="ok"

**Test Case 3: Latency Microbenchmark**
- Given: SimpleCNN exported to ONNX
- When: Latency test over 100 iterations
- Then: Reports mean latency (e.g., 3.2ms ± 0.5ms)

**Test Case 4: Quantized Model Parity - Relaxed Threshold**
- Given: Dynamically quantized LM (qint8)
- When: Parity test with threshold=1e-2
- Then: max_abs_diff < 1e-2, status="ok" (not "fail")

**Test Case 5: Export Failure Handling**
- Given: ONNX export failed, only TorchScript available
- When: Tier 4 runs
- Then: Tests TorchScript only, logs warning about missing ONNX

**Test Case 6: CLI Integration**
- Given: `python -m cli.run_tiers --config configs/example_tiers_export.json`
- When: Command executes
- Then: Runs Tier 4 validation, prints summary report

## Technical Implementation

```python
# utils/training/tier4_export_validation.py
def run_tier4_export_validation(
    model: nn.Module,
    task_spec: TaskSpec,
    export_dir: Path | str,
    num_samples: int = 10,
    thresholds: dict[str, float] = {"fp32": 1e-4, "quantized": 1e-2},
) -> dict:
    """
    Validate exported models against PyTorch reference.

    Returns:
        {
            "status": "ok" | "warn" | "fail",
            "formats": {
                "onnx": {"parity": "ok", "max_abs_diff": 1.2e-5, "latency_ms": 3.4},
                "torchscript": {"parity": "ok", ...}
            }
        }
    """
    export_dir = Path(export_dir)
    results = {"status": "ok", "formats": {}}

    # Generate test inputs
    dummy_input = generate_dummy_input(task_spec)

    # Test each format
    for fmt in ["onnx", "torchscript"]:
        model_path = export_dir / f"model.{fmt}"
        if not model_path.exists():
            continue

        # Load exported model
        exported_model = load_exported_model(model_path, fmt)

        # Numerical parity
        parity = check_numerical_parity(
            model, exported_model, dummy_input, num_samples, thresholds
        )

        # Latency benchmark
        latency = measure_latency(exported_model, dummy_input, n_iters=100)

        results["formats"][fmt] = {
            "parity": parity["status"],
            "max_abs_diff": parity["max_abs_diff"],
            "latency_ms": latency["mean_ms"],
        }

    return results
```

## Dependencies

**Hard Dependencies**:
- [T071] Harden export_utilities APIs - Provides export_model function to create artifacts

## Design Decisions

**Decision 1: Separate thresholds for FP32 vs quantized**
- **Rationale**: Quantization introduces expected precision loss; fail threshold must be relaxed
- **Trade-offs**: More configuration, but prevents false positives

**Decision 2: Measure latency over 100 iterations**
- **Rationale**: Amortizes warmup overhead, captures variance
- **Trade-offs**: Longer test time (~10 seconds), but more reliable measurements

**Decision 3: Skip missing formats gracefully**
- **Rationale**: Users may export to subset of formats (e.g., only ONNX)
- **Trade-offs**: Can't validate all formats, but doesn't block testing

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| ONNX Runtime not installed | H - Cannot test ONNX | M | Lazy import with graceful skip; log warning to install onnxruntime |
| Parity fails due to non-determinism | M - Flaky tests | M | Use fixed random seed for dummy inputs; average over multiple samples |
| Latency measurements noisy on Colab | M - Unreliable benchmarks | H | Report std dev; document that latency is informational, not strict gate |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Second export tier task (EX-02 from enhancement1.0.md)
**Dependencies:** T071 (export API)
**Estimated Complexity:** Complex (numerical validation + benchmarking + multi-format support)

## Completion Checklist

- [ ] tier4_export_validation.py module created
- [ ] Shape, parity, latency checks implemented
- [ ] Unit tests validate parity calculation
- [ ] Integration test with LM + vision models
- [ ] Exposed via test_functions.py
- [ ] All 10 acceptance criteria met
- [ ] All 6 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 3 risks mitigated

**Definition of Done:** Tier 4 tests validate exported models for parity and latency, support ONNX/TorchScript, handle missing formats gracefully, and integrate into CLI.
