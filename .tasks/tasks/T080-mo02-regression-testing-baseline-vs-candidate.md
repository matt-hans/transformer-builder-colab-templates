---
id: T080
enhancement_id: MO-02
title: Regression Testing Utility - Baseline vs Candidate Model Comparison
status: pending
priority: 2
agent: backend
dependencies: [T066, T069, T079]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [monitoring, tier5, regression-testing, mlops, enhancement1.0]

context_refs:
  - context/project.md

est_tokens: 11000
actual_tokens: null
---

## Description

Create `regression_testing.py` module with `compare_models()` function that evaluates baseline and candidate models on held-out eval set, computes metric deltas, and logs comparison to ExperimentDB. Enables automated regression detection in CI/CD pipelines.

## Business Context

**User Story**: As an ML engineer, I want automated comparison reports when deploying a new model, so I can catch accuracy regressions before production.

**Why This Matters**: Prevents silent degradations; quantifies improvement claims

**What It Unblocks**: MO-04 (Tier 5 monitoring), CI/CD model validation

**Priority Justification**: Priority 2 - Critical for production safety

## Acceptance Criteria

- [ ] `utils/training/regression_testing.py` created with `compare_models()` function
- [ ] Function signature: `compare_models(baseline_model, candidate_model, adapter, task_spec, eval_cfg, db)`
- [ ] Evaluates both models on same eval set, computes metric deltas
- [ ] Returns dict: `{"metric": "accuracy", "baseline": 0.72, "candidate": 0.76, "delta": +0.04, "status": "improved"}`
- [ ] Status classification: "improved" (delta > threshold), "neutral", "regressed" (delta < -threshold)
- [ ] Logs comparison to ExperimentDB.comparisons table
- [ ] CLI script: `scripts/run_regression_test.py --baseline-run-id 1 --candidate-run-id 2`
- [ ] Works for text and vision modalities
- [ ] Unit test with dummy models validates delta calculation

## Test Scenarios

**Test Case 1: Improved Model**
- Given: Baseline accuracy=0.70, candidate accuracy=0.75
- When: compare_models runs
- Then: Returns delta=+0.05, status="improved"

**Test Case 2: Regressed Model**
- Given: Baseline accuracy=0.80, candidate accuracy=0.72
- When: Comparison runs
- Then: Returns delta=-0.08, status="regressed", warning logged

**Test Case 3: Neutral Change**
- Given: Baseline loss=0.50, candidate loss=0.51 (within threshold=0.02)
- When: Comparison with threshold=0.02
- Then: Returns status="neutral"

**Test Case 4: Multi-Metric Comparison**
- Given: Compare on accuracy, loss, and perplexity
- When: compare_models with multiple metrics
- Then: Returns dict with delta for each metric

**Test Case 5: ExperimentDB Logging**
- Given: Comparison between run_id=1 and run_id=2
- When: Comparison completes
- Then: Comparison row inserted with baseline_run_id=1, candidate_run_id=2, notes="accuracy: +0.04"

**Test Case 6: CLI Integration**
- Given: `python scripts/run_regression_test.py --baseline-run-id 1 --candidate-run-id 2 --dataset lm_tiny`
- When: Script executes
- Then: Loads checkpoints, runs comparison, prints JSON report

## Technical Implementation

```python
# utils/training/regression_testing.py
def compare_models(
    baseline_model: nn.Module,
    candidate_model: nn.Module,
    adapter: ModelAdapter,
    task_spec: TaskSpec,
    eval_cfg: EvalConfig,
    db: ExperimentDB | None = None,
    comparison_name: str | None = None,
    threshold: float = 0.01,
) -> dict:
    """
    Compare baseline and candidate models on held-out eval set.

    Returns:
        {
            "metrics": {
                "accuracy": {"baseline": 0.72, "candidate": 0.76, "delta": 0.04, "status": "improved"},
                "loss": ...
            },
            "comparison_id": 123  # if db provided
        }
    """
    # Evaluate baseline
    baseline_metrics = run_eval(baseline_model, adapter, eval_cfg, task_spec)

    # Evaluate candidate
    candidate_metrics = run_eval(candidate_model, adapter, eval_cfg, task_spec)

    # Compute deltas
    comparison = {"metrics": {}}
    for metric_name in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric_name]
        candidate_val = candidate_metrics[metric_name]
        delta = candidate_val - baseline_val

        # Classify status
        if abs(delta) < threshold:
            status = "neutral"
        elif delta > 0:
            status = "improved" if metric_name != "loss" else "regressed"  # Lower loss is better
        else:
            status = "regressed" if metric_name != "loss" else "improved"

        comparison["metrics"][metric_name] = {
            "baseline": baseline_val,
            "candidate": candidate_val,
            "delta": delta,
            "status": status,
        }

    # Log to ExperimentDB
    if db:
        # Assume baseline/candidate have run_ids stored
        comparison_id = db.create_comparison(
            baseline_run_id=baseline_model.run_id,
            candidate_run_id=candidate_model.run_id,
            notes=comparison_name or f"Automated comparison: {datetime.now()}"
        )
        comparison["comparison_id"] = comparison_id

    return comparison
```

## Dependencies

**Hard Dependencies**:
- [T066] TaskSpec extension - Modality support
- [T069] Vision eval - run_eval works for vision
- [T079] ExperimentDB schema - Comparisons table

## Design Decisions

**Decision 1: Threshold-based status classification**
- **Rationale**: Small deltas (<1%) may be noise; avoid false alarms
- **Trade-offs**: Configurable threshold, but users must choose wisely

**Decision 2: Inverted logic for loss metrics**
- **Rationale**: Lower loss is better; delta < 0 is improvement
- **Trade-offs**: Adds complexity, but semantically correct

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Eval set drift over time | M - Comparisons misleading | M | Document recommended practice: freeze eval set, version it |
| Threshold too strict (false positives) | M - Alert fatigue | M | Default to 1%; document tuning guidance |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Second monitoring tier task (MO-02 from enhancement1.0.md)
**Dependencies:** T066 (TaskSpec), T069 (eval), T079 (ExperimentDB)
**Estimated Complexity:** Standard (comparison logic + CLI script)

## Completion Checklist

- [ ] regression_testing.py module created
- [ ] compare_models function implemented
- [ ] Status classification logic
- [ ] CLI script created
- [ ] All 9 acceptance criteria met
- [ ] All 6 test scenarios validated
- [ ] 2 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** compare_models compares baseline/candidate correctly, status classification works, ExperimentDB logs comparisons, CLI script functional.
