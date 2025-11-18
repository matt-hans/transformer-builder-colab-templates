---
id: T082
enhancement_id: MO-04
title: Tier 5 Monitoring & Drift Entrypoint
status: pending
priority: 2
agent: fullstack
dependencies: [T079, T080, T081]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [monitoring, tier5, cli, integration, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - docs/USAGE_GUIDE_COLAB_AND_CLI.md

est_tokens: 9000
actual_tokens: null
---

## Description

Create Tier 5 monitoring entrypoint (`tier5_monitoring.py`) that runs evaluation, baseline comparison (if specified), and drift detection in one command. Integrates MO-01, MO-02, MO-03 into unified monitoring workflow accessible via CLI.

## Business Context

**User Story**: As a production engineer, I want to run `python -m cli.run_tiers --config monitoring.json` to validate a new model against baseline and check for data drift.

**Why This Matters**: One-command production validation; streamlines deployment checks

**What It Unblocks**: Production CI/CD pipelines, automated monitoring dashboards

**Priority Justification**: Priority 2 - Completes monitoring tier infrastructure

## Acceptance Criteria

- [ ] `utils/training/tier5_monitoring.py` created with `run_tier5_monitoring()` function
- [ ] Function evaluates current model, optionally compares to baseline (if baseline_run_id provided)
- [ ] Optionally computes drift profile on eval set (if reference profile exists)
- [ ] Returns structured dict: `{"eval_metrics": {...}, "comparison": {...}, "drift": {...}, "status": "ok"|"warn"|"fail"}`
- [ ] `cli/run_tiers.py` supports `tier="5"` mode
- [ ] `configs/example_tiers_monitoring.json` created
- [ ] Command `python -m cli.run_tiers --config configs/example_tiers_monitoring.json` runs Tier 5
- [ ] Documentation: "Tier 5 Monitoring & Drift" section in USAGE_GUIDE
- [ ] Works for text and vision modalities

## Test Scenarios

**Test Case 1: Tier 5 with Baseline Comparison**
- Given: Config with baseline_run_id=1, candidate model loaded
- When: `run_tier5_monitoring(model, adapter, task_spec, eval_cfg, db, baseline_run_id=1)`
- Then: Evaluates candidate, compares to baseline, returns comparison report

**Test Case 2: Tier 5 with Drift Detection**
- Given: Reference profile stored in ExperimentDB for run_id=1
- When: Tier 5 runs with drift detection enabled
- Then: Computes new profile, compares to reference, logs drift scores

**Test Case 3: Tier 5 Eval Only (No Baseline)**
- Given: No baseline_run_id specified
- When: Tier 5 runs
- Then: Evaluates model, logs metrics, skips comparison

**Test Case 4: CLI Integration**
- Given: `configs/example_tiers_monitoring.json` with baseline_run_id=1
- When: `python -m cli.run_tiers --config configs/example_tiers_monitoring.json`
- Then: Runs Tier 5, prints summary: "Candidate improved accuracy by +3%, no severe drift detected"

**Test Case 5: Combined Report**
- Given: Tier 5 runs with eval + comparison + drift
- When: All checks complete
- Then: Returns JSON: `{"eval_metrics": {"accuracy": 0.76}, "comparison": {"delta": +0.04, "status": "improved"}, "drift": {"status": "ok"}}`

**Test Case 6: Failure Classification**
- Given: Comparison shows regression (delta < -0.05), drift status="alert"
- When: Tier 5 runs
- Then: Overall status="fail", detailed report explains why

## Technical Implementation

```python
# utils/training/tier5_monitoring.py
def run_tier5_monitoring(
    model: nn.Module,
    adapter: ModelAdapter,
    task_spec: TaskSpec,
    eval_cfg: EvalConfig,
    db: ExperimentDB,
    baseline_run_id: int | None = None,
    reference_profile_id: int | None = None,
) -> dict:
    """
    Run Tier 5 monitoring: eval + comparison + drift.

    Returns:
        {
            "eval_metrics": {...},
            "comparison": {...} | None,
            "drift": {...} | None,
            "status": "ok" | "warn" | "fail"
        }
    """
    # 1) Evaluate current model
    eval_metrics = run_eval(model, adapter, eval_cfg, task_spec)

    # Log to ExperimentDB
    run_id = db.register_run({"run_name": "tier5_validation", "task_name": task_spec.task_name})
    db.log_metrics(run_id, eval_metrics, split="val")

    result = {"eval_metrics": eval_metrics, "run_id": run_id}

    # 2) Optional: Baseline comparison
    if baseline_run_id is not None:
        baseline_model = load_checkpoint_from_run(baseline_run_id, db)
        comparison = compare_models(baseline_model, model, adapter, task_spec, eval_cfg, db)
        result["comparison"] = comparison
    else:
        result["comparison"] = None

    # 3) Optional: Drift detection
    if reference_profile_id is not None:
        reference_profile = db.get_artifact(reference_profile_id, "profile")
        eval_dataset = build_dataloader(task_spec, eval_cfg.task_name, batch_size=32)
        new_profile = compute_dataset_profile(eval_dataset, task_spec, sample_size=1000)
        drift_analysis = compare_profiles(reference_profile, new_profile)
        result["drift"] = drift_analysis
    else:
        result["drift"] = None

    # 4) Classify overall status
    status = "ok"
    if result.get("comparison") and result["comparison"]["metrics"].get("accuracy", {}).get("status") == "regressed":
        status = "fail"
    if result.get("drift") and result["drift"]["status"] == "alert":
        status = "fail" if status == "fail" else "warn"

    result["status"] = status
    return result
```

**CLI Integration:**

```python
# cli/run_tiers.py
def main(config_path: str):
    config = load_config(config_path)

    if config.get("tier") == "5":
        model, task_spec, adapter = build_model_from_config(config)
        eval_cfg = build_eval_config(config)
        db = ExperimentDB()

        tier5_results = run_tier5_monitoring(
            model, adapter, task_spec, eval_cfg, db,
            baseline_run_id=config.get("baseline_run_id"),
            reference_profile_id=config.get("reference_profile_id")
        )

        print_tier5_summary(tier5_results)
```

**configs/example_tiers_monitoring.json:**

```json
{
  "task_name": "lm_tiny",
  "modality": "text",
  "tier": "5",
  "baseline_run_id": 1,
  "reference_profile_id": null,
  "eval": {
    "batch_size": 8
  }
}
```

## Dependencies

**Hard Dependencies**:
- [T079] ExperimentDB Schema - Stores run metadata
- [T080] Regression Testing - compare_models function
- [T081] Drift Metrics - compute_dataset_profile, compare_profiles

## Design Decisions

**Decision 1: Optional baseline and drift (not mandatory)**
- **Rationale**: Users may only want eval metrics (simple validation)
- **Trade-offs**: More conditional logic, but flexible

**Decision 2: Unified status classification (ok/warn/fail)**
- **Rationale**: Simple CI/CD integration (check status for pass/fail gates)
- **Trade-offs**: Loses granularity, but clear for automation

**Decision 3: Store profiles in ExperimentDB artifacts**
- **Rationale**: Centralized storage, queryable by run_id
- **Trade-offs**: Profiles can be large JSON blobs, but manageable for 1k samples

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Tier 5 too slow for CI/CD (eval + comparison + drift) | M - Timeout in CI | M | Make drift optional; use small eval sets; parallelize where possible |
| Status classification too simplistic | M - Misses nuanced issues | M | Log detailed report even if status is binary; users can inspect details |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Fourth monitoring tier task (MO-04 from enhancement1.0.md)
**Dependencies:** T079 (ExperimentDB), T080 (regression), T081 (drift)
**Estimated Complexity:** Standard (integration of existing components + CLI wiring)

## Completion Checklist

- [ ] tier5_monitoring.py module created
- [ ] run_tier5_monitoring function implemented
- [ ] CLI integration in run_tiers.py
- [ ] example_tiers_monitoring.json config created
- [ ] Documentation section written
- [ ] All 9 acceptance criteria met
- [ ] All 6 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** Tier 5 monitoring runs eval + comparison + drift in one command, CLI integration works, config documented, status classification functional.
