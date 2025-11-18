---
id: T073
enhancement_id: EX-03
title: Wire Export Tier into CLI and Config System
status: pending
priority: 2
agent: fullstack
dependencies: [T071, T072]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [export, tier4, cli, configuration, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - docs/USAGE_GUIDE_COLAB_AND_CLI.md

est_tokens: 8000
actual_tokens: null
---

## Description

Integrate Tier 4 export validation into `cli/run_tiers.py` with config-driven export settings. Adds `mode="EXPORT"` or `tier="4"` support, creates example configs (`configs/example_tiers_export.json`), and documents export workflow for users.

## Business Context

**User Story**: As a developer, I want to run `python -m cli.run_tiers --config configs/export.json` to export and validate my model in one command.

**Why This Matters**: One-command export + validation workflow; reduces deployment friction

**What It Unblocks**: EX-04 (serving examples), production deployment pipelines

**Priority Justification**: Priority 2 - Completes export tier but not blocking other features

## Acceptance Criteria

- [ ] `cli/run_tiers.py` supports `tier="4"` or `mode="EXPORT"` in config
- [ ] `configs/example_tiers_export.json` created with export settings (formats, quantization)
- [ ] Command `python -m cli.run_tiers --config configs/example_tiers_export.json` runs export + Tier 4 validation
- [ ] Config schema documented with all export options
- [ ] Works for both LM and vision models (text/vision modalities)
- [ ] Prints human-readable summary: export paths, parity status, latency
- [ ] Returns JSON output for CI/CD integration (optional --json flag)
- [ ] Documentation updated in `docs/USAGE_GUIDE_COLAB_AND_CLI.md` with export guide
- [ ] No regressions in Tier 1/2/3 CLI modes

## Test Scenarios

**Test Case 1: CLI Export Command**
- Given: `configs/example_tiers_export.json` with formats=["onnx", "torchscript"]
- When: `python -m cli.run_tiers --config configs/example_tiers_export.json`
- Then: Exports model, runs Tier 4, prints summary

**Test Case 2: Vision Model Export via CLI**
- Given: Config with task_name="vision_tiny", modality="vision"
- When: CLI export runs
- Then: Vision model exported to ONNX, parity validated

**Test Case 3: Config Schema Validation**
- Given: Malformed config (e.g., formats="onnx" instead of ["onnx"])
- When: CLI parses config
- Then: Clear error message explaining correct schema

**Test Case 4: JSON Output Mode**
- Given: `python -m cli.run_tiers --config export.json --json`
- When: Command completes
- Then: Prints JSON to stdout (parseable by jq or Python scripts)

**Test Case 5: Quantization Config**
- Given: Config with quantization="dynamic"
- When: CLI export runs
- Then: Exports quantized TorchScript, validates with relaxed threshold

**Test Case 6: Documentation Walkthrough**
- Given: User follows USAGE_GUIDE export section
- When: They copy-paste example commands
- Then: Successfully exports and validates model

## Technical Implementation

**Required Components:**

1. **`configs/example_tiers_export.json`**
   ```json
   {
     "task_name": "lm_tiny",
     "modality": "text",
     "tier": "4",
     "export": {
       "formats": ["torchscript", "onnx"],
       "quantization": null,
       "export_dir": "exports/lm_tiny"
     }
   }
   ```

2. **Extend `cli/run_tiers.py`**
   ```python
   def main(config_path: str, json_output: bool = False):
       config = load_config(config_path)

       if config.get("tier") == "4" or config.get("mode") == "EXPORT":
           # Build model + task_spec
           model, task_spec, adapter = build_model_from_config(config)

           # Export
           export_config = config.get("export", {})
           exports = export_model(
               model, adapter, task_spec,
               export_dir=export_config.get("export_dir", "exports/"),
               formats=export_config.get("formats", ["torchscript"]),
               quantization=export_config.get("quantization")
           )

           # Validate
           tier4_results = run_tier4_export_validation(
               model, task_spec, export_config["export_dir"]
           )

           # Output
           if json_output:
               print(json.dumps({**exports, **tier4_results}, indent=2))
           else:
               print_export_summary(exports, tier4_results)
   ```

3. **Documentation in `docs/USAGE_GUIDE_COLAB_AND_CLI.md`**

**Validation Commands:**

```bash
# Test CLI export
python -m cli.run_tiers --config configs/example_tiers_export.json

# Expected output:
# ✓ Model exported to exports/lm_tiny/
#   - model.torchscript (2.3 MB)
#   - model.onnx (2.1 MB)
#   - metadata.json
# ✓ Tier 4 Validation:
#   - ONNX: parity OK (max_diff=1.2e-5), latency=3.4ms
#   - TorchScript: parity OK (max_diff=8.3e-6), latency=2.1ms
```

## Dependencies

**Hard Dependencies**:
- [T071] Harden export_utilities APIs
- [T072] Implement Tier 4 Export Validation

## Design Decisions

**Decision 1: Use tier="4" instead of new CLI flag**
- **Rationale**: Consistent with tier="1" pattern; config-driven
- **Trade-offs**: Requires JSON config (not one-liner), but more expressive

**Decision 2: Optional --json flag for CI/CD**
- **Rationale**: Machine-readable output for automation
- **Trade-offs**: Extra flag, but enables scripting

**Decision 3: Export dir in config, not CLI arg**
- **Rationale**: Different runs may export to different dirs (versioned exports)
- **Trade-offs**: Less convenient for quick tests, but better for production

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Config schema ambiguous | M - User confusion | M | Provide annotated example config; validate schema on load |
| Export fails silently in CLI | H - No error surfaced | L | Add comprehensive error handling; log export steps |
| JSON output breaks existing tools | M - CI/CD failures | L | Only activate with --json flag; default is human-readable |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Third export tier task (EX-03 from enhancement1.0.md)
**Dependencies:** T071 (export API), T072 (Tier 4 tests)
**Estimated Complexity:** Simple (CLI integration + config)

## Completion Checklist

- [ ] cli/run_tiers.py supports tier="4" mode
- [ ] configs/example_tiers_export.json created
- [ ] CLI command exports + validates model
- [ ] JSON output mode implemented
- [ ] Documentation updated
- [ ] All 9 acceptance criteria met
- [ ] All 6 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 3 risks mitigated

**Definition of Done:** CLI export mode works for text/vision models, config schema documented, human and JSON output modes functional.
