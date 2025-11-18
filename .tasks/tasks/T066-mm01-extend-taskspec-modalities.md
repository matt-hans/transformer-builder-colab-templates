---
id: T066
enhancement_id: MM-01
title: Extend TaskSpec to Support Modalities (Vision, Audio, Tabular)
status: pending
priority: 1
agent: fullstack
dependencies: []
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [multimodal, architecture, foundation, enhancement1.0, critical-path]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/DEVELOPER_GUIDE_TASKS_EVAL.md

est_tokens: 12000
actual_tokens: null
---

## Description

Extend the `TaskSpec` class to support multiple modalities beyond text (vision, audio, tabular), enabling the training infrastructure to handle diverse input types. This task establishes the foundational abstraction for the entire multimodal platform by adding modality-aware fields, input/output schemas, and canonical task types.

The `TaskSpec` will become the single source of truth for task semantics across all modalities, defining input shapes, preprocessing requirements, and output structures in a unified interface. This enables downstream components (adapters, dataset loaders, evaluation) to operate on heterogeneous tasks without modality-specific branching.

**Technical Approach**: Add new fields (`modality`, `input_schema`, `output_schema`, `preprocessing_config`) with backward-compatible defaults, introduce a `task_type` enum covering text and vision use cases, and update factory helpers to populate these fields for existing text tasks.

**Integration Points**:
- `utils/training/task_spec.py` (primary changes)
- `utils/adapters/model_adapter.py` (will consume new fields in MM-02)
- `utils/training/dataset_utilities.py` (will use schemas in MM-03)
- `utils/training/eval_runner.py` (will route metrics by modality in MM-04)

## Business Context

**User Story**: As an ML engineer, I want to define vision classification tasks using the same `TaskSpec` interface as text tasks, so that I can leverage existing training infrastructure without modality-specific code paths.

**Why This Matters**:
- **Enables multimodal platform**: Unlocks vision, audio, and tabular tasks across the entire stack (Tier 1-5, training, export, monitoring)
- **Reduces complexity**: Single abstraction replaces multiple modality-specific implementations
- **Future-proofs architecture**: Easy to add new modalities (audio, multimodal) without refactoring

**What It Unblocks**:
- MM-02: VisionClassificationAdapter (needs input/output schemas)
- MM-03: Image dataset loaders (needs input_schema for transforms)
- MM-04: Vision evaluation (needs modality routing)
- EX-01 to EX-04: Export tier (needs task_type for dummy input generation)
- DT-01 to DT-04: Distributed training (modality-agnostic coordinator)
- MO-01 to MO-04: Monitoring (needs modality for drift detection)

**Priority Justification**: Priority 1 - Foundation task that blocks all multimodal work. Must complete before any vision-specific features.

## Acceptance Criteria

- [ ] `TaskSpec` has `modality` field: `Literal["text", "vision", "audio", "tabular"]` with default `"text"`
- [ ] `TaskSpec` has `input_schema: dict[str, Any]` field for modality-specific input specs (e.g., `{"image_size": [3, 224, 224], "channels_first": True}`)
- [ ] `TaskSpec` has `output_schema: dict[str, Any]` field for output specs (e.g., `{"num_classes": 10}`)
- [ ] `TaskSpec` has optional `preprocessing_config: dict[str, Any] | None` field for normalization/augmentation configs
- [ ] `task_type` field added: `Literal["lm", "seq2seq", "text_classification", "vision_classification", "vision_multilabel"]`
- [ ] Existing text configs (`lm_tiny`, `cls_tiny`, `seq2seq_tiny`) work without modification (backward compatibility)
- [ ] Type checking passes (`mypy utils/training/task_spec.py`) with no new errors
- [ ] Factory helpers (`build_task_spec` or similar) populate new fields with sensible defaults for text tasks
- [ ] Comprehensive docstrings added explaining each new field with vision + text examples
- [ ] Unit test validates that a vision `TaskSpec` can be created with all new fields populated
- [ ] Unit test validates that existing text task creation continues to work (regression test)
- [ ] Documentation in `docs/DEVELOPER_GUIDE_TASKS_EVAL.md` updated with vision TaskSpec example

## Test Scenarios

**Test Case 1: Vision Classification TaskSpec Creation**
- Given: Developer wants to create a vision classification task for CIFAR-10
- When: They instantiate `TaskSpec(modality="vision", task_type="vision_classification", input_schema={"image_size": [3, 32, 32]}, output_schema={"num_classes": 10})`
- Then: TaskSpec validates successfully, all fields accessible, type hints correct

**Test Case 2: Text Task Backward Compatibility**
- Given: Existing code creates text LM task with `TaskSpec(task_name="lm_tiny")` (no new fields)
- When: Code runs after this task is implemented
- Then: TaskSpec defaults to `modality="text"`, `task_type="lm"`, works identically to v3.5.0 behavior

**Test Case 3: Input Schema Validation for Vision**
- Given: Vision task with `input_schema={"image_size": [3, 224, 224], "channels_first": True}`
- When: Dataset loader or adapter queries `task_spec.input_schema["image_size"]`
- Then: Returns `[3, 224, 224]` correctly, enabling dynamic transform configuration

**Test Case 4: Factory Helper for Text Tasks**
- Given: `build_task_spec("lm_tiny")` is called (existing factory pattern)
- When: Function populates TaskSpec for language modeling
- Then: Returns TaskSpec with `modality="text"`, `task_type="lm"`, `input_schema={"max_seq_len": 128}`, backward compatible

**Test Case 5: Type Checking for Modality Enum**
- Given: Developer tries to set `modality="invalid_modality"`
- When: MyPy runs static type checking
- Then: Type error raised: "Literal['text', 'vision', 'audio', 'tabular'] expected, got 'invalid_modality'"

**Test Case 6: Optional Preprocessing Config**
- Given: Vision task may or may not have custom preprocessing
- When: `preprocessing_config=None` or `preprocessing_config={"normalize": True, "mean": [0.485, 0.456, 0.406]}`
- Then: Both cases handled gracefully, downstream code checks for None before accessing

**Test Case 7: Output Schema for Multi-Label Vision**
- Given: Vision multi-label task (e.g., COCO object detection)
- When: `output_schema={"num_classes": 80, "multi_label": True}`
- Then: Adapter can use this to configure `BCEWithLogitsLoss` instead of `CrossEntropyLoss`

**Test Case 8: Documentation Examples Validate**
- Given: Updated `DEVELOPER_GUIDE_TASKS_EVAL.md` contains vision TaskSpec example
- When: Developer copy-pastes example into Python REPL
- Then: Code executes without errors, produces valid TaskSpec instance

## Technical Implementation

**Required Components:**

1. **`utils/training/task_spec.py`** (primary changes)
   - Add new fields to `TaskSpec` dataclass/class:
     ```python
     @dataclass
     class TaskSpec:
         # Existing fields
         task_name: str
         dataset_name: str | None = None

         # NEW: Modality fields (v4.0.0)
         modality: Literal["text", "vision", "audio", "tabular"] = "text"
         task_type: Literal["lm", "seq2seq", "text_classification",
                            "vision_classification", "vision_multilabel"] = "lm"
         input_schema: dict[str, Any] = field(default_factory=dict)
         output_schema: dict[str, Any] = field(default_factory=dict)
         preprocessing_config: dict[str, Any] | None = None
     ```
   - Update `__post_init__` or validation to check schema consistency
   - Add helper methods: `is_text()`, `is_vision()`, `get_input_shape()`, etc.

2. **Factory helper updates** (if `build_task_spec` exists)
   - Populate `input_schema` for text: `{"max_seq_len": 128, "vocab_size": 50257}`
   - Set `task_type` based on task name patterns (e.g., "lm" → `task_type="lm"`)

3. **Type stubs and mypy config**
   - Ensure `Literal` types imported from `typing` (Python 3.10+)
   - Validate no type errors in `mypy.ini` scope

**Validation Commands:**

```bash
# Type checking
mypy utils/training/task_spec.py --config-file mypy.ini

# Manual validation (no automated tests for TaskSpec yet)
python -c "
from utils.training.task_spec import TaskSpec
from typing import Literal

# Test vision task
vision_task = TaskSpec(
    task_name='vision_tiny',
    modality='vision',
    task_type='vision_classification',
    input_schema={'image_size': [3, 32, 32], 'channels_first': True},
    output_schema={'num_classes': 10},
    preprocessing_config={'normalize': True, 'mean': [0.5, 0.5, 0.5]}
)
assert vision_task.modality == 'vision'
assert vision_task.input_schema['image_size'] == [3, 32, 32]
print('✓ Vision TaskSpec validated')

# Test backward compatibility
text_task = TaskSpec(task_name='lm_tiny')
assert text_task.modality == 'text'
assert text_task.task_type == 'lm'
print('✓ Text TaskSpec backward compatible')
"
```

**Code Patterns:**

Follow existing `TaskSpec` patterns from `utils/training/task_spec.py`:
- Use `@dataclass` for clean field definitions
- Provide `default_factory=dict` for mutable defaults (avoid shared dict instances)
- Add validation in `__post_init__` if needed
- Use type hints extensively for IDE support

## Dependencies

**Hard Dependencies** (must be complete first):
- None - This is a foundation task

**Soft Dependencies** (nice to have):
- Existing training infrastructure (T001-T065) provides context but not blocking

**External Dependencies:**
- Python 3.10+ for `Literal` type hints
- No new package dependencies (pure Python refactor)

## Design Decisions

**Decision 1: Use dict for schemas instead of dataclasses**
- **Rationale**: Flexibility for diverse modalities with varying schema requirements; avoid creating 10+ schema classes
- **Alternatives**: Create `VisionInputSchema`, `AudioInputSchema` dataclasses (more type safety but rigid)
- **Trade-offs**: Less type safety (dict keys are strings), but easier to extend for new modalities

**Decision 2: Make preprocessing_config optional**
- **Rationale**: Not all tasks need custom preprocessing (e.g., pre-normalized datasets)
- **Alternatives**: Make required with empty dict default (more consistent but forces boilerplate)
- **Trade-offs**: Downstream code must check for None, but reduces API clutter

**Decision 3: task_type separate from modality**
- **Rationale**: Modality = input type (vision/text), task_type = semantic task (classification/generation); orthogonal concerns
- **Alternatives**: Nest task_type under modality (e.g., `vision.classification`) - less composable
- **Trade-offs**: Two fields instead of one, but enables future multimodal tasks (vision + text → VQA)

**Decision 4: Backward compatibility via defaults**
- **Rationale**: Existing text tasks must work without code changes (zero migration cost)
- **Alternatives**: Require explicit migration (cleaner API but breaks existing code)
- **Trade-offs**: Default values must be sensible for text tasks (modality="text", task_type="lm")

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Schema dict keys typo errors (e.g., "imag_size" vs "image_size") | H - Silent failures in dataset loaders | M | Add schema validation helpers; document canonical keys; use constants for common keys |
| Backward compatibility breaks existing notebooks | H - Users cannot run v3.5.0 code | L | Comprehensive regression tests; test with existing `lm_tiny` configs; default values cover legacy usage |
| MyPy type errors from dict[str, Any] | M - Type checking less effective | H | Accept as trade-off; add runtime validation in `__post_init__`; document expected keys clearly |
| Confusion between modality and task_type | M - Developers misuse fields | M | Clear docstrings with examples; validation that checks consistency (e.g., vision modality can't have task_type="lm") |
| Schema grows unwieldy for complex tasks | M - Dict becomes 20+ keys | L | Limit to essential keys; use nested dicts if needed; review in MM-03 when used by dataset loaders |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Foundation task for enhancement1.0.md multimodal platform
**Dependencies:** None - critical path starter
**Estimated Complexity:** Standard (involves schema design + backward compatibility)

## Completion Checklist

**Code Implementation:**
- [ ] `TaskSpec` class updated with 5 new fields (modality, task_type, input_schema, output_schema, preprocessing_config)
- [ ] Type hints added for all new fields using Literal and dict[str, Any]
- [ ] Default values set to maintain backward compatibility (modality="text", task_type="lm")
- [ ] Docstrings updated with vision + text examples for each field
- [ ] Factory helpers updated to populate new fields for text tasks

**Testing:**
- [ ] Manual validation script runs successfully (vision + text TaskSpec creation)
- [ ] MyPy type checking passes with no new errors
- [ ] Regression test: existing text task configs work unchanged
- [ ] Vision TaskSpec can be instantiated with all fields

**Documentation:**
- [ ] `docs/DEVELOPER_GUIDE_TASKS_EVAL.md` updated with:
  - Vision TaskSpec example
  - Input/output schema format documentation
  - Modality + task_type field explanations
- [ ] Code comments added explaining design rationale for dict schemas
- [ ] Examples showing optional vs required fields

**Integration:**
- [ ] No breaking changes to existing code (verified by running training.ipynb cells 1-10)
- [ ] New fields accessible by downstream code (MM-02, MM-03, MM-04)
- [ ] Validation logic prevents invalid combinations (e.g., modality="vision" + task_type="lm")

**Quality Gates:**
- [ ] All 12 acceptance criteria checked
- [ ] All 8 test scenarios pass manual validation
- [ ] 5 design decisions documented with rationale
- [ ] 5 risks identified with concrete mitigations
- [ ] Token estimate (12,000) reasonable for scope

**Definition of Done:**
Task is complete when ALL acceptance criteria are met, type checking passes, existing text tasks work without modification, and vision TaskSpec creation is validated. Documentation must include runnable examples for both text and vision tasks.
