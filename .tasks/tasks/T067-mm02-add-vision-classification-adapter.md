---
id: T067
enhancement_id: MM-02
title: Add VisionClassificationAdapter to ModelAdapter Family
status: pending
priority: 1
agent: backend
dependencies: [T066]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [multimodal, adapters, vision, enhancement1.0, critical-path]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/DEVELOPER_GUIDE_TASKS_EVAL.md

est_tokens: 10000
actual_tokens: null
---

## Description

Implement a `VisionClassificationAdapter` that extends the `ModelAdapter` abstraction to support vision classification tasks. This adapter will handle forward passes, loss computation, and metrics calculation for image classification models, enabling the training infrastructure to work with CNNs and vision transformers.

The adapter follows the existing `ModelAdapter` pattern (similar to `DecoderOnlyLMAdapter`), providing a unified interface for model interactions across modalities. It consumes the `TaskSpec` created in MM-01 to configure input expectations and output handling.

**Technical Approach**: Create a new adapter class implementing `forward()`, `compute_loss()`, and `compute_metrics()` methods for vision tasks. The adapter expects batch dictionaries with `pixel_values` keys and uses F.cross_entropy for classification loss. Integrate into adapter registry/factory for auto-selection based on `task_spec.task_type`.

**Integration Points**:
- `utils/adapters/model_adapter.py` (add VisionClassificationAdapter)
- `utils/adapters/__init__.py` (export new adapter)
- Tier 1/2 tests (will call adapter.forward with dummy images)
- `utils/tier3_training_utilities.py` (will use adapter in training loops)

## Business Context

**User Story**: As a developer training a vision model, I want the same adapter-based training interface as text models, so that I can reuse training loops, metrics tracking, and export utilities without modality-specific code.

**Why This Matters**:
- **Unifies training interface**: Vision models train using same `TrainingCoordinator` as text models
- **Enables architecture diversity**: Works with CNNs (ResNet), vision transformers (ViT), custom architectures
- **Reuses MLOps stack**: W&B logging, checkpointing, export tier all work automatically

**What It Unblocks**:
- MM-03: Dataset loaders (need adapter to process batches during training)
- MM-04: Vision evaluation (adapter computes metrics)
- MM-05: CLI integration (adapter wires vision models into run_tiers.py)
- EX-02: Export validation (Tier 4 tests use adapter for parity checks)

**Priority Justification**: Priority 1 - Required for any vision model training; blocks MM-03, MM-04, and MM-05.

## Acceptance Criteria

- [ ] `VisionClassificationAdapter` class created in `utils/adapters/model_adapter.py`
- [ ] `forward(model, batch, task_spec)` method expects `batch["pixel_values"]` shaped `[B, C, H, W]` and returns `{"logits": Tensor}`
- [ ] `compute_loss(outputs, batch, task_spec)` computes `F.cross_entropy(logits, labels)` correctly
- [ ] `compute_metrics(outputs, batch, task_spec)` calculates accuracy: `(preds == labels).float().mean()`
- [ ] `task_type` class attribute set to `"vision_classification"` for adapter routing
- [ ] Adapter integrates into factory pattern (e.g., `get_adapter_for(task_spec)` returns VisionClassificationAdapter for vision tasks)
- [ ] Unit test with dummy CNN (e.g., 3-layer ConvNet) and synthetic batch `[4, 3, 32, 32]` validates forward pass
- [ ] Unit test validates loss computation returns scalar tensor with `requires_grad=True`
- [ ] Unit test validates metrics return dict with "accuracy" key in range [0.0, 1.0]
- [ ] No regressions: existing text LM adapter tests still pass
- [ ] Type hints added to all adapter methods
- [ ] Docstrings explain expected batch format and output format

## Test Scenarios

**Test Case 1: Forward Pass with Dummy CNN**
- Given: Dummy CNN model with Conv2d → ReLU → Linear → output[B, 10], batch `{"pixel_values": [4, 3, 32, 32], "labels": [4]}`
- When: `adapter.forward(model, batch, task_spec)` called
- Then: Returns `{"logits": Tensor[4, 10]}` with correct shape

**Test Case 2: Loss Computation**
- Given: Model outputs logits `[4, 10]`, batch has labels `[2, 5, 1, 9]`
- When: `adapter.compute_loss(outputs, batch, task_spec)` called
- Then: Returns scalar tensor ~2.3 (cross-entropy for random logits), requires_grad=True for backprop

**Test Case 3: Metrics Calculation - Perfect Predictions**
- Given: Predictions `[0, 1, 2, 3]`, labels `[0, 1, 2, 3]`
- When: `adapter.compute_metrics(outputs, batch, task_spec)` called
- Then: Returns `{"accuracy": 1.0}`

**Test Case 4: Metrics Calculation - Partial Correct**
- Given: Predictions `[0, 1, 9, 3]`, labels `[0, 1, 2, 3]` (2/4 correct)
- When: Metrics computed
- Then: Returns `{"accuracy": 0.5}`

**Test Case 5: Adapter Factory Selection**
- Given: `task_spec` with `task_type="vision_classification"`, `modality="vision"`
- When: `get_adapter_for(task_spec)` called
- Then: Returns instance of `VisionClassificationAdapter`, not `DecoderOnlyLMAdapter`

**Test Case 6: Tier 1 Shape Test Integration**
- Given: Vision model + `VisionClassificationAdapter`, Tier 1 shape test with batches `[1, 3, 32, 32]`, `[4, 3, 64, 64]`
- When: `test_shape_robustness(model, config, adapter)` runs
- Then: All shape tests pass, outputs have correct dimensions

**Test Case 7: Gradient Flow Validation**
- Given: Dummy CNN, adapter computes loss on synthetic batch
- When: `loss.backward()` called
- Then: All Conv2d and Linear layers have non-None gradients (no vanishing)

**Test Case 8: Missing pixel_values Key**
- Given: Batch has `{"images": [...], "labels": [...]}` (wrong key name)
- When: `adapter.forward()` called
- Then: Raises `KeyError` with helpful message: "Expected 'pixel_values' key in batch, found keys: ['images', 'labels']"

## Technical Implementation

**Required Components:**

1. **`utils/adapters/model_adapter.py`** (add new class)
   ```python
   class VisionClassificationAdapter(ModelAdapter):
       """Adapter for vision classification tasks (CNNs, ViTs)."""
       task_type = "vision_classification"

       def forward(
           self,
           model: nn.Module,
           batch: dict[str, torch.Tensor],
           task_spec: TaskSpec
       ) -> dict[str, torch.Tensor]:
           """
           Forward pass for vision classification.

           Args:
               model: Vision model (expects [B, C, H, W] input)
               batch: {"pixel_values": [B, C, H, W], "labels": [B]}
               task_spec: Task specification with output_schema

           Returns:
               {"logits": [B, num_classes]} - raw logits before softmax
           """
           if "pixel_values" not in batch:
               raise KeyError(f"Expected 'pixel_values' in batch, found: {list(batch.keys())}")

           logits = model(batch["pixel_values"])
           return {"logits": logits}

       def compute_loss(
           self,
           outputs: dict[str, torch.Tensor],
           batch: dict[str, torch.Tensor],
           task_spec: TaskSpec
       ) -> torch.Tensor:
           """Compute cross-entropy loss for classification."""
           labels = batch["labels"]
           return F.cross_entropy(outputs["logits"], labels)

       def compute_metrics(
           self,
           outputs: dict[str, torch.Tensor],
           batch: dict[str, torch.Tensor],
           task_spec: TaskSpec
       ) -> dict[str, float]:
           """Compute accuracy metric."""
           preds = outputs["logits"].argmax(dim=-1)
           accuracy = (preds == batch["labels"]).float().mean().item()
           return {"accuracy": accuracy}
   ```

2. **`utils/adapters/__init__.py`** (export)
   ```python
   from .model_adapter import (
       ModelAdapter,
       DecoderOnlyLMAdapter,
       VisionClassificationAdapter,  # NEW
   )
   ```

3. **Adapter factory/registry** (if exists, else create)
   ```python
   def get_adapter_for(task_spec: TaskSpec) -> ModelAdapter:
       """Auto-select adapter based on task type."""
       if task_spec.task_type == "vision_classification":
           return VisionClassificationAdapter()
       elif task_spec.task_type == "lm":
           return DecoderOnlyLMAdapter()
       else:
           raise ValueError(f"Unknown task_type: {task_spec.task_type}")
   ```

**Validation Commands:**

```bash
# Type checking
mypy utils/adapters/model_adapter.py --config-file mypy.ini

# Manual unit test
python -c "
import torch
import torch.nn as nn
from utils.adapters.model_adapter import VisionClassificationAdapter
from utils.training.task_spec import TaskSpec

# Dummy CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleCNN()
task_spec = TaskSpec(
    task_name='vision_test',
    modality='vision',
    task_type='vision_classification',
    output_schema={'num_classes': 10}
)
adapter = VisionClassificationAdapter()

# Test forward
batch = {
    'pixel_values': torch.randn(4, 3, 32, 32),
    'labels': torch.tensor([0, 1, 2, 3])
}
outputs = adapter.forward(model, batch, task_spec)
assert outputs['logits'].shape == (4, 10), f'Expected (4, 10), got {outputs[\"logits\"].shape}'

# Test loss
loss = adapter.compute_loss(outputs, batch, task_spec)
assert loss.requires_grad, 'Loss must have requires_grad=True'
assert loss.ndim == 0, 'Loss must be scalar'

# Test metrics
metrics = adapter.compute_metrics(outputs, batch, task_spec)
assert 'accuracy' in metrics, 'Metrics must include accuracy'
assert 0.0 <= metrics['accuracy'] <= 1.0, f'Invalid accuracy: {metrics[\"accuracy\"]}'

print('✓ VisionClassificationAdapter validated')
"
```

**Code Patterns:**

- Follow existing `DecoderOnlyLMAdapter` structure for consistency
- Use dict batch format (not positional args) for extensibility
- Return dicts from `forward()` and `compute_metrics()` for structured outputs
- Add comprehensive docstrings with expected shapes
- Include input validation with helpful error messages

## Dependencies

**Hard Dependencies** (must be complete first):
- [T066] Extend TaskSpec to Support Modalities - Provides `task_type`, `modality` fields consumed by adapter

**Soft Dependencies** (nice to have):
- Existing `DecoderOnlyLMAdapter` implementation provides reference pattern

**External Dependencies:**
- PyTorch 2.6+ (already in requirements.txt)
- No new packages required

## Design Decisions

**Decision 1: Use pixel_values key instead of images**
- **Rationale**: Matches HuggingFace transformers convention (ViTFeatureExtractor outputs pixel_values)
- **Alternatives**: Use "images" key (more intuitive but inconsistent with HF datasets)
- **Trade-offs**: Slight learning curve for users unfamiliar with HF, but enables drop-in HF model compatibility

**Decision 2: Return dict from forward() instead of raw tensor**
- **Rationale**: Extensibility for multi-output models (e.g., vision + auxiliary loss), consistent with text adapters
- **Alternatives**: Return logits tensor directly (simpler but less flexible)
- **Trade-offs**: Extra dict unpacking in calling code, but enables future features like attention rollout

**Decision 3: Compute accuracy in adapter, not separate metrics module**
- **Rationale**: Simple metric tightly coupled to classification task; keeps adapter self-contained
- **Alternatives**: Create separate `VisionMetrics` class (more modular but overengineered for single metric)
- **Trade-offs**: Harder to add complex metrics later (top-5 accuracy, per-class metrics), but sufficient for MVP

**Decision 4: Use F.cross_entropy instead of manual softmax + NLL**
- **Rationale**: F.cross_entropy is numerically stable, applies log-softmax internally
- **Alternatives**: Separate nn.LogSoftmax() + nn.NLLLoss() (more verbose, no benefit)
- **Trade-offs**: None - F.cross_entropy is PyTorch best practice

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Adapter incompatible with vision transformers (ViT) | H - Cannot train ViT models | M | Test with both CNN and ViT architectures; ViTs also expect [B, C, H, W] input so should work |
| Missing labels key in batch causes cryptic error | M - Poor debugging experience | M | Add input validation with KeyError explaining required keys |
| Accuracy metric wrong for multi-label tasks | H - Incorrect metrics for multi-label | L | Document that adapter is for single-label only; create separate MultiLabelAdapter in future |
| Adapter registry pattern doesn't exist yet | M - Cannot auto-select adapter | H | Create minimal factory function `get_adapter_for(task_spec)` as part of this task |
| Tier 1 tests fail with vision models | M - Cannot validate vision architectures | L | Update `test_shape_robustness` to accept optional adapter parameter; fallback to direct model call if None |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Second task in multimodal foundation (MM-02 from enhancement1.0.md)
**Dependencies:** T066 (TaskSpec extension)
**Estimated Complexity:** Standard (adapter pattern implementation + integration)

## Completion Checklist

**Code Implementation:**
- [ ] `VisionClassificationAdapter` class created with task_type attribute
- [ ] `forward()` method handles pixel_values → logits correctly
- [ ] `compute_loss()` uses F.cross_entropy with proper shape handling
- [ ] `compute_metrics()` calculates accuracy as float in [0, 1]
- [ ] Adapter factory function `get_adapter_for(task_spec)` created
- [ ] Exported from `utils/adapters/__init__.py`

**Testing:**
- [ ] Unit test with dummy CNN validates forward pass (logits shape correct)
- [ ] Unit test validates loss computation (scalar, requires_grad=True)
- [ ] Unit test validates metrics (accuracy in [0, 1])
- [ ] Adapter factory test (vision task_type returns VisionClassificationAdapter)
- [ ] Regression test: text adapter tests still pass
- [ ] Manual validation with ResNet-18 or MobileNetV2 (if available)

**Documentation:**
- [ ] Docstrings added to all methods with Args/Returns/Raises sections
- [ ] `docs/DEVELOPER_GUIDE_TASKS_EVAL.md` updated with adapter usage example
- [ ] Batch format documented (pixel_values + labels keys)
- [ ] Output format documented (logits key)

**Integration:**
- [ ] Adapter works with Tier 1 shape tests (if updated to accept adapter param)
- [ ] Can be used in training loop with dummy vision model
- [ ] Compatible with future dataset loaders (MM-03)

**Quality Gates:**
- [ ] All 12 acceptance criteria checked
- [ ] All 8 test scenarios validated manually
- [ ] 4 design decisions documented
- [ ] 5 risks with mitigations
- [ ] Token estimate (10,000) appropriate

**Definition of Done:**
Task is complete when VisionClassificationAdapter works with dummy CNN, computes loss and accuracy correctly, integrates into adapter factory, and no regressions occur in text adapter tests.
