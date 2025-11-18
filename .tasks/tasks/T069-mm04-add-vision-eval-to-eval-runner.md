---
id: T069
enhancement_id: MM-04
title: Add Vision Evaluation to eval_runner.py with Metrics Routing
status: pending
priority: 1
agent: backend
dependencies: [T066, T067, T068]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [multimodal, evaluation, vision, metrics, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/DEVELOPER_GUIDE_TASKS_EVAL.md

est_tokens: 11000
actual_tokens: null
---

## Description

Generalize `eval_runner.py` to support vision classification evaluation by adding modality-aware metrics routing. This extends the existing text-focused evaluation framework to compute accuracy, top-k accuracy, and per-class metrics for vision tasks.

The implementation adds a metrics factory that selects appropriate metric sets based on `task_spec.modality`, enabling `run_eval` to work seamlessly with both text and vision models. Metrics are computed via the `VisionClassificationAdapter` and aggregated across batches.

**Technical Approach**: Add modality detection logic to `run_eval`, implement vision metrics (accuracy, top-3/top-5 accuracy), and integrate with `EvalConfig` to support vision_tiny dataset. Reuse existing eval loop structure with minimal branching.

**Integration Points**:
- `utils/training/eval_runner.py` (primary changes)
- `utils/adapters/model_adapter.py` (VisionClassificationAdapter.compute_metrics)
- `cli/run_training.py` (will call run_eval for vision tasks)
- `training.ipynb` (will use for vision model validation)

## Business Context

**User Story**: As a vision model developer, I want to evaluate my model on a held-out test set using the same `run_eval` function as text models, so that I can track validation accuracy without writing custom evaluation code.

**Why This Matters**:
- **Unified evaluation**: Same API for text perplexity and vision accuracy reduces cognitive load
- **Standard metrics**: Top-k accuracy matches industry benchmarks (ImageNet evaluation)
- **Enables monitoring**: Metrics can be logged to W&B and ExperimentDB for tracking

**What It Unblocks**:
- MM-05: CLI integration (run_training.py needs eval for vision tasks)
- MO-02: Regression testing (baseline vs candidate comparison needs eval metrics)
- Tier 3 vision training (fine-tuning loop needs validation metrics)

**Priority Justification**: Priority 1 - Required for vision training pipeline; blocks MM-05 and monitoring tier.

## Acceptance Criteria

- [ ] `run_eval` function accepts `task_spec` with `modality="vision"` without errors
- [ ] Vision metrics computed: accuracy (top-1), top-3 accuracy, top-5 accuracy (if num_classes >= 5)
- [ ] Metrics aggregated across all batches correctly (not per-batch average)
- [ ] `EvalConfig` can point to vision_tiny dataset via `task_name="vision_tiny"`
- [ ] `build_eval_config` helper updated to handle vision tasks (if it exists)
- [ ] Type hints added to metrics computation functions
- [ ] Docstrings explain vision-specific metric calculations
- [ ] Unit test validates top-k accuracy calculation with known predictions
- [ ] Integration test runs eval on dummy vision model for 1 epoch
- [ ] No regressions in text evaluation (existing LM eval still works)

## Test Scenarios

**Test Case 1: Vision Accuracy Computation**
- Given: Vision model evaluated on 16-image dataset, 12 correct predictions, 4 incorrect
- When: `run_eval(model, adapter, eval_config, task_spec)` called
- Then: Returns `{"accuracy": 0.75}` (12/16 = 0.75)

**Test Case 2: Top-5 Accuracy**
- Given: 10-class vision model, predictions have true class in top-5 for 14/16 images
- When: Metrics computed
- Then: Returns `{"top5_accuracy": 0.875}` (14/16)

**Test Case 3: Batch Aggregation**
- Given: Eval dataset split into 4 batches of 4 images each
- When: run_eval iterates over batches and aggregates metrics
- Then: Final accuracy is global (not average of per-batch accuracies)

**Test Case 4: EvalConfig with Vision Dataset**
- Given: `EvalConfig(task_name="vision_tiny", batch_size=4)`
- When: `build_eval_config` or direct EvalConfig creation
- Then: Config points to correct vision dataset path, DataLoader created successfully

**Test Case 5: CLI Integration**
- Given: `python -m cli.run_training --config configs/example_train_vision.json` with eval enabled
- When: Training completes 2 epochs
- Then: Validation accuracy logged each epoch to W&B/console

**Test Case 6: Per-Class Accuracy (Optional)**
- Given: Vision model with 4 classes, unbalanced accuracy per class
- When: Metrics include per-class breakdown
- Then: Returns `{"class_0_acc": 0.8, "class_1_acc": 0.6, ...}` (optional feature)

**Test Case 7: Text Eval Regression Test**
- Given: Existing text LM with perplexity eval
- When: run_eval called after this task implemented
- Then: Text metrics still computed correctly, no errors

**Test Case 8: Missing Modality Field Fallback**
- Given: Old TaskSpec without modality field (backward compatibility)
- When: run_eval infers modality from task_type or defaults to text
- Then: Evaluation runs without errors, assumes text metrics

## Technical Implementation

**Required Components:**

1. **`utils/training/eval_runner.py`** (extend with vision support)
   ```python
   def run_eval(
       model: nn.Module,
       adapter: ModelAdapter,
       eval_config: EvalConfig,
       task_spec: TaskSpec,
       use_wandb: bool = False,
   ) -> dict[str, float]:
       """Run evaluation on held-out set, supports text and vision tasks."""

       # Build eval dataloader
       eval_loader = build_dataloader(
           task_spec,
           task_name=eval_config.task_name,
           batch_size=eval_config.batch_size,
           shuffle=False,
       )

       model.eval()
       all_preds = []
       all_labels = []
       total_loss = 0.0
       num_batches = 0

       with torch.no_grad():
           for batch in eval_loader:
               # Move batch to device
               batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

               # Forward pass via adapter
               outputs = adapter.forward(model, batch, task_spec)
               loss = adapter.compute_loss(outputs, batch, task_spec)

               total_loss += loss.item()
               num_batches += 1

               # Collect predictions for metrics
               if task_spec.modality == "vision":
                   preds = outputs["logits"].argmax(dim=-1).cpu()
                   all_preds.append(preds)
                   all_labels.append(batch["labels"].cpu())

       # Compute modality-specific metrics
       metrics = {}
       metrics["loss"] = total_loss / num_batches

       if task_spec.modality == "vision":
           all_preds = torch.cat(all_preds)
           all_labels = torch.cat(all_labels)

           # Top-1 accuracy
           metrics["accuracy"] = (all_preds == all_labels).float().mean().item()

           # Top-k accuracy (if applicable)
           num_classes = task_spec.output_schema.get("num_classes", 10)
           if num_classes >= 5:
               # Recompute logits for top-k (need to store during loop)
               # Simplified: assume accuracy computed correctly
               metrics["top5_accuracy"] = compute_topk_accuracy(all_preds, all_labels, k=5)

       elif task_spec.modality == "text":
           metrics["perplexity"] = math.exp(metrics["loss"])

       return metrics
   ```

2. **Helper function for top-k accuracy**
   ```python
   def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
       """Compute top-k accuracy: true label in top-k predictions."""
       topk_preds = logits.topk(k, dim=-1).indices  # [N, k]
       labels_expanded = labels.unsqueeze(-1).expand_as(topk_preds)  # [N, k]
       correct = (topk_preds == labels_expanded).any(dim=-1).float().mean().item()
       return correct
   ```

3. **Update `build_eval_config` (if exists)**
   - Add vision task name handling
   - Set appropriate batch size for vision (e.g., 8 for 64x64 images)

**Validation Commands:**

```bash
# Manual unit test
python -c "
import torch
from utils.training.eval_runner import run_eval, compute_topk_accuracy
from utils.adapters.model_adapter import VisionClassificationAdapter
from utils.training.task_spec import TaskSpec
from utils.training.eval_config import EvalConfig

# Test top-k accuracy
logits = torch.tensor([
    [0.1, 0.8, 0.05, 0.05],  # Pred: 1, Label: 1 → Correct
    [0.6, 0.2, 0.1, 0.1],    # Pred: 0, Label: 2 → Top-5: Yes
    [0.1, 0.1, 0.7, 0.1],    # Pred: 2, Label: 2 → Correct
])
labels = torch.tensor([1, 2, 2])
acc = compute_topk_accuracy(logits, labels, k=2)
print(f'Top-2 accuracy: {acc}')  # Should be 1.0 (all correct in top-2)

print('✓ Top-k accuracy validated')

# Integration test requires model + dataset (manual validation in training.ipynb)
"
```

**Code Patterns:**

- Reuse existing eval loop structure from text evaluation
- Use adapter.forward and adapter.compute_loss for modality abstraction
- Collect predictions in lists, concatenate after loop (not in-place append to tensor)
- Use torch.no_grad() for memory efficiency
- Log metrics to W&B if `use_wandb=True`

## Dependencies

**Hard Dependencies** (must be complete first):
- [T066] Extend TaskSpec to Support Modalities - Provides modality field
- [T067] Add VisionClassificationAdapter - Computes vision metrics
- [T068] Extend Dataset Utilities with Image Loaders - Provides vision DataLoader

**Soft Dependencies**:
- None

**External Dependencies:**
- PyTorch 2.6+ (already required)
- No new packages

## Design Decisions

**Decision 1: Compute top-k accuracy only if num_classes >= k**
- **Rationale**: Top-5 meaningless for 3-class problem; avoid confusing metrics
- **Alternatives**: Always compute top-3, top-5, top-10 (cluttered output)
- **Trade-offs**: Need conditional logic, but cleaner metrics dictionary

**Decision 2: Aggregate predictions across batches, not average per-batch metrics**
- **Rationale**: Batch averages can be skewed by uneven batch sizes (last batch may be smaller)
- **Alternatives**: Weighted average of per-batch metrics (more complex, no benefit)
- **Trade-offs**: Must store predictions in memory (negligible for tiny eval sets)

**Decision 3: Store logits for top-k, not just argmax predictions**
- **Rationale**: Top-k needs full probability distribution, argmax loses information
- **Alternatives**: Store top-k indices during loop (more memory-efficient but less flexible)
- **Trade-offs**: Slightly higher memory usage, but enables future metrics like ECE (Expected Calibration Error)

**Decision 4: Use modality field for routing, not task_type**
- **Rationale**: Modality (text/vision) determines metric type; task_type (lm/classification) determines subtask
- **Alternatives**: Use task_type for routing (less clear for multimodal tasks)
- **Trade-offs**: Assumes modality is always set (safe after T066)

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Top-k accuracy incorrect for edge cases (k > num_classes) | M - Metrics report nonsense values | M | Add validation: if k > num_classes, skip or cap k to num_classes |
| Memory overflow storing logits for large eval sets | M - OOM in Colab | L | For eval sets >10k images, compute metrics incrementally without storing all logits |
| Text eval regression due to modality branching | H - Breaks existing notebooks | L | Comprehensive regression tests; modality defaults to "text" if not set |
| Missing top-k in adapter.compute_metrics | M - Duplicate logic | M | Keep top-k in eval_runner (not adapter) since it requires full eval set, not per-batch |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Fourth task in multimodal foundation (MM-04 from enhancement1.0.md)
**Dependencies:** T066 (TaskSpec), T067 (VisionAdapter), T068 (Dataset)
**Estimated Complexity:** Standard (metrics computation + integration)

## Completion Checklist

**Code Implementation:**
- [ ] `run_eval` extended with modality-aware metrics routing
- [ ] Vision metrics: accuracy (top-1), top-3, top-5 (conditional on num_classes)
- [ ] Helper function `compute_topk_accuracy` implemented
- [ ] Predictions aggregated across batches correctly
- [ ] `build_eval_config` handles vision tasks (if function exists)

**Testing:**
- [ ] Unit test: top-k accuracy calculation with known logits/labels
- [ ] Integration test: run_eval on vision_tiny dataset returns correct accuracy
- [ ] Regression test: text LM eval still works (perplexity computed)
- [ ] Manual validation: accuracy matches expected value for dummy model

**Documentation:**
- [ ] Docstrings updated for run_eval with vision example
- [ ] Top-k accuracy calculation explained in comments
- [ ] `docs/DEVELOPER_GUIDE_TASKS_EVAL.md` updated with vision eval example

**Integration:**
- [ ] Works with cli/run_training.py for vision tasks
- [ ] Compatible with training.ipynb validation cells
- [ ] Metrics log to W&B correctly (if use_wandb=True)

**Quality Gates:**
- [ ] All 10 acceptance criteria checked
- [ ] All 8 test scenarios validated
- [ ] 4 design decisions documented
- [ ] 4 risks with mitigations
- [ ] Token estimate (11,000) appropriate

**Definition of Done:**
Task is complete when run_eval computes vision metrics correctly, top-k accuracy validated with unit tests, text evaluation still works, and vision eval integrates into training pipeline.
