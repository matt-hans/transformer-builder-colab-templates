# Training Pipeline Refactor & General Purpose Upgrade Plan

## 1. Vision & Objectives
The goal is to transform the current monolithic, Causal-LM-specific training script (`tier3_training_utilities.py`) into a **modular, general-purpose training engine** capable of handling diverse AI tasks (Text, Vision, Classification) and custom datasets.

**Key Objectives:**
1.  **Decomposition**: Break the 1700-line file into focused, maintainable modules within `utils/training/engine/`.
2.  **Task Agnosticism**: Remove hardcoded Causal LM assumptions (e.g., token shifting) and use `TaskSpec` to drive execution logic via a Strategy Pattern.
3.  **Data Flexibility**: Support custom `collate_fn` (fixing the current crash), dictionary inputs, and dynamic padding.
4.  **Robustness**: Isolate complex logic (AMP, Logging, Visualization) from the core training loop.

## 2. Proposed Architecture: `utils.training.engine`

We will replace the single `tier3_training_utilities.py` file with a structured package `utils/training/engine/`.

### 2.1 Module Structure
```text
utils/training/engine/
├── __init__.py              # Exposes test_fine_tuning (backward compatibility)
├── trainer.py               # Main entry point (High-level orchestration)
├── loop.py                  # Epoch execution logic (_run_training_epoch)
├── data.py                  # Data setup, DataLoader creation, Collate handling
├── loss.py                  # Task-aware loss computation strategies
├── metrics.py               # Metric calculation and logging adapters
├── visualization.py         # Plotting and dashboarding
└── checkpoint.py            # State management and snapshots
```

### 2.2 Component Responsibilities

| Module | Responsibility | Key Changes |
| :--- | :--- | :--- |
| **`trainer.py`** | Orchestrates the setup, loop, and teardown. | Accepts `TaskSpec` and `collate_fn`. Selects appropriate strategies. |
| **`loop.py`** | Iterates over batches. | Handles `Dict` batches. Delegates loss/backward to `loss.py`. |
| **`data.py`** | Prepares DataLoaders. | **Crucial:** Accepts `collate_fn`. Filters empty samples. |
| **`loss.py`** | Computes loss & gradients. | Implements Strategy Pattern based on `TaskSpec` (LM vs. Vision vs. BERT). |
| **`metrics.py`** | Tracks performance. | Decouples metric logic from loop. Supports task-specific metrics. |

## 3. The "General Purpose" Core: Task Strategies

To solve the "Hardcoded Causal LM" issue, we will implement a Strategy Pattern.

### 3.1 The `LossStrategy` Interface
Instead of hardcoding `F.cross_entropy(shift_logits, shift_labels)`, we define strategies:

```python
class LossStrategy(ABC):
    @abstractmethod
    def compute_loss(self, model_output, batch, task_spec) -> torch.Tensor:
        pass

class CausalLMLoss(LossStrategy):
    def compute_loss(self, model_output, batch, task_spec):
        # Current logic: Shift tokens
        logits = model_output[:, :-1, :].contiguous()
        labels = batch['input_ids'][:, 1:].contiguous()
        return F.cross_entropy(logits.view(-1), labels.view(-1))

class ClassificationLoss(LossStrategy):
    def compute_loss(self, model_output, batch, task_spec):
        # Vision/Text Classification: Direct comparison
        return F.cross_entropy(model_output, batch['labels'])

class MaskedLMLoss(LossStrategy):
    def compute_loss(self, model_output, batch, task_spec):
        # BERT style: 1:1 mapping
        return F.cross_entropy(model_output.view(-1), batch['labels'].view(-1))
```

### 3.2 Dynamic Selection
In `trainer.py`, we select the strategy based on `TaskSpec`:

```python
def get_loss_strategy(task_spec):
    if task_spec.task_type == 'lm':
        return CausalLMLoss()
    elif task_spec.task_type == 'classification':
        return ClassificationLoss()
    # ...
```

## 4. Addressing Specific Failures

### 4.1 Fix: The `RuntimeError` (Stack Mismatch)
**Location:** `utils/training/engine/data.py`
**Solution:**
1.  Modify `_setup_training_environment` to accept `collate_fn`.
2.  If `collate_fn` is provided (from `TaskSpec` or user), pass it to `DataLoader`.
3.  If not provided, use a smart default that handles dictionary inputs (unlike `default_collate`).

### 4.2 Fix: Empty Tensor Crash
**Location:** `utils/training/engine/data.py`
**Solution:**
Add a sanitation pass before creating the `DataLoader`:
```python
def sanitize_dataset(dataset):
    # Filter out empty samples that break batching
    return dataset.filter(lambda x: len(x['input_ids']) > 0)
```

## 5. Refactor Execution Plan

### Phase 1: Extraction & Decomposition
1.  Create the `utils/training/engine/` directory.
2.  Move `_create_training_visualization` to `visualization.py`.
3.  Move `_setup_training_environment` to `data.py`.
4.  Move `_compute_loss_and_backward` to `loss.py` (initially as a function, then refactor).

### Phase 2: Generalization (The Upgrade)
1.  **Upgrade Data Loading:** Update `data.py` to support `collate_fn` and `Dict` inputs.
2.  **Implement Strategies:** Create the `LossStrategy` classes in `loss.py`.
3.  **Update Loop:** Modify `loop.py` to use `LossStrategy` instead of hardcoded logic.

### Phase 3: Integration & Verification
1.  Update `training.ipynb` to import from the new package (or re-export from `tier3` for compatibility).
2.  Run the "Causal LM" test (WikiText).
3.  Run a "Vision" test (Synthetic) to prove general purpose capability.

## 6. Benefits
-   **True General Purpose:** Can train Vision Transformers, BERT, and GPT models with the same pipeline.
-   **Maintainability:** Smaller files, clear responsibilities.
-   **Extensibility:** Adding a new task (e.g., Object Detection) only requires adding a new `LossStrategy`.
