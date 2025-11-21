# Type System Documentation

This document explains the type system architecture used in the training pipeline engine modules.

## Overview

The training engine uses Python 3.10+ type hints with strict mypy validation. All engine modules (`utils/training/engine/`) pass `mypy --strict` with 0 errors.

## Type Hierarchy

### Core Type Categories

1. **Data Types** - Input/output data structures
2. **Protocol Types** - Duck-typed interfaces
3. **TypedDict** - Structured dictionaries
4. **Generic Types** - Parameterized containers
5. **Union Types** - Multiple type alternatives

## Protocol Pattern Usage

### DataModuleProtocol

Defines the interface for data modules:

```python
from typing import Protocol, Optional, runtime_checkable
from torch.utils.data import DataLoader

@runtime_checkable
class DataModuleProtocol(Protocol):
    """Protocol for data modules that provide train/val dataloaders."""

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        ...

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return validation dataloader (None if not available)."""
        ...
```

**Usage**:
```python
def setup_data(data_module: DataModuleProtocol) -> None:
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
```

**Runtime checking**:
```python
if isinstance(my_module, DataModuleProtocol):
    # Safe to use as data module
    train_loader = my_module.train_dataloader()
```

### LossStrategy Protocol

Defines the interface for loss computation strategies:

```python
from typing import Protocol
import torch

class LossStrategy(Protocol):
    """Protocol for loss computation strategies."""

    def compute(self, inputs: LossInputs) -> torch.Tensor:
        """Compute loss from inputs."""
        ...

    @property
    def name(self) -> str:
        """Strategy name for logging."""
        ...
```

**Implementations**:
- `LanguageModelingLoss` - Causal LM with padding mask
- `ClassificationLoss` - Cross-entropy for classification
- `VisionLoss` - Vision tasks (classification, multilabel)
- `PEFTAwareLoss` - Parameter-Efficient Fine-Tuning support
- `QuantizationSafeLoss` - Quantization-aware training

**Usage**:
```python
def train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    loss_strategy: LossStrategy
) -> torch.Tensor:
    model_output = model(batch['input_ids'])
    loss_inputs = prepare_loss_inputs(batch, model_output)
    return loss_strategy.compute(loss_inputs)
```

## TypedDict Usage

### LossInputs

Structured input for loss computation:

```python
from typing import TypedDict, Optional
import torch

class LossInputs(TypedDict, total=False):
    """Inputs for loss computation (all fields optional for flexibility)."""

    # Required for most tasks
    logits: torch.Tensor           # Model predictions [batch, seq_len, vocab_size]
    labels: torch.Tensor           # Ground truth labels [batch, seq_len]

    # Optional masks and metadata
    attention_mask: torch.Tensor   # Attention mask [batch, seq_len]
    pad_token_id: int              # Padding token ID to ignore
    task_type: str                 # Task type (lm, classification, vision)
```

**Usage**:
```python
def prepare_loss_inputs(
    batch: Dict[str, Any],
    model_output: torch.Tensor
) -> LossInputs:
    return LossInputs(
        logits=model_output,
        labels=batch['labels'],
        attention_mask=batch.get('attention_mask'),
        pad_token_id=batch.get('pad_token_id', 0)
    )
```

## Generic Type Parameters

### Proper Parameterization

Always provide type parameters for generic containers:

✅ **Good**:
```python
from typing import Dict, List, Optional, Callable, Any

def process_metrics(metrics: Dict[str, float]) -> List[str]:
    return list(metrics.keys())

def create_collator(config: Dict[str, Any]) -> Callable[..., Any]:
    return lambda batch: collate_fn(batch, **config)
```

❌ **Bad** (mypy --strict error):
```python
def process_metrics(metrics: dict) -> list:  # Missing type parameters!
    return list(metrics.keys())

def create_collator(config: dict) -> Callable:  # Missing parameters!
    return lambda batch: collate_fn(batch, **config)
```

### Common Generic Patterns

```python
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

# Dictionary with string keys and mixed values
config: Dict[str, Any] = {'lr': 0.001, 'epochs': 10}

# List of tensors
batch: List[torch.Tensor] = [tensor1, tensor2]

# Optional return value
def get_loader() -> Optional[DataLoader]:
    return loader if available else None

# Union types for multiple alternatives
data: Union[Dataset, HFDataset, List[torch.Tensor]]

# Tuple with mixed types
result: Tuple[torch.Tensor, float, int]

# Callable with parameters
transform: Callable[[torch.Tensor], torch.Tensor]
collate_fn: Callable[..., Any]  # Variable arguments
```

## Handling `Any` Types

### When `Any` is Acceptable

1. **Third-party library returns** with no type stubs:
   ```python
   import wandb  # No type stubs available
   wandb.log(metrics, step=epoch)  # type: ignore[attr-defined]
   ```

2. **Dynamic plugin systems** where type cannot be known:
   ```python
   def get_collator(...) -> Any:  # Could be various collator classes
       return registry.get(name)

   # When returning from function, document:
   collator = registry.get_collator(...)  # type: ignore[no-any-return]
   ```

3. **Compatibility with legacy code**:
   ```python
   def legacy_function(config: Any) -> Any:  # Legacy API
       ...
   ```

### Always Document `Any` Usage

```python
# Good: Documented
collator = self.collator_registry.get_collator(
    task_spec=task_spec,
    tokenizer=tokenizer
)
# get_collator returns Any (could be various collator classes)
# All collators are callable, so this is safe
return collator  # type: ignore[no-any-return]

# Bad: Undocumented
return self.collator_registry.get_collator(task_spec)  # Why Any?
```

## Type Ignore Comments

### Error Code Reference

Always include error codes in `# type: ignore` comments:

```python
# wandb module has no type stubs
import wandb  # type: ignore[import-untyped]
wandb.log(metrics)  # type: ignore[attr-defined]

# DataLoader not in UniversalDataModule signature (compatibility)
data_module = UniversalDataModule(
    train_data=train_data,  # type: ignore[arg-type]
    val_data=val_data        # type: ignore[arg-type]
)

# get_collator returns Any (various collator classes)
return collator  # type: ignore[no-any-return]
```

### Common Error Codes

- `[attr-defined]` - Attribute doesn't exist (common with missing stubs)
- `[arg-type]` - Argument type mismatch
- `[no-any-return]` - Returning Any from typed function
- `[import-untyped]` - Importing module without type stubs
- `[assignment]` - Type mismatch in assignment
- `[union-attr]` - Attribute access on Union type without narrowing

## Best Practices

### 1. Always Annotate Public APIs

```python
# Good: Full annotations
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossStrategy
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # ...
    return metrics

# Bad: Missing annotations
def train_epoch(model, dataloader, optimizer, loss_fn):
    metrics = {}
    # ...
    return metrics
```

### 2. Use Type Aliases for Clarity

```python
from typing import Dict, Any, Union, List
import torch

# Define aliases
BatchDict = Dict[str, torch.Tensor]
MetricsDict = Dict[str, float]
DatasetUnion = Union[Dataset, HFDataset, List[torch.Tensor]]

# Use in signatures
def process_batch(batch: BatchDict) -> MetricsDict:
    ...
```

### 3. Narrow Union Types When Possible

```python
from typing import Union

def process_data(data: Union[int, str]) -> str:
    # Narrow the type
    if isinstance(data, int):
        # mypy knows data is int here
        return str(data * 2)
    else:
        # mypy knows data is str here
        return data.upper()
```

### 4. Use TypeGuard for Complex Checks

```python
from typing import TypeGuard, Any, Dict

def is_batch_dict(obj: Any) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Type guard for batch dictionaries."""
    return (
        isinstance(obj, dict) and
        all(isinstance(k, str) for k in obj.keys()) and
        all(isinstance(v, torch.Tensor) for v in obj.values())
    )

# Usage
def process(data: Any) -> None:
    if is_batch_dict(data):
        # mypy knows data is Dict[str, torch.Tensor] here
        batch_size = data['input_ids'].size(0)
```

## Common Patterns

### 1. Optional Parameters with Defaults

```python
from typing import Optional

def create_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: Optional[float] = None
) -> torch.optim.Optimizer:
    wd = weight_decay if weight_decay is not None else 0.0
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
```

### 2. Dataclass with Type Annotations

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Training configuration with full type annotations."""

    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float = 0.01
    gradient_clip_norm: Optional[float] = None
    use_wandb: bool = False
```

### 3. Protocol Implementation

```python
from typing import Protocol

class Transformer(nn.Module, DataModuleProtocol):
    """Model that implements DataModuleProtocol."""

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> Optional[DataLoader]:
        return self._val_loader
```

## Troubleshooting

### Problem: "Returning Any from function"

```python
# Problem
def get_value(d: Dict[str, Any]) -> int:
    return d.get('key', 0)  # Error: Returning Any

# Solution 1: Cast
def get_value(d: Dict[str, Any]) -> int:
    value = d.get('key', 0)
    return int(value)

# Solution 2: Type assertion
def get_value(d: Dict[str, Any]) -> int:
    return cast(int, d.get('key', 0))
```

### Problem: "Missing type parameters"

```python
# Problem
def process(items: list) -> dict:  # Error: Missing type parameters

# Solution
def process(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
```

### Problem: "Incompatible types in assignment"

```python
# Problem
x: int = some_dict.get('key', 0)  # Error: get returns int | None

# Solution 1: Type narrowing
value = some_dict.get('key', 0)
x: int = int(value) if isinstance(value, (int, float)) else 0

# Solution 2: Default ensures non-None
x: int = some_dict.get('key', 0)  # If default is provided, not None
```

## IDE Integration

### VS Code

Install Python extension and configure:

```json
// .vscode/settings.json
{
    "python.linting.mypyEnabled": true,
    "python.linting.mypyArgs": [
        "--strict",
        "--show-error-codes",
        "--config-file=mypy.ini"
    ]
}
```

### PyCharm

Enable type checking:
- Settings → Editor → Inspections → Python → Type Checker
- Check "Mypy"
- Configure mypy path and arguments

## Testing Type Annotations

Run mypy validation:

```bash
# Check specific module
mypy --strict --show-error-codes utils/training/engine/trainer.py

# Check entire package
mypy --strict --show-error-codes utils/training/engine/

# Generate coverage report
mypy --strict --html-report mypy-report utils/training/engine/
```

## Summary

✅ Use type annotations for all public APIs
✅ Parameterize generic types (`Dict[str, int]`, not `dict`)
✅ Use Protocols for duck typing
✅ Use TypedDict for structured dictionaries
✅ Document all `# type: ignore` comments with error codes
✅ Minimize `Any` usage and document when necessary
✅ Leverage type stubs (`.pyi`) for distribution
✅ Run `mypy --strict` in CI pipeline

For questions or issues, see `TYPE_SAFETY_REPORT.md` for current status and common errors.
