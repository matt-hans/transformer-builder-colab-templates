# Data Loading Quick Reference Card

Quick reference for the data loading engine (`utils/training/engine/data`).

## Common Use Cases

### 1. Basic Training Setup (Text)

```python
from utils.training.engine.data import UniversalDataModule
from utils.training.task_spec import TaskSpec
from datasets import load_dataset

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Create data module (auto-selects text collator)
data_module = UniversalDataModule(
    train_data=dataset,
    task_spec=TaskSpec.lm_tiny(),
    tokenizer=tokenizer,
    batch_size=32,
    val_split=0.1,
    seed=42
)

# Get loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Train
for batch in train_loader:
    loss = model(batch['input_ids'], labels=batch['labels'])
    # ... backward pass ...
```

### 2. Vision Task Setup

```python
from utils.training.engine.data import UniversalDataModule
from utils.training.task_spec import TaskSpec
from datasets import load_dataset

# Load vision dataset
dataset = load_dataset('cifar10', split='train')

# Create data module (auto-selects vision collator with ImageNet normalization)
data_module = UniversalDataModule(
    train_data=dataset,
    task_spec=TaskSpec.vision_tiny(),
    batch_size=64,
    val_split=0.2,
    num_workers=4,  # Use more workers for vision
    seed=42
)

train_loader = data_module.train_dataloader()
```

### 3. Custom Collator

```python
from utils.training.engine.data import CollatorRegistry, DataLoaderFactory, DataLoaderConfig

# Register custom collator
registry = CollatorRegistry.get_instance()

@registry.register('custom', modality='text')
def create_custom_collator(tokenizer=None):
    def collate_fn(batch):
        # Your custom collation logic
        return {'custom_field': ...}
    return collate_fn

# Use custom collator
factory = DataLoaderFactory()
config = DataLoaderConfig(
    batch_size=32,
    collate_fn=registry.get_collator(collator_name='custom', tokenizer=tokenizer)
)
loader = factory.create_dataloader(dataset, config)
```

### 4. Reproducible Experiments

```python
from utils.training.training_config import TrainingConfig
from utils.training.seed_manager import set_random_seed
from utils.training.engine.data import UniversalDataModule

# Configure experiment
config = TrainingConfig(
    batch_size=32,
    random_seed=42,
    deterministic=True  # Bit-exact reproducibility
)

# Set global seed
set_random_seed(config.random_seed, config.deterministic)

# Create data module with same seed
data_module = UniversalDataModule(
    train_data=dataset,
    batch_size=config.batch_size,
    seed=config.random_seed,
    task_spec=task_spec
)

# Identical batch order across runs with same seed
```

### 5. Performance Optimization

```python
from utils.training.engine.data import UniversalDataModule
import torch

# Auto-optimized for GPU
data_module = UniversalDataModule(
    train_data=dataset,
    batch_size=64,
    num_workers=4 if torch.cuda.is_available() else 0,
    # pin_memory and prefetch_factor auto-detected
    seed=42
)

# Expected improvements:
# - pin_memory=True: +10-20% GPU throughput
# - 4 workers: +20-30% async loading
# - prefetch_factor=2: +5-10% reduced waiting
```

## API Quick Reference

### DataModuleProtocol

```python
class DataModuleProtocol(Protocol):
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> Optional[DataLoader]: ...
```

### CollatorRegistry

```python
registry = CollatorRegistry.get_instance()

# Auto-select from TaskSpec
collator = registry.get_collator(task_spec=task_spec, tokenizer=tokenizer)

# Explicit selection
collator = registry.get_collator(collator_name='vision', normalize=True)

# Register custom
@registry.register('name', modality='text')
def factory(**kwargs): ...
```

### DataLoaderConfig

```python
@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: Optional[bool] = None  # Auto-detect
    prefetch_factor: Optional[int] = 2
    persistent_workers: Optional[bool] = None  # Auto-detect
    drop_last: bool = False
    seed: int = 42
    collate_fn: Optional[Callable] = None
```

### DataLoaderFactory

```python
factory = DataLoaderFactory()

config = DataLoaderConfig(batch_size=32, shuffle=True, seed=42)
loader = factory.create_dataloader(
    dataset=dataset,
    config=config,
    task_spec=task_spec,  # For auto-collator
    tokenizer=tokenizer  # For text tasks
)
```

### UniversalDataModule

```python
data_module = UniversalDataModule(
    train_data: Union[Dataset, HFDataset, List[Tensor]],
    val_data: Optional[...] = None,  # Auto-split if None
    task_spec: Optional[TaskSpec] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 32,
    val_split: float = 0.2,  # Fraction for val
    num_workers: int = 2,
    seed: int = 42
)

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

## Performance Tuning

| Scenario | Configuration | Expected Speedup |
|----------|--------------|------------------|
| CPU training | `num_workers=0` | Baseline (no multiprocessing overhead) |
| GPU, fast SSD | `num_workers=2, pin_memory=True` | +15-25% |
| GPU, slow HDD | `num_workers=4, prefetch_factor=4` | +25-35% |
| Multi-epoch | `persistent_workers=True` | +5-10% (reduced startup) |

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Non-reproducible batches | Use `seed` parameter in DataLoaderConfig |
| Multiprocessing pickle error | Use `num_workers=0` or module-level classes |
| High DataLoader overhead | Reduce `num_workers` or optimize collator |
| Memory issues | Reduce `batch_size` or `prefetch_factor` |
| Slow iteration | Increase `num_workers`, enable `pin_memory` |

## Testing

```bash
# Run all data loading tests
pytest tests/training/engine/test_data.py -v

# Run specific test
pytest tests/training/engine/test_data.py::TestCollatorRegistry -v

# Performance benchmark
pytest tests/training/engine/test_data.py::TestPerformance -v -s
```

## Integration Checklist

- [ ] Import from `utils.training.engine.data`
- [ ] Create `TaskSpec` for collator auto-selection
- [ ] Set `seed` for reproducibility
- [ ] Configure `num_workers` based on hardware
- [ ] Test with small batch to verify collation
- [ ] Benchmark DataLoader overhead (<5%)
- [ ] Verify batch order reproducibility

## Related Files

- **Implementation:** `utils/training/engine/data.py`
- **Tests:** `tests/training/engine/test_data.py`
- **Guide:** `docs/DATA_LOADING_GUIDE.md`
- **Task Summary:** `docs/P1-1_TASK_SUMMARY.md`

## Support

For detailed documentation, see [DATA_LOADING_GUIDE.md](DATA_LOADING_GUIDE.md).

For issues or questions, check test cases in `tests/training/engine/test_data.py` for working examples.
