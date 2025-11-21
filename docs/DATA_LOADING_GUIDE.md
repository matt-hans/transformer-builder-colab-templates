# Data Loading and Collation Guide

This guide covers the data loading and collation system in the training engine (Phase 1, Task P1-1).

## Overview

The data loading engine provides:
- **Protocol-based interfaces** for data modules (framework-agnostic)
- **Registry-based collator system** with auto-selection from TaskSpec
- **Worker seeding** for reproducibility
- **Performance optimizations** (pin_memory, prefetch, persistent_workers)
- **Support for multiple dataset types** (HuggingFace, PyTorch, List[Tensor])

## Core Components

### 1. DataModuleProtocol

A Protocol class defining the interface for data modules. Any class implementing `train_dataloader()` and `val_dataloader()` methods satisfies this protocol.

```python
from utils.training.engine.data import DataModuleProtocol
from torch.utils.data import DataLoader

class MyDataModule:
    def train_dataloader(self) -> DataLoader:
        return DataLoader(train_dataset, batch_size=32)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(val_dataset, batch_size=32)

# Automatically satisfies protocol
dm: DataModuleProtocol = MyDataModule()
```

**Benefits:**
- Framework-agnostic (no PyTorch Lightning dependency)
- Type-safe duck typing
- Easy to test and mock

### 2. CollatorRegistry

A registry system for task-specific data collators with auto-selection based on TaskSpec modality.

#### Built-in Collators

- **TextCollator**: Dynamic padding, attention masks, causal/masked LM support
- **VisionCollator**: Normalization (ImageNet/CIFAR-10/custom), RGB/grayscale support
- **DefaultCollator**: Fallback for unsupported modalities

#### Usage Examples

**Auto-selection from TaskSpec:**
```python
from utils.training.engine.data import CollatorRegistry
from utils.training.task_spec import TaskSpec

registry = CollatorRegistry.get_instance()

# Vision task - automatically selects VisionDataCollator
task_spec = TaskSpec.vision_tiny()
collator = registry.get_collator(task_spec=task_spec)

# Text task - automatically selects LanguageModelingDataCollator
task_spec = TaskSpec.lm_tiny()
collator = registry.get_collator(task_spec=task_spec, tokenizer=tokenizer)
```

**Explicit collator selection:**
```python
# Use specific collator by name
collator = registry.get_collator(
    collator_name='vision',
    normalize=True,
    mean=[0.5, 0.5, 0.5],  # CIFAR-10 normalization
    std=[0.5, 0.5, 0.5]
)
```

### 3. Custom Collator Registration

You can register custom collators for domain-specific tasks:

```python
from utils.training.engine.data import CollatorRegistry

registry = CollatorRegistry.get_instance()

# Register custom audio collator
@registry.register('audio', modality='audio', description='Audio collator with spectrogram')
def create_audio_collator(task_spec=None, sample_rate=16000, n_fft=512):
    """Create audio collator for speech recognition."""
    class AudioCollator:
        def __init__(self, sample_rate, n_fft):
            self.sample_rate = sample_rate
            self.n_fft = n_fft

        def __call__(self, batch):
            import torch
            import torchaudio.transforms as T

            # Stack audio waveforms
            waveforms = torch.stack([item['waveform'] for item in batch])

            # Convert to spectrogram
            spectrogram_transform = T.Spectrogram(n_fft=self.n_fft)
            spectrograms = spectrogram_transform(waveforms)

            # Stack labels
            labels = torch.tensor([item['label'] for item in batch])

            return {'spectrograms': spectrograms, 'labels': labels}

    return AudioCollator(sample_rate, n_fft)

# Use custom collator
audio_task_spec = TaskSpec(
    name="speech_recognition",
    modality="audio",
    task_type="classification",
    # ... other fields
)
collator = registry.get_collator(task_spec=audio_task_spec, sample_rate=22050)
```

**Advanced Example: Multi-modal Collator**

```python
@registry.register('multimodal', modality='vision', description='Vision-language collator')
def create_multimodal_collator(task_spec=None, tokenizer=None, normalize=True):
    """Create collator for vision-language tasks (e.g., image captioning)."""
    from utils.tokenization.data_collator import VisionDataCollator, LanguageModelingDataCollator

    vision_collator = VisionDataCollator(normalize=normalize)
    text_collator = LanguageModelingDataCollator(tokenizer=tokenizer)

    def multimodal_collate_fn(batch):
        # Separate vision and text components
        vision_batch = [{'pixel_values': item['pixel_values']} for item in batch]
        text_batch = [{'input_ids': item['input_ids']} for item in batch]

        # Apply collators
        vision_out = vision_collator(vision_batch)
        text_out = text_collator(text_batch)

        # Merge outputs
        return {
            'pixel_values': vision_out['pixel_values'],
            'input_ids': text_out['input_ids'],
            'attention_mask': text_out['attention_mask'],
            'labels': text_out['labels']
        }

    return multimodal_collate_fn
```

### 4. DataLoaderFactory

Factory for creating optimized DataLoaders with reproducibility guarantees.

#### Basic Usage

```python
from utils.training.engine.data import DataLoaderFactory, DataLoaderConfig
from torch.utils.data import TensorDataset
import torch

factory = DataLoaderFactory()

# Create dataset
dataset = TensorDataset(torch.randn(100, 32))

# Configure DataLoader
config = DataLoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,  # Auto-detected if None
    prefetch_factor=2,
    seed=42  # Reproducible shuffling
)

# Create DataLoader
loader = factory.create_dataloader(dataset, config)

# Iterate
for batch in loader:
    # Training step
    pass
```

#### Automatic GPU Optimizations

```python
# Factory auto-detects CUDA and optimizes accordingly
config = DataLoaderConfig(
    batch_size=32,
    pin_memory=None,  # Auto: True if CUDA available, False otherwise
    prefetch_factor=2,  # Pre-load 2 batches per worker
    persistent_workers=None  # Auto: True if num_workers > 0
)

loader = factory.create_dataloader(dataset, config)

# On GPU: pin_memory=True, prefetch_factor=2
# On CPU: pin_memory=False, prefetch_factor=None
```

#### Reproducibility

```python
# Same seed guarantees identical batch order
config = DataLoaderConfig(batch_size=32, shuffle=True, seed=42, num_workers=0)

loader1 = factory.create_dataloader(dataset, config)
loader2 = factory.create_dataloader(dataset, config)

# Batches will be identical
batch1 = next(iter(loader1))
batch2 = next(iter(loader2))
assert torch.equal(batch1[0], batch2[0])  # True
```

### 5. UniversalDataModule

Unified data module supporting multiple dataset types with automatic collator selection.

#### From HuggingFace Dataset

```python
from utils.training.engine.data import UniversalDataModule
from utils.training.task_spec import TaskSpec
from datasets import load_dataset

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Create data module
data_module = UniversalDataModule(
    train_data=dataset,
    task_spec=TaskSpec.lm_tiny(),
    tokenizer=tokenizer,
    batch_size=32,
    val_split=0.1,  # 10% validation split
    num_workers=2,
    seed=42
)

# Get DataLoaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Use in training loop
for batch in train_loader:
    # batch automatically collated with text collator
    input_ids = batch['input_ids']
    labels = batch['labels']
```

#### From PyTorch Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return {
            'pixel_values': torch.randn(3, 224, 224),
            'labels': idx % 10
        }

dataset = MyDataset()

data_module = UniversalDataModule(
    train_data=dataset,
    task_spec=TaskSpec.vision_tiny(),
    batch_size=32,
    val_split=0.2,
    seed=42
)

# Automatically uses VisionDataCollator
train_loader = data_module.train_dataloader()
```

#### With External Validation Set

```python
train_dataset = load_dataset('cifar10', split='train')
val_dataset = load_dataset('cifar10', split='test')

data_module = UniversalDataModule(
    train_data=train_dataset,
    val_data=val_dataset,  # External val set
    task_spec=TaskSpec.vision_tiny(),
    batch_size=64,
    val_split=0.0,  # Ignored when val_data provided
    seed=42
)
```

#### Legacy List[Tensor] Support

```python
# Backward compatibility with legacy code
import torch

train_data = [torch.randint(0, 1000, (128,)) for _ in range(500)]

data_module = UniversalDataModule(
    train_data=train_data,  # List[Tensor] automatically converted
    batch_size=32,
    val_split=0.2,
    seed=42
)
```

## Performance Optimization

### Worker Processes

```python
# CPU: Use 0 workers (multiprocessing overhead not worth it)
config = DataLoaderConfig(
    batch_size=32,
    num_workers=0 if not torch.cuda.is_available() else 2
)

# GPU: Use 2-4 workers for async loading
config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,
    pin_memory=True,  # Faster CPU->GPU transfer
    prefetch_factor=2,  # Pre-load 2 batches
    persistent_workers=True  # Keep workers alive between epochs
)
```

### Pin Memory

Pin memory enables faster data transfer from CPU to GPU:

```python
# Automatic (recommended)
config = DataLoaderConfig(pin_memory=None)  # Auto-detect CUDA

# Manual
config = DataLoaderConfig(pin_memory=torch.cuda.is_available())

# Expected speedup: 10-20% on GPU-bound workloads
```

### Prefetch Factor

Prefetch factor controls how many batches to pre-load:

```python
# Conservative (default)
config = DataLoaderConfig(num_workers=2, prefetch_factor=2)

# Aggressive (more memory, less waiting)
config = DataLoaderConfig(num_workers=4, prefetch_factor=4)

# Note: prefetch_factor only applies when num_workers > 0
```

## Integration Examples

### With TrainingConfig

```python
from utils.training.training_config import TrainingConfig
from utils.training.engine.data import UniversalDataModule

# Define training config
training_config = TrainingConfig(
    batch_size=32,
    epochs=10,
    learning_rate=5e-5,
    random_seed=42
)

# Create data module with config parameters
data_module = UniversalDataModule(
    train_data=dataset,
    task_spec=task_spec,
    tokenizer=tokenizer,
    batch_size=training_config.batch_size,
    seed=training_config.random_seed,
    num_workers=2
)
```

### With CheckpointManager

The CheckpointManager can save DataLoader state for resuming:

```python
from utils.training.engine.checkpoint import CheckpointManager

# Save checkpoint
manager = CheckpointManager(checkpoint_dir='./checkpoints')
path = manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=5,
    metrics={'val_loss': 0.38},
    custom_state={
        'data_module_seed': data_module.seed,
        'epoch_step': epoch * len(train_loader) + step
    }
)

# Resume
state = manager.load(path)
# Recreate data module with same seed for reproducible resume
data_module = UniversalDataModule(
    train_data=dataset,
    seed=state['custom_state']['data_module_seed'],
    # ... other params
)
```

### With PyTorch Lightning (Optional)

The Protocol-based design makes integration with Lightning trivial:

```python
import pytorch_lightning as pl

class MyLightningModule(pl.LightningModule):
    def __init__(self, data_module: DataModuleProtocol):
        super().__init__()
        self.data_module = data_module

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

# Works with both UniversalDataModule and Lightning's LightningDataModule
module = MyLightningModule(data_module)
trainer = pl.Trainer()
trainer.fit(module)
```

## Best Practices

### 1. Always Use Seeded DataLoaders

```python
# Good: Reproducible
config = DataLoaderConfig(shuffle=True, seed=42)

# Bad: Non-reproducible
config = DataLoaderConfig(shuffle=True)  # Random seed each run
```

### 2. Use TaskSpec for Auto-Collator Selection

```python
# Good: Auto-selects correct collator
data_module = UniversalDataModule(
    train_data=dataset,
    task_spec=task_spec,  # Collator auto-selected
    tokenizer=tokenizer
)

# Acceptable: Explicit collator (for custom requirements)
factory = DataLoaderFactory()
config = DataLoaderConfig(collate_fn=my_custom_collator)
```

### 3. Optimize Workers for Your Hardware

```python
# CPU: No workers (avoids multiprocessing overhead)
config = DataLoaderConfig(num_workers=0)

# GPU with fast SSD: 2-4 workers
config = DataLoaderConfig(num_workers=2)

# GPU with slow HDD: 4-8 workers (compensate for I/O)
config = DataLoaderConfig(num_workers=8)

# Rule of thumb: num_workers = min(4, num_cpus // 2)
```

### 4. Profile DataLoader Overhead

```python
import time

# Measure DataLoader iteration time
start = time.time()
for batch in train_loader:
    pass
dataloader_time = time.time() - start

# Measure total training time
start = time.time()
for batch in train_loader:
    # ... training step ...
    time.sleep(0.01)  # Simulated step
total_time = time.time() - start

overhead_ratio = dataloader_time / total_time
print(f"DataLoader overhead: {overhead_ratio:.1%}")

# Target: <2% overhead for efficient training
```

### 5. Use Persistent Workers for Multi-Epoch Training

```python
# Single epoch: Persistent workers add overhead
config = DataLoaderConfig(num_workers=2, persistent_workers=False)

# Multi-epoch: Persistent workers reduce startup cost
config = DataLoaderConfig(num_workers=2, persistent_workers=True)

# Auto-detect (recommended)
config = DataLoaderConfig(num_workers=2, persistent_workers=None)
```

## Testing

Run tests to verify your data module implementation:

```bash
# Run all data loading tests
pytest tests/training/engine/test_data.py -v

# Run specific test class
pytest tests/training/engine/test_data.py::TestCollatorRegistry -v

# Run with coverage
pytest tests/training/engine/test_data.py --cov=utils.training.engine.data
```

## Performance Benchmarks

Expected performance characteristics:

| Configuration | DataLoader Overhead | Throughput Improvement |
|--------------|---------------------|------------------------|
| CPU, 0 workers | <1% | Baseline |
| GPU, pin_memory=True | <2% | +10-20% |
| GPU, 2 workers | <2% | +15-25% |
| GPU, 4 workers + prefetch=2 | <2% | +20-30% |

Note: Actual speedups depend on:
- Dataset size and complexity
- Collator computation cost
- Storage speed (SSD vs HDD)
- Training step duration

## Troubleshooting

### Issue: DataLoader Hangs with num_workers > 0

**Cause:** Multiprocessing pickle errors with local classes or lambda functions

**Solution:** Use `num_workers=0` or move collator to module scope

```python
# Bad: Local class not picklable
class LocalDataset(Dataset):
    pass

# Good: Module-level class
# (Move to top of file)
class MyDataset(Dataset):
    pass
```

### Issue: Non-Reproducible Batch Order

**Cause:** Missing seed or generator in DataLoader

**Solution:** Always use `DataLoaderFactory` or explicit seed

```python
# Bad: No seed
loader = DataLoader(dataset, shuffle=True)

# Good: With seed
config = DataLoaderConfig(shuffle=True, seed=42)
loader = factory.create_dataloader(dataset, config)
```

### Issue: High DataLoader Overhead (>5%)

**Cause:** Too many workers or slow collator

**Solution:** Profile and reduce workers or optimize collator

```python
# Profile collator
import time

batch = [dataset[i] for i in range(32)]
start = time.time()
result = collator(batch)
collator_time = time.time() - start
print(f"Collator time: {collator_time:.3f}s per batch")

# Reduce workers if overhead high
config = DataLoaderConfig(num_workers=1)  # Down from 4
```

## API Reference

See docstrings in `utils/training/engine/data.py` for detailed API documentation:

```python
# Get help on any component
from utils.training.engine.data import UniversalDataModule
help(UniversalDataModule)
```

## Related Documentation

- [Training Pipeline Overview](TRAINING_PIPELINE.md)
- [Task Specification Guide](TASK_SPEC.md)
- [Reproducibility Best Practices](REPRODUCIBILITY.md)
- [Performance Optimization Guide](PERFORMANCE.md)
