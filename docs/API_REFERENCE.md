# API Reference

Complete API documentation for Transformer Builder Colab utilities.

## Table of Contents

- [Installation](#installation)
- [Adapters](#adapters)
- [Tokenization](#tokenization)
- [Training](#training)
- [Export](#export)
- [UI Components](#ui-components)
- [Testing](#testing)

---

## Installation

```python
# Install from repository
!pip install -q torch pytorch-lightning transformers datasets tokenizers

# Download utils
!wget -q https://github.com/matt-hans/transformer-builder-colab-templates/archive/refs/heads/main.zip
!unzip -q main.zip
!mv transformer-builder-colab-templates-main/utils .
```

---

## Adapters

### ModelSignatureInspector

Analyzes model forward() signatures to detect complexity.

```python
from utils.adapters import ModelSignatureInspector

inspector = ModelSignatureInspector(model)

# Get parameter names
params = inspector.get_parameters()  # ['input_ids', 'mhsa_0_output', ...]

# Check if complex
is_complex = inspector.requires_intermediate_outputs()  # True/False

# Get signature info
info = inspector.get_signature_info()
```

**Methods**:
- `get_parameters() -> List[str]`: Return parameter names
- `requires_intermediate_outputs() -> bool`: Check if needs intermediate outputs
- `get_signature_info() -> Dict[str, Any]`: Get full signature details

---

### ComputationalGraphExecutor

Executes models with complex signatures requiring intermediate outputs.

```python
from utils.adapters import ComputationalGraphExecutor

executor = ComputationalGraphExecutor(model, inspector)

# Execute with automatic dependency resolution
output = executor.forward(input_ids, attention_mask)
```

**Methods**:
- `forward(input_ids, attention_mask=None) -> torch.Tensor`: Execute model
- `get_layer_map() -> Dict[str, nn.Module]`: Get layer mapping

---

### UniversalModelAdapter

PyTorch Lightning wrapper for ANY transformer architecture.

```python
from utils.adapters import UniversalModelAdapter

adapter = UniversalModelAdapter(
    model=your_model,
    learning_rate=1e-4,
    vocab_size=50257,
    warmup_steps=500
)

# Use with Lightning Trainer
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=3)
trainer.fit(adapter, datamodule)

# Generate text
text = adapter.generate(
    input_ids=start_tokens,
    max_length=100,
    temperature=0.8
)
```

**Parameters**:
- `model` (nn.Module): PyTorch model
- `learning_rate` (float): Learning rate (default: 1e-4)
- `vocab_size` (int): Vocabulary size
- `warmup_steps` (int): LR warmup steps (default: 0)
- `weight_decay` (float): AdamW weight decay (default: 0.01)

**Methods**:
- `forward(input_ids, attention_mask, labels) -> Dict`: Training forward pass
- `generate(input_ids, max_length, temperature) -> torch.Tensor`: Text generation
- `training_step(batch, batch_idx) -> torch.Tensor`: Lightning training step
- `validation_step(batch, batch_idx)`: Lightning validation step
- `configure_optimizers() -> Tuple`: Optimizer and scheduler

---

## Tokenization

### AdaptiveTokenizer

4-tier adaptive tokenization supporting ANY vocabulary size.

```python
from utils.tokenization import AdaptiveTokenizer

# Create or load tokenizer
tokenizer = AdaptiveTokenizer.load_or_create(
    vocab_size=50257,
    dataset=your_dataset,
    cache_dir='./tokenizers'
)

# Encode text
encoded = tokenizer.encode(
    "Hello world!",
    max_length=512,
    padding='max_length'
)

# Decode
text = tokenizer.decode(encoded['input_ids'])
```

**Class Methods**:
- `load_or_create(vocab_size, dataset, cache_dir) -> Tokenizer`: Get tokenizer
- `detect_strategy(vocab_size, dataset_size) -> str`: Determine best strategy

**Strategies**:
1. **Pretrained**: Exact vocab match (40+ models)
2. **Train BPE**: Custom BPE for 5K-100K vocab
3. **Character**: Universal fallback for any size
4. **User Upload**: Custom tokenizer (optional)

---

### FastBPETrainer

Train custom BPE tokenizers efficiently.

```python
from utils.tokenization import FastBPETrainer, BPETrainerConfig

config = BPETrainerConfig(
    vocab_size=25000,
    min_frequency=2,
    special_tokens=['<pad>', '<unk>', '<s>', '</s>']
)

trainer = FastBPETrainer(config)
tokenizer = trainer.train_on_dataset(
    texts=dataset['text'],
    show_progress=True
)

# Save
tokenizer.save('my_tokenizer.json')
```

**Parameters**:
- `vocab_size` (int): Target vocabulary size
- `min_frequency` (int): Minimum token frequency (default: 2)
- `special_tokens` (List[str]): Special tokens to add

---

### CharacterLevelTokenizer

Universal fallback tokenizer for any vocabulary size.

```python
from utils.tokenization import CharacterLevelTokenizer

tokenizer = CharacterLevelTokenizer(
    vocab_size=100000,
    special_tokens=['<pad>', '<unk>', '<s>', '</s>']
)

# Encode/decode like HuggingFace tokenizers
encoded = tokenizer.encode("Hello 世界!", max_length=512)
text = tokenizer.decode(encoded['input_ids'])
```

**Parameters**:
- `vocab_size` (int): Vocabulary size (100 to 500,000+)
- `special_tokens` (List[str]): Special tokens

---

### TokenizerValidator

Validate tokenizers meet requirements.

```python
from utils.tokenization import TokenizerValidator

# Strict validation (raises exception)
TokenizerValidator.validate(
    tokenizer,
    expected_vocab_size=50257,
    strict=True
)

# Non-strict (returns bool)
is_valid = TokenizerValidator.validate(
    tokenizer,
    expected_vocab_size=50257,
    strict=False
)
```

**Checks**:
1. Vocabulary size matches
2. Special tokens present
3. Encode/decode round-trip works
4. Token IDs in valid range

---

### AdaptiveTokenizerDataModule

PyTorch Lightning DataModule with automatic tokenization.

```python
from utils.tokenization import AdaptiveTokenizerDataModule

datamodule = AdaptiveTokenizerDataModule(
    dataset=hf_dataset,
    tokenizer=tokenizer,
    batch_size=16,
    max_length=512,
    val_split=0.1
)

# Use with trainer
trainer.fit(model, datamodule)
```

**Parameters**:
- `dataset` (Dataset): HuggingFace Dataset
- `tokenizer` (Tokenizer): Any HuggingFace-compatible tokenizer
- `batch_size` (int): Training batch size
- `max_length` (int): Maximum sequence length
- `val_split` (float): Validation split ratio
- `num_workers` (int): DataLoader workers

---

## Training

### train_model() - Simple API

One-function training for quick experiments.

```python
from utils.training import train_model

results = train_model(
    model=your_model,
    dataset='wikitext',
    vocab_size=50257,
    max_epochs=3,
    batch_size=16,
    learning_rate=1e-4
)

print(f"Best checkpoint: {results['best_model_path']}")
print(f"Final metrics: {results['final_metrics']}")
```

**Parameters**:
- `model` (nn.Module): Model to train
- `dataset` (str | Dataset): HuggingFace dataset name or Dataset object
- `vocab_size` (int): Vocabulary size
- `max_epochs` (int): Training epochs
- `batch_size` (int): Batch size (default: 16)
- `learning_rate` (float): Learning rate (default: 1e-4)
- `**kwargs`: Additional arguments passed to TrainingCoordinator

**Returns**: `Dict[str, Any]` with keys:
- `best_model_path`: Path to best checkpoint
- `final_metrics`: Final validation metrics
- `trainer`: Lightning Trainer instance
- `model`: Trained UniversalModelAdapter
- `tokenizer`: Used tokenizer

---

### TrainingCoordinator - Advanced API

Full control over training pipeline.

```python
from utils.training import TrainingCoordinator

coordinator = TrainingCoordinator(
    output_dir='./training_output',
    use_gpu=True,
    precision='16',
    gradient_clip_val=1.0
)

results = coordinator.train(
    model=your_model,
    dataset='wikitext',
    config_name='wikitext-2-raw-v1',
    vocab_size=50257,
    batch_size=32,
    max_length=512,
    learning_rate=5e-4,
    max_epochs=10,
    val_split=0.1,
    accumulate_grad_batches=2,
    early_stopping_patience=3,
    save_top_k=3,
    resume_from_checkpoint=None
)
```

**Constructor Parameters**:
- `output_dir` (str): Base directory for outputs
- `use_gpu` (bool): Use GPU if available (default: True)
- `precision` (str): Training precision ('32', '16', 'bf16')
- `gradient_clip_val` (float): Gradient clipping value

**train() Parameters**:
- `model`: PyTorch model
- `dataset`: HuggingFace dataset name or Dataset object
- `dataset_path`: Path to local file (alternative to dataset)
- `config_name`: HuggingFace dataset config
- `vocab_size`: Vocabulary size
- `batch_size`: Training batch size
- `max_length`: Maximum sequence length
- `learning_rate`: Learning rate
- `max_epochs`: Maximum epochs
- `val_split`: Validation split fraction
- `accumulate_grad_batches`: Gradient accumulation steps
- `early_stopping_patience`: Early stopping patience (None to disable)
- `save_top_k`: Number of best checkpoints to keep
- `tokenizer`: Pre-created tokenizer (optional)
- `datamodule`: Pre-created datamodule (optional)
- `resume_from_checkpoint`: Checkpoint path to resume from
- `seed`: Random seed

**Methods**:
- `train(**kwargs) -> Dict`: Full training pipeline
- `quick_train(model, dataset, ...) -> Dict`: Quick training with defaults
- `resume_training(checkpoint_path, ...) -> Dict`: Resume from checkpoint

---

### DatasetLoader

Load datasets from multiple sources.

```python
from utils.training import DatasetLoader

loader = DatasetLoader(
    preprocessing=True,
    min_length=10,
    max_length=None,
    remove_duplicates=False
)

# HuggingFace
dataset = loader.load_huggingface('wikitext', 'wikitext-2-raw-v1')

# Local file
dataset = loader.load_local_file('data.txt', text_column='text')

# Google Drive (Colab)
dataset = loader.load_from_drive('/content/drive/MyDrive/data.txt')

# Statistics
stats = loader.get_statistics(dataset)
loader.print_statistics(dataset)
loader.preview_samples(dataset, num_samples=3)
```

**Methods**:
- `load_huggingface(dataset_name, config_name, split) -> Dataset`
- `load_local_file(file_path, file_format, text_column) -> Dataset`
- `load_from_drive(drive_path, text_column) -> Dataset`
- `get_statistics(dataset) -> Dict[str, Any]`
- `print_statistics(dataset)`
- `preview_samples(dataset, num_samples)`

---

### CheckpointManager

Manage training checkpoints.

```python
from utils.training import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='./checkpoints',
    save_top_k=3,
    monitor='val_loss',
    mode='min',
    drive_backup=True,
    drive_backup_path='MyDrive/checkpoints'
)

# Get Lightning callback
callback = manager.get_callback()
trainer = pl.Trainer(callbacks=[callback])

# Load checkpoint
checkpoint = manager.load_checkpoint()
model = manager.load_model_from_checkpoint(UniversalModelAdapter)

# Manage checkpoints
checkpoints = manager.list_checkpoints()
manager.cleanup_old_checkpoints(keep_top_k=3)
manager.print_checkpoint_info()
```

**Methods**:
- `get_callback() -> ModelCheckpoint`: Lightning callback
- `get_backup_callback() -> Optional[DriveBackupCallback]`: Drive backup
- `load_checkpoint(checkpoint_path) -> Dict`: Load checkpoint
- `load_model_from_checkpoint(model_class, checkpoint_path) -> nn.Module`: Load model
- `get_best_checkpoint_path() -> Optional[str]`: Path to best checkpoint
- `list_checkpoints(sort_by) -> List[str]`: List all checkpoints
- `cleanup_old_checkpoints(keep_top_k)`: Remove old checkpoints
- `print_checkpoint_info()`: Print checkpoint status

---

## Export

### ONNXExporter

Export models to ONNX format.

```python
from utils.training import ONNXExporter

exporter = ONNXExporter(
    opset_version=14,
    optimize=True,
    validate=True,
    benchmark=True
)

result = exporter.export(
    model=trained_model,
    output_path='model.onnx',
    vocab_size=50257,
    max_seq_len=512,
    dynamic_axes=True
)

print(f"Exported: {result['output_path']}")
print(f"Size: {result['file_size_mb']:.2f} MB")
print(f"Speedup: {result['benchmark']['speedup']:.2f}x")
```

**Features**:
- Dynamic batch/sequence dimensions
- ONNX optimization passes
- Output validation vs PyTorch
- Inference benchmarking (2-5x CPU speedup)

---

### TorchScriptExporter

Export models to TorchScript format.

```python
from utils.training import TorchScriptExporter

exporter = TorchScriptExporter(validate=True, benchmark=True)

result = exporter.export(
    model=trained_model,
    output_path='model.pt',
    vocab_size=50257,
    mode='auto'  # 'trace', 'script', or 'auto'
)

print(f"Mode: {result['mode']}")
print(f"Speedup: {result['benchmark']['speedup']:.2f}x")
```

**Features**:
- Both tracing and scripting modes
- Automatic fallback (trace → script)
- Optimization for inference
- Benchmarking (10-20% GPU speedup)

---

### ModelCardGenerator

Generate HuggingFace-style model cards.

```python
from utils.training import ModelCardGenerator

generator = ModelCardGenerator()

card = generator.generate(
    model_name='my-gpt2-wikitext',
    model=trained_model,
    training_results=results,
    dataset_name='wikitext-2-raw-v1',
    vocab_size=50257,
    description='GPT-2 trained on WikiText',
    output_path='MODEL_CARD.md'
)
```

**Generated Sections**:
- Model details (type, parameters, vocab)
- Training data information
- Performance metrics
- Usage examples
- Limitations
- Citation

---

## UI Components

### SetupWizard

Interactive 5-step training configuration.

```python
from utils.ui import SetupWizard

wizard = SetupWizard()

# Interactive mode (Colab)
config = wizard.run(model=your_model, interactive=True, preset='small')

# Quick setup (non-interactive)
config = wizard.quick_setup(
    model=your_model,
    preset='small',
    dataset_name='wikitext'
)

# Print configuration
wizard.print_config(config)

# Validate
is_valid, errors = wizard.validate_config(config)

# Use for training
results = coordinator.train(model=your_model, **config.to_dict())
```

**Steps**:
1. Dataset selection (HuggingFace/local/Drive/upload)
2. Tokenizer configuration
3. Model verification
4. Training parameters
5. Validation and summary

---

### ConfigPresets

Pre-configured training settings.

```python
from utils.ui import ConfigPresets, PRESETS

presets = ConfigPresets()

# List available presets
presets.print_all_presets()

# Get preset
config = presets.get('small')
print(config.description)
print(config.estimated_time_hours)

# Customize preset
custom = presets.customize(
    'small',
    max_epochs=10,
    batch_size=32
)

# Get recommendation
preset_name = presets.get_recommendation(
    goal='learning',
    time_budget_hours=5.0
)
```

**Available Presets**:
- `tiny`: Debug/testing (~1 hour, ~10M params)
- `small`: Educational (~4 hours, ~125M params)
- `medium`: Production (~12 hours, ~350M params)
- `large`: Research (~48 hours, ~774M params)
- `code_generation`: Code tasks
- `chat`: Dialogue systems
- `summarization`: Text summarization

---

## Testing

### Test Functions

Validate generated models with 3-tier test suite.

```python
from utils.test_functions import (
    run_all_tier1_tests,
    run_all_tier2_tests,
    run_all_tests
)

# Tier 1: Critical validation (~1 minute)
run_all_tier1_tests(model, config)

# Tier 2: Advanced analysis (~4 minutes)
run_all_tier2_tests(model, config)

# All tiers (~120+ minutes)
run_all_tests(model, config)
```

**Tier 1 Tests** (Critical):
- Shape robustness
- Gradient flow
- Output stability
- Parameter initialization
- Memory footprint
- Inference speed

**Tier 2 Tests** (Advanced):
- Attention pattern analysis
- Feature attribution
- Input perturbation sensitivity

**Tier 3 Tests** (Training):
- Fine-tuning loop
- Hyperparameter search
- GLUE benchmarks

---

## Common Workflows

### Complete Training Pipeline

```python
# 1. Load model
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(vocab_size=50257, n_layer=6)
model = GPT2LMHeadModel(config)

# 2. Train with one function
from utils.training import train_model

results = train_model(
    model=model,
    dataset='wikitext',
    vocab_size=50257,
    max_epochs=3
)

# 3. Export to ONNX
from utils.training import ONNXExporter

exporter = ONNXExporter()
exporter.export(
    results['model'].model,
    'model.onnx',
    vocab_size=50257
)

# 4. Generate model card
from utils.training import ModelCardGenerator

generator = ModelCardGenerator()
generator.generate(
    model_name='my-model',
    model=results['model'],
    training_results=results,
    output_path='MODEL_CARD.md'
)
```

### Using Presets

```python
from utils.ui import ConfigPresets
from utils.training import TrainingCoordinator

# Get preset
presets = ConfigPresets()
config = presets.get('small')

# Train
coordinator = TrainingCoordinator()
results = coordinator.train(
    model=your_model,
    **config.to_dict()
)
```

### Interactive Setup

```python
from utils.ui import SetupWizard
from utils.training import TrainingCoordinator

# Interactive configuration
wizard = SetupWizard()
config = wizard.run(model=your_model, preset='small')

# Train with configured settings
coordinator = TrainingCoordinator()
results = coordinator.train(
    model=your_model,
    **config.to_dict()
)
```

---

## Error Handling

All functions include comprehensive error handling with helpful messages:

```python
try:
    results = train_model(model=model, dataset='invalid_dataset')
except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Training error: {e}")
```

---

## Performance Tips

### Memory Optimization

```python
# Reduce batch size
results = train_model(model=model, batch_size=8)

# Enable gradient accumulation
results = coordinator.train(
    model=model,
    batch_size=4,
    accumulate_grad_batches=4  # Effective batch size: 16
)

# Shorter sequences
results = train_model(model=model, max_length=256)
```

### Speed Optimization

```python
# Mixed precision (enabled by default)
coordinator = TrainingCoordinator(precision='16')

# More workers
datamodule = AdaptiveTokenizerDataModule(
    dataset=dataset,
    tokenizer=tokenizer,
    num_workers=4
)

# Faster dataset
results = train_model(
    model=model,
    dataset='wikitext',
    config_name='wikitext-2-raw-v1'  # Smaller than wikitext-103
)
```

---

## Version Information

**Current Version**: 2.0.0

**Compatibility**:
- Python: 3.8+
- PyTorch: 2.0+
- PyTorch Lightning: 2.0+
- Transformers: 4.30+

---

## Support

- **Documentation**: This file
- **Examples**: `/examples/` directory
- **Issues**: https://github.com/matt-hans/transformer-builder-colab-templates/issues
- **Discussions**: GitHub Discussions
