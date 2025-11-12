# Example Notebooks

Hands-on examples demonstrating the Transformer Builder Colab utilities.

## Quick Start

**New to the platform?** Start here:

### 📘 [01_quick_start.ipynb](./01_quick_start.ipynb)
Train your first transformer model in 10 minutes.
- **Level**: Beginner
- **Time**: ~10 minutes
- **Hardware**: Colab free tier (T4 GPU)
- **Topics**: Basic training, text generation

## Advanced Examples

### 📗 [02_custom_architecture.ipynb](./02_custom_architecture.ipynb) *(Coming soon)*
Use your own custom model architecture from Transformer Builder.
- **Level**: Intermediate
- **Time**: ~15 minutes
- **Hardware**: Colab free/pro tier
- **Topics**: Custom models, architecture verification, adapter usage

### 📙 [03_large_scale_training.ipynb](./03_large_scale_training.ipynb) *(Coming soon)*
Train large models with checkpointing and resumption.
- **Level**: Advanced
- **Time**: ~2-4 hours
- **Hardware**: Colab Pro+ recommended
- **Topics**: Multi-GPU, checkpointing, Drive backup, resuming training

### 📕 [04_model_export.ipynb](./04_model_export.ipynb) *(Coming soon)*
Export trained models for deployment.
- **Level**: Intermediate
- **Time**: ~10 minutes
- **Hardware**: Any
- **Topics**: ONNX export, TorchScript export, model cards, benchmarking

### 📔 [05_advanced_tokenization.ipynb](./05_advanced_tokenization.ipynb) *(Coming soon)*
Train custom BPE tokenizers for any vocabulary size.
- **Level**: Advanced
- **Time**: ~20 minutes
- **Hardware**: Any (CPU fine)
- **Topics**: BPE training, character tokenizers, tokenizer validation, multilingual support

## How to Use

### In Google Colab

1. Click the notebook link
2. Click "Open in Colab" badge (if present) or File → Open notebook → GitHub
3. Run cells in order (Shift+Enter)

### Locally

```bash
# Clone repository
git clone https://github.com/matt-hans/transformer-builder-colab-templates.git
cd transformer-builder-colab-templates

# Install dependencies
pip install -r requirements-colab.txt

# Launch Jupyter
jupyter notebook examples/
```

## Notebook Structure

Each notebook follows this pattern:

1. **Setup**: Install dependencies, download utils
2. **Load Model**: Create or load transformer model
3. **Configure**: Set training parameters
4. **Train**: Run training with progress bars
5. **Evaluate**: Test the trained model
6. **Export** (if applicable): Save for production

## Common Patterns

### Quick Training

```python
from utils.training import train_model

results = train_model(
    model=your_model,
    dataset='wikitext',
    vocab_size=50257,
    max_epochs=3
)
```

### Using Presets

```python
from utils.ui import ConfigPresets

presets = ConfigPresets()
config = presets.get('small')  # or 'tiny', 'medium', 'large'

from utils.training import TrainingCoordinator
coordinator = TrainingCoordinator()
results = coordinator.train(model=your_model, **config.to_dict())
```

### Setup Wizard

```python
from utils.ui import SetupWizard

wizard = SetupWizard()
config = wizard.run(model=your_model, preset='small')
# Interactive configuration in 5 steps
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` (try 8, 4, 2)
- Reduce `max_seq_len` (try 256, 128)
- Enable gradient accumulation: `gradient_accumulation_steps=4`
- Use smaller preset: `'tiny'` instead of `'small'`

### Slow Training

- Check GPU is being used: `torch.cuda.is_available()`
- Increase batch size if memory allows
- Enable mixed precision: `precision='16'` (enabled by default)
- Use faster dataset (smaller one for testing)

### Import Errors

```python
# Reinstall dependencies
!pip install -U torch pytorch-lightning transformers datasets

# Re-download utils
!wget -q https://github.com/matt-hans/transformer-builder-colab-templates/archive/refs/heads/main.zip
!unzip -q main.zip
!mv transformer-builder-colab-templates-main/utils .
```

### Model Loading Issues

Ensure your model is a PyTorch `nn.Module` and has a `forward()` method:

```python
import torch.nn as nn

class YourModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        # Your forward pass
        return output
```

## Need Help?

- **Documentation**: See `/docs/API_REFERENCE.md`
- **Issues**: https://github.com/matt-hans/transformer-builder-colab-templates/issues
- **Discussions**: GitHub Discussions tab

## Contributing

Want to add an example? See `CONTRIBUTING.md` for guidelines.

Example notebook template:
1. Clear objective and target audience
2. Step-by-step with explanations
3. Working code (tested in Colab)
4. Expected outputs and timing
5. Troubleshooting section
