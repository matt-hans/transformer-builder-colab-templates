# Transformer Builder - Colab Testing Templates

Advanced testing and training infrastructure for transformer models built with [Transformer Builder](https://transformer-builder.com).

## Quick Start

1. Build a transformer in [Transformer Builder](https://transformer-builder.com)
2. Click "Open in Colab" in the export panel
3. The notebook will automatically load your model and run validation tests

No setup required - everything is pre-configured!

## What's Included

### Tier 1: Critical Validation (~1 minute)
- ✅ Multi-input shape verification across edge cases
- ✅ Gradient flow analysis (detect vanishing/exploding gradients)
- ✅ Numerical stability checks (NaN/Inf detection, output distribution)
- ✅ Parameter initialization validation
- ✅ Memory footprint profiling
- ✅ Inference speed benchmarks

### Tier 2: Advanced Analysis (~4 minutes)
- 🔬 Attention pattern analysis (entropy, sparsity, pattern classification)
- 🔬 Feature attribution using Integrated Gradients
- 🔬 Input perturbation sensitivity testing
- 🔬 Adversarial token search
- 🔬 Effective rank & capacity utilization analysis

### Tier 3: Training & Fine-Tuning (5-120 minutes)
- 🚀 Training loop with comprehensive diagnostics
- 🚀 Hyperparameter optimization using Optuna
- 🚀 GLUE benchmark evaluation

## Repository Structure

```
transformer-builder-colab-templates/
├── template.ipynb              # Main template (auto-populated from URL)
├── examples/
│   ├── example_distilgpt2.ipynb      # Pre-filled decoder example
│   └── example_bert_encoder.ipynb    # Pre-filled encoder example
├── utils/
│   └── test_functions.py       # Importable test utilities
└── README.md
```

## Manual Usage

If you have model code outside Transformer Builder:

1. Open `template.ipynb` in Colab
2. Modify Cell 3 to include your model code
3. Update config in Cell 4
4. Run all cells

## Requirements

- Google account (Colab free tier is sufficient)
- Generated model must be a PyTorch `nn.Module`

## Examples

See `examples/` directory for pre-populated notebooks demonstrating common architectures.

## Support

Issues? Report at [transformer-builder/issues](https://github.com/your-org/transformer-builder/issues)

## License

MIT License - see LICENSE file
