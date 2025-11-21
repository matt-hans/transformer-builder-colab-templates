# Model Export Bundle

**Task:** demo-task
**Modality:** text
**Task Type:** language_modeling
**Exported:** 2025-11-20 17:31:24

## Overview

This bundle contains a production-ready exported model with all necessary artifacts for deployment.

**Exported Formats:**
- pytorch

## Directory Structure

```
.
├── artifacts/          # Model files
│   ├── model.onnx              (ONNX format)
│   ├── model.torchscript.pt    (TorchScript format)
│   └── model.pytorch.pt        (PyTorch state dict)
├── configs/            # Configuration files
│   ├── task_spec.json          (Task specification)
│   ├── training_config.json    (Training configuration)
│   └── torchserve_config.json  (TorchServe deployment config)
├── inference.py        # Standalone inference script
├── Dockerfile          # Container deployment
├── requirements.txt    # Runtime dependencies
└── README.md           # This file
```

## Quick Start

### Local Inference

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run inference:**
```bash
python inference.py --input "Your text here" --model artifacts/model.onnx
```

## Docker Deployment

**Build container:**
```bash
docker build -t model-inference .
```

**Run container:**
```bash
docker run -p 8080:8080 model-inference
```

**Test endpoint:**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{{"text": "Your input text"}}' \
     http://localhost:8080/predict
```

## TorchServe Deployment

**Create model archive:**
```bash
torch-model-archiver \
    --model-name transformer-model \
    --version 1.0 \
    --serialized-file artifacts/model.torchscript.pt \
    --handler inference.py \
    --export-path model-store
```

**Start TorchServe:**
```bash
torchserve \
    --start \
    --model-store model-store \
    --models transformer-model=transformer-model.mar \
    --ncs
```

**Configuration:**
See `configs/torchserve_config.json` for detailed deployment settings.

## Model Information

**Task Specification:**
- Input fields: {", ".join(task_spec.input_fields)}
- Target field: {task_spec.target_field or "N/A"}
- Loss type: {task_spec.loss_type}
- Metrics: {", ".join(task_spec.metrics)}

**Input Schema:**
```json
{json.dumps(dict(task_spec.input_schema), indent=2)}
```

**Output Schema:**
```json
{json.dumps(dict(task_spec.output_schema), indent=2)}
```

## Runtime Requirements

See `requirements.txt` for complete dependency list.

**Minimum requirements:**
- Python >= 3.8
- PyTorch >= 2.0 (for TorchScript)
- ONNX Runtime >= 1.15 (for ONNX)

## Performance

**Inference Speed:**
- TorchScript: ~1.1-1.2x faster than PyTorch
- ONNX: ~1.5-2.5x faster than PyTorch (CPU)

**Model Size:**
Check `artifacts/` directory for file sizes.

## Troubleshooting

**ONNX inference fails:**
- Ensure onnxruntime is installed: `pip install onnxruntime`
- For GPU: `pip install onnxruntime-gpu`

**TorchScript shape errors:**
- Verify input dimensions match model expectations
- Check `configs/task_spec.json` for input schema

**Import errors:**
- Install all requirements: `pip install -r requirements.txt`

## License

Model exported from Transformer Builder training pipeline.

## Citation

If using this model in research, please cite:

```bibtex
@misc{{transformer_builder_{task_spec.name.replace("-", "_")},
  title={{{{Transformer Model: {task_spec.name}}}}},
  year={{{datetime.now().year}}},
  note={{Generated using Transformer Builder Training Pipeline v3.5}}
}}
```

## Support

For issues or questions:
- Check task_spec.json for model configuration
- Review training_config.json for training details
- Consult Transformer Builder documentation
