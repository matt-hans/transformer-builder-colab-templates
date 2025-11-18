# Minimal Serving Examples (FastAPI + Gradio)

This directory contains **minimal serving examples** showing how to load
exported models (TorchScript/ONNX) and serve them via FastAPI or Gradio.

> These examples are intentionally simple and are **not** production-ready.
> They are useful as starting points for building real deployment stacks.

## Installation

Serving dependencies are **optional** and not included in the main
requirements. Install them into your environment:

```bash
pip install fastapi uvicorn gradio

# Optional (for ONNX runtime serving)
pip install onnxruntime
```

You should also have models exported via `export_model` (Tier 4) into
directories like:

- `exports/lm_tiny/`
- `exports/vision_tiny/`

## FastAPI Server

File: `examples/serving/fastapi_server.py`

### Endpoints

- `POST /generate`
  - Request JSON: `{"prompt": "Hello world"}`
  - Response JSON: `{"logits": [...]}` (raw logits from the exported LM).

- `POST /predict`
  - Request JSON: `{"image": "data:image/png;base64,..."}` or just `"base64_data"`.
  - Response JSON:
    ```json
    {
      "predictions": [
        {"class": "class_0", "prob": 0.92},
        {"class": "class_1", "prob": 0.05},
        {"class": "class_2", "prob": 0.03}
      ]
    }
    ```

### Running the Server

```bash
export LM_EXPORT_DIR=exports/lm_tiny
export VISION_EXPORT_DIR=exports/vision_tiny

uvicorn examples.serving.fastapi_server:app --reload --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

## Gradio Demo

File: `examples/serving/gradio_demo.py`

This script creates a simple UI with two tabs:

- **Text LM** – textbox input, shows last-token logits (first 10 values).
- **Vision Classification** – image upload, shows top-3 predictions.

### Running the Demo

```bash
export LM_EXPORT_DIR=exports/lm_tiny
export VISION_EXPORT_DIR=exports/vision_tiny

python examples/serving/gradio_demo.py
```

Locally this opens a browser window. In Colab, `share=True` will also
provide a public URL (you can configure ngrok as needed).

### Error Handling

- If models are not exported, the demo will display a friendly message:
  - “LM model not available. Please export a model first.”
  - “Vision model not available. Please export a model first.”
- Invalid or corrupted images in FastAPI `/predict` result in:
  - HTTP 400 with `"Invalid image format"` detail.

## Colab End-to-End Snippet

In Colab you can run everything from a single notebook cell:

```python
# Install serving dependencies
!pip install fastapi uvicorn gradio

# (Optional) export a tiny LM stub via Tier 4 CLI
!python -m cli.run_tiers --config configs/example_tiers_export.json --json

# Set export directories for Gradio
import os
os.environ["LM_EXPORT_DIR"] = "exports/lm_tiny"
os.environ["VISION_EXPORT_DIR"] = "exports/vision_tiny"  # if you have a vision export

# Launch Gradio demo (prints a public URL in Colab)
!python examples/serving/gradio_demo.py
```

This will export the LM stub (if not already exported) and then start the
Gradio demo with a shareable URL.
