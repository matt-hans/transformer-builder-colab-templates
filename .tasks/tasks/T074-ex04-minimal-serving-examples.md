---
id: T074
enhancement_id: EX-04
title: Create Minimal Serving Examples (FastAPI + Gradio)
status: pending
priority: 3
agent: fullstack
dependencies: [T071]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [export, deployment, serving, examples, enhancement1.0]

context_refs:
  - context/project.md

est_tokens: 11000
actual_tokens: null
---

## Description

Build minimal FastAPI and Gradio serving examples demonstrating how to deploy exported models. Creates reference implementations for text generation (LM) and image classification (vision) endpoints with base64 image support and JSON APIs.

## Business Context

**User Story**: As a deployment engineer, I want runnable serving examples for my exported ONNX model, so I can quickly prototype a production API.

**Why This Matters**: Bridges gap between training and production; accelerates deployment

**What It Unblocks**: Production serving, user-facing demos, edge deployment

**Priority Justification**: Priority 3 - Valuable examples but not blocking core infrastructure

## Acceptance Criteria

- [ ] `examples/serving/fastapi_server.py` created with /generate (text LM) and /predict (vision) endpoints
- [ ] `examples/serving/gradio_demo.py` created with text input box and image upload UI
- [ ] `load_exported_model(export_dir, runtime)` helper loads TorchScript or ONNX models
- [ ] FastAPI server accepts JSON: `{"prompt": "text"}` or `{"image": "base64_data"}`
- [ ] Gradio demo works locally and in Colab (with ngrok for public URL)
- [ ] README in `examples/serving/` with usage instructions
- [ ] Requirements: `fastapi`, `uvicorn`, `gradio` (not in main requirements.txt)
- [ ] Works with models exported by T071

## Test Scenarios

**Test Case 1: FastAPI LM Endpoint**
- Given: Exported LM model in `exports/lm/`
- When: `uvicorn examples.serving.fastapi_server:app --reload`
- Then: POST to `/generate` returns JSON with logits or sampled tokens

**Test Case 2: Gradio Vision Demo**
- Given: Exported vision model
- When: `python examples/serving/gradio_demo.py`
- Then: Opens browser UI, user uploads image, sees top-3 predictions

**Test Case 3: Base64 Image Encoding**
- Given: JPEG image encoded as base64 string
- When: POST to `/predict` with `{"image": "data:image/jpeg;base64,..."}`
- Then: Returns `{"predictions": [{"class": "cat", "prob": 0.92}, ...]}`

**Test Case 4: ONNX Runtime Loading**
- Given: ONNX model, runtime="onnx"
- When: load_exported_model called
- Then: Returns callable function that accepts tensors and returns predictions

**Test Case 5: Gradio Colab Integration**
- Given: Colab notebook, ngrok token set
- When: Launch gradio demo
- Then: Returns public URL (e.g., https://abc123.ngrok.io)

**Test Case 6: Error Handling**
- Given: Invalid image upload (corrupted file)
- When: Gradio processes image
- Then: Displays error message: "Invalid image format"

## Technical Implementation

**Required Components:**

1. **`examples/serving/fastapi_server.py`**
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   import torch

   app = FastAPI()

   class TextRequest(BaseModel):
       prompt: str

   class ImageRequest(BaseModel):
       image: str  # base64 encoded

   @app.post("/generate")
   def generate(req: TextRequest):
       # Load model, tokenize, generate
       return {"logits": [...]}

   @app.post("/predict")
   def predict(req: ImageRequest):
       # Decode base64, preprocess, predict
       return {"predictions": [{"class": "cat", "prob": 0.92}]}
   ```

2. **`examples/serving/gradio_demo.py`**
   ```python
   import gradio as gr
   from utils.training.export_utilities import load_exported_model

   model = load_exported_model("exports/vision/", runtime="torchscript")

   def predict_image(image):
       # Preprocess, predict, format output
       return "Top predictions: cat (92%), dog (5%), bird (3%)"

   demo = gr.Interface(
       fn=predict_image,
       inputs=gr.Image(type="pil"),
       outputs="text"
   )
   demo.launch(share=True)
   ```

3. **`examples/serving/README.md`** with installation and usage

**Validation Commands:**

```bash
# Install serving dependencies
pip install fastapi uvicorn gradio

# Run FastAPI server
uvicorn examples.serving.fastapi_server:app --reload --port 8000

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_data"}'

# Run Gradio demo
python examples/serving/gradio_demo.py
```

## Dependencies

**Hard Dependencies**:
- [T071] Harden export_utilities APIs - Provides load_exported_model helper

**External Dependencies:**
- fastapi, uvicorn, gradio (optional, user installs for serving)

## Design Decisions

**Decision 1: Separate examples for FastAPI and Gradio**
- **Rationale**: Different use cases (API vs UI); keep examples focused
- **Trade-offs**: Code duplication, but clearer for users

**Decision 2: Base64 image encoding in FastAPI**
- **Rationale**: Works with JSON APIs (no multipart form data)
- **Trade-offs**: Larger payloads (~33% overhead), but simpler client code

**Decision 3: Serving deps not in main requirements.txt**
- **Rationale**: Most users won't deploy (training focus); avoid bloat
- **Trade-offs**: Users must manually install, but documented clearly

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Gradio share=True fails on Colab | M - No public URL | M | Document ngrok alternative; provide fallback instructions |
| FastAPI examples too simplistic | M - Not production-ready | H | Add disclaimer that examples are minimal; link to production guides |
| ONNX Runtime not installed | H - ONNX serving fails | M | Add try/except with clear error message to install onnxruntime |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Fourth export tier task (EX-04 from enhancement1.0.md)
**Dependencies:** T071 (export utilities)
**Estimated Complexity:** Standard (example code + documentation)

## Completion Checklist

- [ ] FastAPI server with 2 endpoints created
- [ ] Gradio demo with image upload created
- [ ] load_exported_model helper implemented
- [ ] README with installation and usage
- [ ] All 8 acceptance criteria met
- [ ] All 6 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 3 risks mitigated

**Definition of Done:** FastAPI and Gradio examples run successfully, load exported models, handle text/vision inputs, documented with usage instructions.
