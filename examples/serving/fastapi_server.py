import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from utils.training.export_utilities import load_exported_model


app = FastAPI(title="Transformer Builder Serving Example")


class TextRequest(BaseModel):
    prompt: str


class ImageRequest(BaseModel):
    image: str  # base64-encoded image (optionally with data URL prefix)


def _load_metadata(export_dir: Path) -> Dict[str, Any]:
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _decode_base64_image(data: str) -> Image.Image:
    if "," in data:
        # Handle data URLs like "data:image/png;base64,...."
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image format") from exc
    return image


# Default export directories (can be overridden via env vars)
LM_EXPORT_DIR = Path(os.environ.get("LM_EXPORT_DIR", "exports/lm_tiny"))
VISION_EXPORT_DIR = Path(os.environ.get("VISION_EXPORT_DIR", "exports/vision_tiny"))

lm_model = None
lm_metadata: Dict[str, Any] = {}
vision_model = None
vision_metadata: Dict[str, Any] = {}


@app.on_event("startup")
def _startup_load_models() -> None:
    global lm_model, lm_metadata, vision_model, vision_metadata

    if LM_EXPORT_DIR.exists():
        lm_model = load_exported_model(LM_EXPORT_DIR, runtime="torchscript")
        lm_metadata = _load_metadata(LM_EXPORT_DIR)

    if VISION_EXPORT_DIR.exists():
        try:
            vision_model = load_exported_model(VISION_EXPORT_DIR, runtime="torchscript")
            vision_metadata = _load_metadata(VISION_EXPORT_DIR)
        except Exception:
            vision_model = None
            vision_metadata = {}


@app.post("/generate")
def generate(req: TextRequest) -> Dict[str, Any]:
    """
    Minimal text generation endpoint.

    For demonstration purposes, this applies a trivial character-level
    mapping to IDs based on metadata's vocab_size/max_seq_len and returns
    the raw logits from the exported model.
    """
    if lm_model is None:
        raise HTTPException(status_code=503, detail="LM model is not loaded")

    input_schema = lm_metadata.get("input_shape", {}).get("schema", {})
    vocab_size = int(input_schema.get("vocab_size", 101))
    max_seq_len = int(input_schema.get("max_seq_len", 16))

    # Simple char-level encoding for example purposes
    ids: List[int] = [ord(c) % vocab_size for c in req.prompt][:max_seq_len]
    if not ids:
        ids = [0]
    # Pad or trim to max_seq_len
    if len(ids) < max_seq_len:
        ids += [0] * (max_seq_len - len(ids))

    input_ids = torch.tensor([ids], dtype=torch.long)
    logits = lm_model(input_ids)
    return {"logits": logits[0].tolist()}


@app.post("/predict")
def predict(req: ImageRequest) -> Dict[str, Any]:
    """
    Minimal vision classification endpoint.

    Accepts a base64-encoded image and returns top-3 class predictions
    based on the exported vision model.
    """
    if vision_model is None:
        raise HTTPException(status_code=503, detail="Vision model is not loaded")

    image = _decode_base64_image(req.image)

    schema = vision_metadata.get("input_shape", {}).get("schema", {})
    image_size = schema.get("image_size", [3, 32, 32])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 3:
        c, h, w = 3, 32, 32
    else:
        c, h, w = int(image_size[0]), int(image_size[1]), int(image_size[2])

    # Resize and convert to tensor in [0, 1]
    image = image.resize((w, h))
    import numpy as np

    arr = np.asarray(image).astype("float32") / 255.0  # HWC
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
    pixel_values = tensor.unsqueeze(0)  # BCHW

    logits = vision_model(pixel_values)
    probs = torch.softmax(logits, dim=-1)[0]
    topk = min(3, probs.shape[-1])
    values, indices = torch.topk(probs, k=topk)

    preds: List[Dict[str, Any]] = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        preds.append({"class": f"class_{idx}", "prob": float(score)})

    return {"predictions": preds}

