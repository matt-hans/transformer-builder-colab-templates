import os
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
import torch
from PIL import Image

from utils.training.export_utilities import load_exported_model


LM_EXPORT_DIR = Path(os.environ.get("LM_EXPORT_DIR", "exports/lm_tiny"))
VISION_EXPORT_DIR = Path(os.environ.get("VISION_EXPORT_DIR", "exports/vision_tiny"))


def _load_metadata(export_dir: Path) -> Dict[str, Any]:
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_lm() -> tuple[Any, Dict[str, Any]]:
    if not LM_EXPORT_DIR.exists():
        return None, {}
    model = load_exported_model(LM_EXPORT_DIR, runtime="torchscript")
    meta = _load_metadata(LM_EXPORT_DIR)
    return model, meta


def _load_vision() -> tuple[Any, Dict[str, Any]]:
    if not VISION_EXPORT_DIR.exists():
        return None, {}
    model = load_exported_model(VISION_EXPORT_DIR, runtime="torchscript")
    meta = _load_metadata(VISION_EXPORT_DIR)
    return model, meta


import json

lm_model, lm_metadata = _load_lm()
vision_model, vision_metadata = _load_vision()


def predict_text(prompt: str) -> str:
    if lm_model is None:
        return "LM model not available. Please export a model first."

    input_schema = lm_metadata.get("input_shape", {}).get("schema", {})
    vocab_size = int(input_schema.get("vocab_size", 101))
    max_seq_len = int(input_schema.get("max_seq_len", 16))

    ids: List[int] = [ord(c) % vocab_size for c in prompt][:max_seq_len]
    if not ids:
        ids = [0]
    if len(ids) < max_seq_len:
        ids += [0] * (max_seq_len - len(ids))

    input_ids = torch.tensor([ids], dtype=torch.long)
    logits = lm_model(input_ids)
    # Show logits for last position as a simple demonstration
    last_logits = logits[0, -1].tolist()
    return f"Last-token logits (first 10): {last_logits[:10]}"


def predict_image(image: Image.Image | None) -> str:
    if vision_model is None:
        return "Vision model not available. Please export a model first."
    if image is None:
        return "No image provided."

    schema = vision_metadata.get("input_shape", {}).get("schema", {})
    image_size = schema.get("image_size", [3, 32, 32])
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 3:
        c, h, w = 3, 32, 32
    else:
        c, h, w = int(image_size[0]), int(image_size[1]), int(image_size[2])

    image = image.convert("RGB").resize((w, h))
    import numpy as np

    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    pixel_values = tensor.unsqueeze(0)

    logits = vision_model(pixel_values)
    probs = torch.softmax(logits, dim=-1)[0]
    topk = min(3, probs.shape[-1])
    values, indices = torch.topk(probs, k=topk)

    parts = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        parts.append(f"class_{idx} ({score:.2%})")

    return "Top predictions: " + ", ".join(parts)


text_iface = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt..."),
    outputs="text",
    title="Text LM (TorchScript)",
)

image_iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Vision Classification (TorchScript)",
)

demo = gr.TabbedInterface([text_iface, image_iface], ["Text LM", "Vision"])

if __name__ == "__main__":
    # share=True is useful in Colab; locally you can omit it.
    demo.launch(share=True)

