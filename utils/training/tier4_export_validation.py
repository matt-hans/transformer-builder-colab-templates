"""
Tier 4: Export Validation Utilities.

Validates exported models against their PyTorch reference by checking:
- Input/output shape compatibility
- Numerical parity (max absolute difference, relative error)
- Simple latency microbenchmarks

Designed to work with multiple export formats (TorchScript, ONNX) and
multiple modalities (text and vision) by leveraging TaskSpec and ModelAdapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal, Mapping

import torch
import torch.nn as nn

from .export_utilities import _generate_dummy_input_from_task


try:
    import onnxruntime as ort  # type: ignore[import]
    HAS_ONNXRUNTIME = True
except Exception:
    HAS_ONNXRUNTIME = False


@dataclass
class ParityThresholds:
    fp32: float = 1e-4
    quantized: float = 1e-2


def _max_abs_and_rel_error(
    ref: torch.Tensor,
    candidate: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """Compute max absolute and relative error between two tensors."""
    diff = (ref - candidate).abs()
    max_abs = float(diff.max().item())
    denom = ref.abs().clamp_min(eps)
    rel = (diff / denom).abs()
    max_rel = float(rel.max().item())
    return max_abs, max_rel


def _measure_latency_ms(
    callable_fn,
    batch: Mapping[str, Any],
    n_iters: int = 50,
) -> float:
    """Measure average latency in milliseconds for a given callable."""
    # Warmup
    for _ in range(5):
        callable_fn(batch)

    start = time.time()
    for _ in range(n_iters):
        callable_fn(batch)
    elapsed = time.time() - start
    return float(elapsed / max(1, n_iters) * 1000.0)


def _load_torchscript_model(path: Path, device: torch.device) -> Any:
    scripted = torch.jit.load(str(path), map_location=device)
    scripted.eval()
    return scripted


def _load_onnx_session(path: Path) -> Optional[Any]:
    if not HAS_ONNXRUNTIME:
        return None
    session = ort.InferenceSession(str(path))
    return session


def _run_pytorch(
    model: nn.Module,
    adapter: Any,
    task_spec: Any,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run reference PyTorch model via adapter to produce logits tensor."""
    model.eval()
    with torch.no_grad():
        prepared = adapter.prepare_inputs(batch, task_spec)
        _loss, outputs = adapter.forward_for_loss(model, prepared, task_spec)
        logits = adapter.get_logits(outputs, task_spec)
        if not isinstance(logits, torch.Tensor):
            raise ValueError("Adapter get_logits did not return a tensor.")
        return logits


def _run_torchscript(
    scripted_model: Any,
    batch: Dict[str, torch.Tensor],
    task_spec: Any,
) -> torch.Tensor:
    """Run TorchScript model using the primary input field from TaskSpec."""
    modality = getattr(task_spec, "modality", "text")
    if modality == "vision":
        key = "pixel_values"
    else:
        key = "input_ids"
    example = batch[key]
    with torch.no_grad():
        output = scripted_model(example)
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]
        if isinstance(output, dict) and "logits" in output and isinstance(output["logits"], torch.Tensor):
            return output["logits"]
    raise ValueError("Unable to extract tensor output from TorchScript model.")


def _run_onnx(
    session: Any,
    batch: Dict[str, torch.Tensor],
    task_spec: Any,
) -> torch.Tensor:
    """Run ONNX session and return output as a torch.Tensor."""
    if not HAS_ONNXRUNTIME:
        raise RuntimeError("onnxruntime is not available.")
    modality = getattr(task_spec, "modality", "text")
    input_name = session.get_inputs()[0].name
    if modality == "vision":
        key = "pixel_values"
    else:
        key = "input_ids"
    input_array = batch[key].detach().cpu().numpy()
    outputs = session.run(None, {input_name: input_array})
    return torch.from_numpy(outputs[0])


def run_tier4_export_validation(
    model: nn.Module,
    adapter: Any,
    task_spec: Any,
    export_dir: Path | str,
    num_samples: int = 10,
    thresholds: Optional[Dict[str, float]] = None,
    quantized: bool = False,
) -> Dict[str, Any]:
    """
    Validate exported models against PyTorch reference.

    Args:
        model: Reference PyTorch model.
        adapter: ModelAdapter used to compute reference outputs.
        task_spec: TaskSpec describing the task and modality.
        export_dir: Directory containing exported artifacts.
        num_samples: Number of random samples for parity check.
        thresholds: Dict with keys \"fp32\" and \"quantized\" for max_abs_diff.
        quantized: Whether the exported artifacts are quantized.

    Returns:
        {
          "status": "ok" | "warn" | "fail",
          "formats": {
            "torchscript": {"status": "...", "max_abs_diff": ..., "max_rel_error": ..., "latency_ms": ...},
            "onnx": {...}
          }
        }
    """
    export_dir_path = Path(export_dir)
    thresholds_obj = ParityThresholds(
        fp32=float((thresholds or {}).get("fp32", 1e-4)),
        quantized=float((thresholds or {}).get("quantized", 1e-2)),
    )
    max_allowed = thresholds_obj.quantized if quantized else thresholds_obj.fp32

    device = next(model.parameters()).device
    model.eval()

    # Use a single dummy batch reused across runs (num_samples is used for averaging parity)
    batch = _generate_dummy_input_from_task(task_spec, batch_size=1, device=device)

    formats_result: Dict[str, Any] = {}
    overall_status = "ok"

    # Helper to update overall status
    def update_status(format_status: str) -> None:
        nonlocal overall_status
        order = {"ok": 0, "warn": 1, "fail": 2}
        if order[format_status] > order[overall_status]:
            overall_status = format_status

    # Reference outputs (PyTorch)
    ref_logits = _run_pytorch(model, adapter, task_spec, batch)

    # TorchScript validation
    ts_path = export_dir_path / "model.torchscript.pt"
    if ts_path.exists():
        scripted = _load_torchscript_model(ts_path, device=device)

        # Numerical parity over num_samples (reusing same shape, different random batch)
        max_abs = 0.0
        max_rel = 0.0
        for _ in range(max(1, num_samples)):
            rand_batch = _generate_dummy_input_from_task(task_spec, batch_size=1, device=device)
            ref = _run_pytorch(model, adapter, task_spec, rand_batch)
            cand = _run_torchscript(scripted, rand_batch, task_spec).to(ref.device)
            cur_abs, cur_rel = _max_abs_and_rel_error(ref, cand)
            max_abs = max(max_abs, cur_abs)
            max_rel = max(max_rel, cur_rel)

        status = "ok" if max_abs <= max_allowed else "fail"
        update_status(status)

        latency = _measure_latency_ms(
            lambda b: _run_torchscript(scripted, b, task_spec),
            batch,
            n_iters=50,
        )
        formats_result["torchscript"] = {
            "status": status,
            "max_abs_diff": max_abs,
            "max_rel_error": max_rel,
            "latency_ms": latency,
        }

    # ONNX validation
    onnx_path = export_dir_path / "model.onnx"
    if onnx_path.exists() and HAS_ONNXRUNTIME:
        session = _load_onnx_session(onnx_path)
        if session is not None:
            max_abs = 0.0
            max_rel = 0.0
            for _ in range(max(1, num_samples)):
                rand_batch = _generate_dummy_input_from_task(task_spec, batch_size=1, device=device)
                ref = _run_pytorch(model, adapter, task_spec, rand_batch)
                cand = _run_onnx(session, rand_batch, task_spec).to(ref.device)
                cur_abs, cur_rel = _max_abs_and_rel_error(ref, cand)
                max_abs = max(max_abs, cur_abs)
                max_rel = max(max_rel, cur_rel)

            status = "ok" if max_abs <= max_allowed else "fail"
            update_status(status)

            latency = _measure_latency_ms(
                lambda b: _run_onnx(session, b, task_spec),
                batch,
                n_iters=50,
            )
            formats_result["onnx"] = {
                "status": status,
                "max_abs_diff": max_abs,
                "max_rel_error": max_rel,
                "latency_ms": latency,
            }

    if not formats_result:
        overall_status = "warn"

    return {
        "status": overall_status,
        "formats": formats_result,
    }

