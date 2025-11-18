import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
import torch.nn as nn

from utils.test_functions import (
    test_shape_robustness,
    test_gradient_flow,
    run_tier4_export_validation,
)
from utils.training import build_task_spec, TrainingConfig
from utils.training.tier5_monitoring import run_tier5_monitoring
from utils.training.eval_config import EvalConfig
from utils.training.experiment_db import ExperimentDB
from utils.training.export_utilities import export_model
from utils.adapters import DecoderOnlyLMAdapter, VisionClassificationAdapter


class LMStub(nn.Module):
    def __init__(self, vocab_size: int = 101, d_model: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        return self.head(x)


class SimpleCNN(nn.Module):
    """
    Tiny vision model used for Tier 1/2 validation of vision tasks.

    Input:
        pixel_values: [batch_size, 3, H, W]
    Output:
        logits: [batch_size, num_classes]
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(pixel_values))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@dataclass
class TiersConfig:
    task_name: str = "lm_tiny"
    mode: str = "FAST_DEV"
    tier: str | None = None
    vocab_size: int = 101
    max_seq_len: int = 16
    num_classes: int = 4

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TiersConfig":
        return TiersConfig(
            task_name=str(data.get("task_name", "lm_tiny")),
            mode=str(data.get("mode", "FAST_DEV")),
            tier=str(data.get("tier")) if data.get("tier") is not None else None,
            vocab_size=int(data.get("vocab_size", 101)),
            max_seq_len=int(data.get("max_seq_len", 16)),
            num_classes=int(data.get("num_classes", 4)),
        )


def _build_training_config(tcfg: TiersConfig) -> TrainingConfig:
    training_cfg = TrainingConfig(vocab_size=tcfg.vocab_size, max_seq_len=tcfg.max_seq_len)
    training_cfg.task_name = tcfg.task_name
    return training_cfg


def _build_stub_model_and_adapter(tiers_cfg: TiersConfig, task: Any) -> tuple[nn.Module, Any]:
    """
    Build a stub model and adapter pair for the given task.

    Uses LMStub/DecoderOnlyLMAdapter for text and SimpleCNN/VisionClassificationAdapter for vision.
    """
    if getattr(task, "modality", "text") == "vision" and getattr(task, "task_type", None) == "vision_classification":
        adapter = VisionClassificationAdapter()
        num_classes = int(task.output_schema.get("num_classes", tiers_cfg.num_classes))
        model = SimpleCNN(num_classes=num_classes)
    else:
        adapter = DecoderOnlyLMAdapter()
        model = LMStub(vocab_size=tiers_cfg.vocab_size)
    return model, adapter


def run_tier1_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal, stub-based tiers runner for text and vision tasks (Tier 1).
    """
    tiers_cfg = TiersConfig.from_dict(cfg)
    training_cfg = _build_training_config(tiers_cfg)
    task = build_task_spec(training_cfg)

    config_ns = SimpleNamespace(
        vocab_size=tiers_cfg.vocab_size,
        max_seq_len=tiers_cfg.max_seq_len,
        max_batch_size=4,
        image_size=task.input_schema.get("image_size", [3, 32, 32]),
    )

    model, adapter = _build_stub_model_and_adapter(tiers_cfg, task)

    tier1 = {
        "shape": test_shape_robustness(model, config_ns, adapter=adapter, task_spec=task),
        "gradients": test_gradient_flow(model, config_ns, adapter=adapter, task_spec=task),
    }
    return {"tier1": "ok", "details": tier1}


def _validate_export_config(export_cfg: Dict[str, Any]) -> None:
    """Basic schema validation for export config with clear error messages."""
    if not isinstance(export_cfg, dict):
        raise ValueError("Config field 'export' must be an object/dict.")

    formats = export_cfg.get("formats", ["torchscript", "onnx"])
    if not isinstance(formats, list) or not all(isinstance(f, str) for f in formats):
        raise ValueError("Config field 'export.formats' must be a list of strings, e.g. [\"torchscript\", \"onnx\"].")

    quant = export_cfg.get("quantization")
    if quant is not None and quant not in ("dynamic", "static"):
        raise ValueError("Config field 'export.quantization' must be one of null, \"dynamic\", or \"static\".")


def run_export_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Tier 4 export + validation pipeline from config.
    """
    tiers_cfg = TiersConfig.from_dict(cfg)
    training_cfg = _build_training_config(tiers_cfg)
    task = build_task_spec(training_cfg)

    # Align TaskSpec schemas with stub model configuration for safe dummy inputs
    if getattr(task, "modality", "text") == "text":
        # Ensure dummy vocab/length do not exceed stub embedding size
        task.input_schema["vocab_size"] = int(tiers_cfg.vocab_size)
        task.input_schema.setdefault("max_seq_len", int(tiers_cfg.max_seq_len))

    model, adapter = _build_stub_model_and_adapter(tiers_cfg, task)

    export_cfg = cfg.get("export", {})
    _validate_export_config(export_cfg)
    export_dir = export_cfg.get("export_dir", f"exports/{tiers_cfg.task_name}")
    formats: List[str] = export_cfg.get("formats", ["torchscript", "onnx"])
    quantization = export_cfg.get("quantization")

    export_paths = export_model(
        model=model,
        adapter=adapter,
        task_spec=task,
        export_dir=export_dir,
        formats=formats,
        quantization=quantization,
    )

    tier4_results = run_tier4_export_validation(
        model=model,
        adapter=adapter,
        task_spec=task,
        export_dir=export_dir,
        num_samples=5,
        thresholds=None,
        quantized=bool(quantization),
    )

    exports_str = {k: str(v) for k, v in export_paths.items()}

    return {
        "export": exports_str,
        "tier4": tier4_results,
    }


def run_tier5_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Tier 5 monitoring (eval + optional baseline comparison + drift) from config.
    """
    tiers_cfg = TiersConfig.from_dict(cfg)
    training_cfg = _build_training_config(tiers_cfg)
    task = build_task_spec(training_cfg)

    # Build EvalConfig from config overrides or defaults
    eval_dict: Dict[str, Any] = {}
    eval_cfg_raw = cfg.get("eval") or {}
    eval_dict["dataset_id"] = eval_cfg_raw.get("dataset_id", f"{tiers_cfg.task_name}_v1")
    eval_dict["split"] = eval_cfg_raw.get("split", "validation")
    eval_dict["max_eval_examples"] = int(eval_cfg_raw.get("max_eval_examples", 32))
    eval_dict["batch_size"] = int(eval_cfg_raw.get("batch_size", 4))
    eval_dict["num_workers"] = int(eval_cfg_raw.get("num_workers", 0))
    eval_dict["max_seq_length"] = int(eval_cfg_raw.get("max_seq_length", tiers_cfg.max_seq_len))
    eval_dict["eval_interval_steps"] = int(eval_cfg_raw.get("eval_interval_steps", 0))
    eval_dict["eval_on_start"] = bool(eval_cfg_raw.get("eval_on_start", True))
    eval_cfg = EvalConfig.from_dict(eval_dict)
    # Attach training config for downstream dataloader helpers
    setattr(eval_cfg, "training_config", training_cfg)

    model, adapter = _build_stub_model_and_adapter(tiers_cfg, task)

    db_path = cfg.get("db_path", "experiments.db")
    db = ExperimentDB(db_path)

    baseline_run_id = cfg.get("baseline_run_id")
    reference_profile_id = cfg.get("reference_profile_id")

    tier5_results = run_tier5_monitoring(
        model=model,
        adapter=adapter,
        task_spec=task,
        eval_cfg=eval_cfg,
        db=db,
        baseline_run_id=int(baseline_run_id) if baseline_run_id is not None else None,
        reference_profile_id=int(reference_profile_id) if reference_profile_id is not None else None,
    )

    return tier5_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier 1/2/4 tests for LM or vision tasks.")
    parser.add_argument("--config", required=False, help="Path to config JSON (optional)")
    parser.add_argument("--json", action="store_true", help="Print JSON output instead of human-readable text")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    tier = (cfg or {}).get("tier")
    mode = (cfg or {}).get("mode")

    if tier == "4" or mode == "EXPORT":
        out = run_export_from_config(cfg)
    elif tier == "5":
        out = run_tier5_from_config(cfg)
    else:
        out = run_tier1_from_config(cfg)

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        if "tier4" in out:
            print("\n=== Tier 4 Export Validation ===")
            print(f"Status: {out['tier4'].get('status')}")
            for fmt, info in out["tier4"].get("formats", {}).items():
                print(
                    f"- {fmt}: status={info.get('status')}, "
                    f"max_abs_diff={info.get('max_abs_diff'):.3e}, "
                    f"latency_ms={info.get('latency_ms'):.2f}"
                )
            print("\nExported artifacts:")
            for name, path in out.get("export", {}).items():
                print(f"- {name}: {path}")
        else:
            print(out)


if __name__ == "__main__":
    main()
