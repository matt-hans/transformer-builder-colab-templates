import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn

from utils.test_functions import test_shape_robustness, test_gradient_flow
from utils.training import build_task_spec, TrainingConfig
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
    vocab_size: int = 101
    max_seq_len: int = 16
    num_classes: int = 4

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TiersConfig":
        return TiersConfig(
            task_name=str(data.get("task_name", "lm_tiny")),
            mode=str(data.get("mode", "FAST_DEV")),
            vocab_size=int(data.get("vocab_size", 101)),
            max_seq_len=int(data.get("max_seq_len", 16)),
            num_classes=int(data.get("num_classes", 4)),
        )


def _build_training_config(tcfg: TiersConfig) -> TrainingConfig:
    training_cfg = TrainingConfig(vocab_size=tcfg.vocab_size, max_seq_len=tcfg.max_seq_len)
    training_cfg.task_name = tcfg.task_name
    return training_cfg


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal, stub-based tiers runner for text and vision tasks.

    For text tasks (e.g., lm_tiny) this uses LMStub + DecoderOnlyLMAdapter.
    For vision tasks (vision_tiny) this uses SimpleCNN + VisionClassificationAdapter.
    """
    tiers_cfg = TiersConfig.from_dict(cfg)
    training_cfg = _build_training_config(tiers_cfg)
    task = build_task_spec(training_cfg)

    # Shared config namespace for Tier 1 tests
    config_ns = SimpleNamespace(
        vocab_size=tiers_cfg.vocab_size,
        max_seq_len=tiers_cfg.max_seq_len,
        max_batch_size=4,
        image_size=task.input_schema.get("image_size", [3, 32, 32]),
    )

    if task.modality == "vision" and task.task_type == "vision_classification":
        adapter = VisionClassificationAdapter()
        model = SimpleCNN(num_classes=int(task.output_schema.get("num_classes", tiers_cfg.num_classes)))
    else:
        adapter = DecoderOnlyLMAdapter()
        model = LMStub(vocab_size=tiers_cfg.vocab_size)

    tier1 = {
        "shape": test_shape_robustness(model, config_ns, adapter=adapter, task_spec=task),
        "gradients": test_gradient_flow(model, config_ns, adapter=adapter, task_spec=task),
    }
    return {"tier1": "ok", "details": tier1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier 1/2 tests for LM or vision tasks.")
    parser.add_argument("--config", required=False, help="Path to config JSON (optional)")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    out = run_from_config(cfg)
    print(out)


if __name__ == "__main__":
    main()
