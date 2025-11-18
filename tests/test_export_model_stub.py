import json
from pathlib import Path

import torch
import torch.nn as nn

from utils.training.task_spec import TaskSpec
from utils.training.export_utilities import export_model
from utils.adapters import DecoderOnlyLMAdapter, VisionClassificationAdapter


class LMStub(nn.Module):
    def __init__(self, vocab_size: int = 101, d_model: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        return self.head(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(pixel_values))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_lm_task() -> TaskSpec:
    return TaskSpec(
        name="lm_tiny_test",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss"],
        modality="text",
        input_schema={"max_seq_len": 8, "vocab_size": 101},
        output_schema={"vocab_size": 101},
    )


def _make_vision_task() -> TaskSpec:
    return TaskSpec(
        name="vision_tiny_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"],
        modality="vision",
        input_schema={"image_size": [3, 32, 32], "channels_first": True},
        output_schema={"num_classes": 4},
    )


def test_export_model_lm_paths(tmp_path):
    model = LMStub()
    adapter = DecoderOnlyLMAdapter()
    task_spec = _make_lm_task()

    out_dir = tmp_path / "lm_export"
    paths = export_model(
        model=model,
        adapter=adapter,
        task_spec=task_spec,
        export_dir=out_dir,
        formats=["pytorch"],
    )

    assert "pytorch" in paths
    assert "metadata" in paths
    assert paths["pytorch"].exists()
    assert paths["metadata"].exists()

    meta = json.loads(paths["metadata"].read_text())
    assert meta["task_type"] == "lm"
    assert meta["modality"] == "text"


def test_export_model_vision_metadata(tmp_path):
    model = SimpleCNN(num_classes=4)
    adapter = VisionClassificationAdapter()
    task_spec = _make_vision_task()

    out_dir = tmp_path / "vision_export"
    paths = export_model(
        model=model,
        adapter=adapter,
        task_spec=task_spec,
        export_dir=out_dir,
        formats=["pytorch"],
    )

    metadata_path = paths["metadata"]
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["task_type"] == "vision_classification"
    assert metadata["modality"] == "vision"
    assert metadata["formats"] == ["pytorch"]

