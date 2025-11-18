import torch
import torch.nn as nn
from types import SimpleNamespace

from utils.training.task_spec import TaskSpec
from utils.training.eval_config import EvalConfig
from utils.training.dataset_utilities import TinyVisionDataset
from utils.training.eval_runner import run_evaluation
from utils.adapters import VisionClassificationAdapter


class SimpleVisionStub(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(pixel_values))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_vision_task(num_classes: int = 10) -> TaskSpec:
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
        output_schema={"num_classes": num_classes},
    )


def test_eval_runner_vision_topk_metrics(tmp_path):
    device = torch.device("cpu")
    num_classes = 6
    model = SimpleVisionStub(num_classes=num_classes).to(device)
    adapter = VisionClassificationAdapter()
    task = _make_vision_task(num_classes=num_classes)

    # Use a synthetic TinyVisionDataset (fallback uses random tensors)
    dataset = TinyVisionDataset(data_dir=tmp_path, image_size=(3, 32, 32))
    dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    train_cfg = SimpleNamespace(task_name="vision_tiny")
    eval_cfg = EvalConfig(
        dataset_id="vision_tiny_v1",
        split="validation",
        max_eval_examples=16,
        batch_size=4,
        num_workers=0,
        max_seq_length=32,
        eval_interval_steps=0,
        eval_on_start=True,
    )

    summary = run_evaluation(model, adapter, task, eval_cfg, train_cfg, dl, metrics_tracker=None)

    assert "loss" in summary
    assert "accuracy" in summary
    assert "top3_accuracy" in summary
    assert "top5_accuracy" in summary
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["top3_accuracy"] <= 1.0
    assert 0.0 <= summary["top5_accuracy"] <= 1.0

