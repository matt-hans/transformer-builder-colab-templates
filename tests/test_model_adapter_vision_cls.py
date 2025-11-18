import torch
import torch.nn as nn

from utils.training.task_spec import TaskSpec
from utils.adapters import VisionClassificationAdapter


class SimpleVisionCNN(nn.Module):
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


def test_vision_adapter_forward_and_loss():
    device = torch.device("cpu")
    num_classes = 10
    model = SimpleVisionCNN(num_classes=num_classes).to(device)
    adapter = VisionClassificationAdapter()
    task = _make_vision_task(num_classes=num_classes)

    batch = {
        "pixel_values": torch.randn(4, 3, 32, 32, device=device),
        "labels": torch.randint(0, num_classes, (4,), device=device),
    }

    prepared = adapter.prepare_inputs(batch, task)
    loss, outputs = adapter.forward_for_loss(model, prepared, task)

    assert loss is not None
    assert loss.ndim == 0
    assert loss.requires_grad

    logits = outputs["logits"]
    assert logits.shape == (4, num_classes)


def test_vision_adapter_metrics_accuracy_range():
    device = torch.device("cpu")
    num_classes = 4
    adapter = VisionClassificationAdapter()
    task = _make_vision_task(num_classes=num_classes)

    logits = torch.zeros(4, num_classes, device=device)
    # Make first two predictions correct, last two incorrect
    labels = torch.tensor([0, 1, 2, 3], device=device)
    logits[0, 0] = 10.0
    logits[1, 1] = 10.0
    logits[2, 0] = 10.0
    logits[3, 0] = 10.0

    outputs = {"logits": logits}
    batch = {"labels": labels}

    metrics = adapter.get_logits(outputs, task)  # smoke test
    assert metrics.shape == (4, num_classes)

    # compute accuracy via predict
    preds = adapter.predict(outputs, task)
    accuracy = float((preds == labels).float().mean().item())
    assert 0.0 <= accuracy <= 1.0

