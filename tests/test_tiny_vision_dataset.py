import json
from pathlib import Path

import torch

from utils.training.dataset_utilities import TinyVisionDataset
from utils.training.task_spec import TaskSpec
from utils.training.dataset_utilities import build_dataloader
from types import SimpleNamespace
from utils.training.eval_config import EvalConfig


def test_tiny_vision_dataset_length(tmp_path):
    data_dir = tmp_path / "vision_tiny"
    data_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "img_000.png": 0,
        "img_001.png": 1,
        "img_002.png": 0,
    }
    labels_path = data_dir / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    dataset = TinyVisionDataset(data_dir=data_dir, image_size=(3, 32, 32))

    assert len(dataset) == len(labels)


def test_tiny_vision_dataset_item_shape(tmp_path):
    data_dir = tmp_path / "vision_tiny"
    data_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "img_000.png": 0,
    }
    (data_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    dataset = TinyVisionDataset(data_dir=data_dir, image_size=(3, 32, 32))
    item = dataset[0]

    pixel_values = item["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert tuple(pixel_values.shape) == (3, 32, 32)


def test_build_dataloader_for_vision_task(tmp_path):
    # Prepare a tiny on-disk vision dataset
    data_dir = tmp_path / "vision" / "vision_tiny"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = {f"img_{i:03d}.png": i % 4 for i in range(8)}
    (data_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    # Monkeypatch base path via symlink or copying is out of scope here; instead
    # we rely on TinyVisionDataset's synthetic fallback for the default path.
    # This test focuses on wiring and shapes rather than actual files under examples/.

    task_spec = TaskSpec(
        name="vision_tiny",
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

    train_cfg = SimpleNamespace(task_name="vision_tiny")
    eval_cfg = EvalConfig(
        dataset_id="vision_tiny_v1",
        split="validation",
        max_eval_examples=4,
        batch_size=2,
        num_workers=0,
        max_seq_length=32,
        eval_interval_steps=0,
        eval_on_start=True,
    )

    dl = build_dataloader(task_spec, eval_cfg, train_cfg)
    batch = next(iter(dl))

    assert "pixel_values" in batch
    assert "labels" in batch
    assert batch["pixel_values"].shape[1:] == (3, 32, 32)
    assert batch["labels"].shape[0] == 2

