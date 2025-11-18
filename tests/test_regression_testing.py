from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.training.task_spec import TaskSpec
from utils.training.eval_config import EvalConfig
from utils.training.dataset_utilities import TinyVisionDataset, build_dataloader
from utils.training.regression_testing import compare_models, _classify_metric_delta
from utils.training.experiment_db import ExperimentDB
from utils.adapters import VisionClassificationAdapter


class ConstantAccuracyModel(nn.Module):
    """
    Vision stub model that produces controllable accuracy by biasing logits.
    """

    def __init__(self, num_classes: int = 3, bias_index: int = 0) -> None:
        super().__init__()
        self.bias_index = bias_index
        self.num_classes = num_classes
        # Dummy parameter to ensure model.parameters() is non-empty
        self._dummy = nn.Linear(1, 1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        logits = torch.zeros(batch_size, self.num_classes, device=pixel_values.device)
        logits[:, self.bias_index] = 1.0
        return logits


def _make_vision_task(num_classes: int = 4) -> TaskSpec:
    return TaskSpec(
        name="vision_tiny_regression",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"],
        modality="vision",
        input_schema={"image_size": [3, 8, 8], "channels_first": True},
        output_schema={"num_classes": num_classes},
    )


def test_classify_metric_delta_improved_regressed_neutral():
    # Accuracy: higher is better
    res_improved = _classify_metric_delta("accuracy", 0.7, 0.75, threshold=0.01)
    assert res_improved["status"] == "improved"

    res_regressed = _classify_metric_delta("accuracy", 0.8, 0.72, threshold=0.01)
    assert res_regressed["status"] == "regressed"

    res_neutral = _classify_metric_delta("accuracy", 0.70, 0.705, threshold=0.02)
    assert res_neutral["status"] == "neutral"

    # Loss: lower is better
    res_improved_loss = _classify_metric_delta("loss", 0.5, 0.4, threshold=0.01)
    assert res_improved_loss["status"] == "improved"

    res_regressed_loss = _classify_metric_delta("loss", 0.4, 0.5, threshold=0.01)
    assert res_regressed_loss["status"] == "regressed"


def test_compare_models_vision_improvement_and_db_logging(tmp_path):
    device = torch.device("cpu")
    num_classes = 4
    task = _make_vision_task(num_classes=num_classes)
    adapter = VisionClassificationAdapter()

    # Dataset with deterministic labels
    dataset = TinyVisionDataset(data_dir=tmp_path, image_size=(3, 8, 8))
    dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    eval_cfg = EvalConfig(
        dataset_id="vision_tiny_regression",
        split="validation",
        max_eval_examples=16,
        batch_size=4,
        num_workers=0,
        max_seq_length=8,
        eval_interval_steps=0,
        eval_on_start=True,
    )

    train_cfg = SimpleNamespace(task_name="vision_tiny")
    _ = build_dataloader  # silence lint; compare_models rebuilds dataloader internally
    baseline_model = ConstantAccuracyModel(num_classes=num_classes, bias_index=0).to(device)
    candidate_model = ConstantAccuracyModel(num_classes=num_classes, bias_index=1).to(device)

    # Attach fake run_ids for DB logging
    baseline_model.run_id = 1
    candidate_model.run_id = 2

    db_path = tmp_path / "experiments.db"
    db = ExperimentDB(db_path)

    result = compare_models(
        baseline_model,
        candidate_model,
        adapter,
        task,
        eval_cfg,
        db=db,
        comparison_name="vision-regression-test",
        threshold=0.0,
    )

    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    acc_info = result["metrics"]["accuracy"]
    assert "baseline" in acc_info and "candidate" in acc_info and "delta" in acc_info
    assert acc_info["status"] in {"improved", "regressed", "neutral"}

    # When db is provided and run_ids are set, a comparison_id should be recorded
    assert "comparison_id" in result
    assert isinstance(result["comparison_id"], int)
