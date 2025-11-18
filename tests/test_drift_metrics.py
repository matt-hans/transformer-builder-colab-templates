import math
from typing import Dict, Any

import torch

from utils.training.task_spec import TaskSpec
from utils.training.drift_metrics import (
    _js_distance,
    compute_dataset_profile,
    compare_profiles,
    log_profile_to_db,
)
from utils.training.experiment_db import ExperimentDB


class _TextDataset:
    def __init__(self, lengths):
        self._lengths = list(lengths)

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, idx) -> Dict[str, Any]:
        length = self._lengths[idx]
        # Simple ascending token IDs
        seq = list(range(length))
        return {"input_ids": seq}


class _VisionDataset:
    def __init__(self, brightness: float, num_items: int = 16):
        self.brightness = brightness
        self.num_items = num_items

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx) -> Dict[str, Any]:
        # RGB image with constant brightness in [0, 1]
        img = torch.full((3, 8, 8), float(self.brightness))
        return {"pixel_values": img, "labels": 0}


def test_js_distance_zero_for_identical_distributions():
    p = [0.2, 0.3, 0.5]
    q = [0.2, 0.3, 0.5]
    d = _js_distance(p, q)
    assert d == 0.0


def test_compare_profiles_text_seq_length_drift_alert():
    # Reference: short sequences, New: long sequences
    ref_ds = _TextDataset([10] * 100)
    new_ds = _TextDataset([200] * 100)

    task = TaskSpec(
        name="lm_drift_test",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "perplexity"],
        modality="text",
        input_schema={"max_seq_len": 256, "vocab_size": 128},
        output_schema={"vocab_size": 128},
    )

    ref_profile = compute_dataset_profile(ref_ds, task, sample_size=1000)
    new_profile = compute_dataset_profile(new_ds, task, sample_size=1000)

    res = compare_profiles(ref_profile, new_profile)
    assert res["status"] in {"warn", "alert"}
    assert res["drift_scores"]["seq_length_js"] > 0.0


def test_compare_profiles_no_drift_ok_status():
    ds = _TextDataset([32] * 50)
    task = TaskSpec(
        name="lm_drift_test_same",
        task_type="lm",
        model_family="decoder_only",
        input_fields=["input_ids"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss"],
        modality="text",
        input_schema={"max_seq_len": 64, "vocab_size": 64},
        output_schema={"vocab_size": 64},
    )

    profile_a = compute_dataset_profile(ds, task, sample_size=1000)
    profile_b = compute_dataset_profile(ds, task, sample_size=1000)
    res = compare_profiles(profile_a, profile_b)

    assert res["status"] == "ok"
    assert math.isclose(res["max_drift"], 0.0, abs_tol=1e-6)


def test_compare_profiles_vision_brightness_and_channel_shift():
    ref_ds = _VisionDataset(brightness=0.5)
    new_ds = _VisionDataset(brightness=0.8)
    task = TaskSpec(
        name="vision_drift_test",
        task_type="vision_classification",
        model_family="encoder_only",
        input_fields=["pixel_values"],
        target_field="labels",
        loss_type="cross_entropy",
        metrics=["loss", "accuracy"],
        modality="vision",
        input_schema={"image_size": [3, 8, 8], "channels_first": True},
        output_schema={"num_classes": 2},
    )

    ref_profile = compute_dataset_profile(ref_ds, task, sample_size=32)
    new_profile = compute_dataset_profile(new_ds, task, sample_size=32)
    res = compare_profiles(ref_profile, new_profile)

    assert "brightness_js" in res["drift_scores"]
    assert res["drift_scores"]["brightness_js"] >= 0.0
    assert "channel_mean_distance" in res["drift_scores"]
    assert res["drift_scores"]["channel_mean_distance"] > 0.0


def test_log_profile_to_db_stores_profile_metadata(tmp_path):
    db_path = tmp_path / "experiments.db"
    db = ExperimentDB(db_path)
    run_id = db.log_run("drift-profile-run", {"learning_rate": 1e-3})

    profile = {"modality": "text", "seq_length_mean": 10.0}
    log_profile_to_db(db, run_id, profile, profile_name="test_profile")

    artifacts = db.get_artifacts(run_id, artifact_type="profile")
    assert len(artifacts) == 1
    assert "profile:test_profile" in artifacts.iloc[0]["filepath"]

