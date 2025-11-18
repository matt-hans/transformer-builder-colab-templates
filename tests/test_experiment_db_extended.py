from pathlib import Path

import pandas as pd

from utils.training.experiment_db import ExperimentDB


def test_register_run_and_log_metrics(tmp_path):
    db_path = tmp_path / "experiments.db"
    db = ExperimentDB(db_path)

    run_info = {
        "run_name": "baseline-v1",
        "task_name": "lm_tiny",
        "modality": "text",
        "strategy": "auto",
        "devices": "1",
        "artifact_paths": {"checkpoint": "checkpoints/run_1/best.ckpt"},
    }

    run_id = db.register_run(run_info)
    assert isinstance(run_id, int)

    # Log a batch of metrics
    db.log_metrics(
        run_id,
        metrics={"train/loss": 0.45, "train/accuracy": 0.82},
        split="train",
        epoch=5,
    )

    df = db.get_run_metrics(run_id, "train/loss")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["epoch"] == 5
    assert df.iloc[0]["value"] == 0.45


def test_create_comparison(tmp_path):
    db_path = tmp_path / "experiments.db"
    db = ExperimentDB(db_path)

    run_id_baseline = db.register_run({"run_name": "baseline", "config": {}})
    run_id_candidate = db.register_run({"run_name": "candidate", "config": {}})

    comparison_id = db.create_comparison(
        baseline_run_id=run_id_baseline,
        candidate_run_id=run_id_candidate,
        notes="New architecture test",
    )

    assert isinstance(comparison_id, int)


def test_get_artifacts_filtered_by_type(tmp_path):
    db_path = tmp_path / "experiments.db"
    db = ExperimentDB(db_path)

    run_id = db.log_run("run-artifacts", {"learning_rate": 1e-3})
    db.log_artifact(run_id, "checkpoint", "checkpoints/epoch_1.pt")
    db.log_artifact(run_id, "plot", "plots/loss.png")

    df_all = db.get_artifacts(run_id)
    assert len(df_all) == 2

    df_ckpt = db.get_artifacts(run_id, artifact_type="checkpoint")
    assert len(df_ckpt) == 1
    assert df_ckpt.iloc[0]["artifact_type"] == "checkpoint"
    assert df_ckpt.iloc[0]["filepath"] == "checkpoints/epoch_1.pt"
