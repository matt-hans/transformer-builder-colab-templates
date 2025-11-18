import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from utils.training import (
    TrainingConfig,
    build_task_spec,
    build_eval_config,
)
from utils.training.regression_testing import compare_models
from utils.training.experiment_db import ExperimentDB
from utils.adapters import DecoderOnlyLMAdapter, VisionClassificationAdapter


def _load_checkpoint_model(checkpoint_path: Path, model_ctor: Any) -> nn.Module:
    model = model_ctor()
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs candidate regression test.")
    parser.add_argument("--baseline-run-id", type=int, required=True)
    parser.add_argument("--candidate-run-id", type=int, required=True)
    parser.add_argument("--db-path", type=str, default="experiments.db")
    parser.add_argument("--task-name", type=str, default="lm_tiny")
    parser.add_argument("--metric-threshold", type=float, default=0.01)
    args = parser.parse_args()

    db = ExperimentDB(args.db_path)
    baseline_run = db.get_run(args.baseline_run_id)
    candidate_run = db.get_run(args.candidate_run_id)

    baseline_cfg = TrainingConfig.from_dict(baseline_run["config"])
    candidate_cfg = TrainingConfig.from_dict(candidate_run["config"])

    # Use baseline config to build task/eval
    if args.task_name:
        baseline_cfg.task_name = args.task_name
    task_spec = build_task_spec(baseline_cfg)
    eval_cfg = build_eval_config(baseline_cfg)

    # Simple adapter selection based on modality/task_type
    if getattr(task_spec, "modality", "") == "vision":
        adapter = VisionClassificationAdapter()

        def _ctor() -> nn.Module:
            num_classes = int(task_spec.output_schema.get("num_classes", 10))
            from tests.test_eval_runner_vision import SimpleVisionStub  # noqa: WPS433

            return SimpleVisionStub(num_classes=num_classes)

    else:
        adapter = DecoderOnlyLMAdapter()

        def _ctor() -> nn.Module:
            vocab_size = int(baseline_cfg.vocab_size)
            from cli.run_training import LMStub  # noqa: WPS433

            return LMStub(vocab_size=vocab_size)

    def _find_best_checkpoint(db_obj: ExperimentDB, run_id: int) -> Path:
        artifacts_df = db_obj.get_artifacts(run_id, artifact_type="checkpoint")
        if artifacts_df.empty:
            raise RuntimeError(f"No checkpoint artifacts found for run_id={run_id}")
        # Use most recent checkpoint (first row due to DESC ordering)
        ckpt_path = artifacts_df.iloc[0]["filepath"]
        return Path(ckpt_path)

    baseline_ckpt = _find_best_checkpoint(db, args.baseline_run_id)
    candidate_ckpt = _find_best_checkpoint(db, args.candidate_run_id)

    baseline_model = _load_checkpoint_model(baseline_ckpt, _ctor)
    candidate_model = _load_checkpoint_model(candidate_ckpt, _ctor)

    baseline_model.run_id = args.baseline_run_id
    candidate_model.run_id = args.candidate_run_id

    result = compare_models(
        baseline_model,
        candidate_model,
        adapter,
        task_spec,
        eval_cfg,
        db=db,
        comparison_name=f"run_{args.baseline_run_id}_vs_{args.candidate_run_id}",
        threshold=args.metric_threshold,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
