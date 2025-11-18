"""
Tier 5 monitoring entrypoint: evaluation + baseline comparison + drift.

This module integrates:
- Evaluation runner (Tier 1-style metrics)
- Baseline vs candidate regression testing (T080)
- Input/output drift metrics (T081)

It is designed to be lightweight and callable both from notebooks/CLI and
from CI-style scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .dataset_utilities import build_dataloader
from .drift_metrics import compare_profiles, compute_dataset_profile, log_profile_to_db
from .eval_runner import run_evaluation
from .experiment_db import ExperimentDB
from .regression_testing import compare_models
from .task_spec import TaskSpec


def _get_training_config_from_eval(eval_cfg: Any, task_spec: TaskSpec) -> Any:
    """
    Obtain a TrainingConfig-like object for dataloader/eval runner.

    If eval_cfg has a training_config attribute, reuse it. Otherwise build
    a minimal namespace with only the fields used by build_dataloader.
    """
    training_config = getattr(eval_cfg, "training_config", None)
    if training_config is not None:
        return training_config

    class _DummyCfg:
        pass

    dummy = _DummyCfg()
    setattr(dummy, "vocab_size", task_spec.input_schema.get("vocab_size", 256))
    max_seq_len = getattr(eval_cfg, "max_seq_length", None) or task_spec.input_schema.get("max_seq_len", 128)
    setattr(dummy, "max_seq_len", max_seq_len)
    setattr(dummy, "task_name", getattr(task_spec, "name", "unknown_task"))
    return dummy


def _load_model_from_run(db: ExperimentDB, run_id: int) -> Optional[nn.Module]:
    """
    Best-effort loader for a baseline model from ExperimentDB.

    This uses the training config stored with the run and the same model
    loading logic as the CLI training entrypoint. If checkpoint artifacts
    exist, the latest one is loaded into the model.
    """
    try:
        from cli.run_training import _load_model_from_cfg
    except Exception:
        return None

    try:
        run = db.get_run(run_id)
    except Exception:
        return None

    cfg_dict: Dict[str, Any] = dict(run.get("config", {}))
    model: nn.Module = _load_model_from_cfg(cfg_dict)

    # Attach run_id for downstream logging in compare_models
    setattr(model, "run_id", run_id)

    try:
        artifacts = db.get_artifacts(run_id, artifact_type="checkpoint")
    except Exception:
        artifacts = None

    if artifacts is None or artifacts.empty:
        return model

    ckpt_path = Path(str(artifacts.iloc[0]["filepath"]))
    if not ckpt_path.exists():
        return model

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # Best-effort; allow partial loading
        pass
    return model


def _load_reference_profile_from_run(db: ExperimentDB, run_id: int) -> Optional[Dict[str, Any]]:
    """
    Load the most recent stored profile artifact for a given run_id.
    """
    try:
        artifacts = db.get_artifacts(run_id, artifact_type="profile")
    except Exception:
        return None

    if artifacts.empty:
        return None

    meta_json = artifacts.iloc[0].get("metadata")
    if not meta_json:
        return None
    try:
        meta = json.loads(meta_json)
    except Exception:
        return None
    profile = meta.get("profile")
    if not isinstance(profile, dict):
        return None
    return profile


def run_tier5_monitoring(
    model: nn.Module,
    adapter: Any,
    task_spec: TaskSpec,
    eval_cfg: Any,
    db: ExperimentDB | None = None,
    baseline_run_id: int | None = None,
    reference_profile_id: int | None = None,
) -> Dict[str, Any]:
    """
    Run Tier 5 monitoring: evaluation + optional baseline comparison + drift.

    Args:
        model: Candidate model to evaluate.
        adapter: ModelAdapter instance.
        task_spec: TaskSpec describing task semantics.
        eval_cfg: EvalConfig-like object.
        db: Optional ExperimentDB instance for logging runs, metrics, profiles.
        baseline_run_id: Optional run_id of baseline model for comparison.
        reference_profile_id: Optional run_id whose stored profile acts as
            the reference for drift detection.

    Returns:
        Dict with keys:
            - eval_metrics: Aggregated evaluation metrics for candidate.
            - comparison: Regression comparison dict or None.
            - drift: Drift analysis dict or None.
            - status: "ok" | "warn" | "fail".
            - run_id: Optional run_id for the candidate evaluation.
    """
    training_config = _get_training_config_from_eval(eval_cfg, task_spec)
    dataloader = build_dataloader(task_spec, eval_cfg, training_config)

    # 1) Evaluation of candidate model
    eval_metrics = run_evaluation(
        model=model,
        adapter=adapter,
        task=task_spec,
        eval_config=eval_cfg,
        training_config=training_config,
        dataloader=dataloader,
        metrics_tracker=None,
    )

    run_id: Optional[int] = None
    if db is not None:
        run_info: Dict[str, Any] = {
            "run_name": getattr(eval_cfg, "run_name", f"tier5_validation_{task_spec.task_name}"),
            "task_name": task_spec.task_name,
            "modality": task_spec.modality,
        }
        run_id = db.register_run(run_info)
        db.log_metrics(run_id, eval_metrics, split=getattr(eval_cfg, "split", "eval"))
        setattr(model, "run_id", run_id)

    result: Dict[str, Any] = {
        "eval_metrics": eval_metrics,
        "comparison": None,
        "drift": None,
        "status": "ok",
    }
    if run_id is not None:
        result["run_id"] = run_id

    # 2) Optional baseline comparison (via compare_models)
    comparison: Optional[Dict[str, Any]] = None
    if db is not None and baseline_run_id is not None:
        baseline_model = _load_model_from_run(db, baseline_run_id)
        if baseline_model is not None:
            comparison = compare_models(
                baseline_model=baseline_model,
                candidate_model=model,
                adapter=adapter,
                task_spec=task_spec,
                eval_cfg=eval_cfg,
                db=db,
                comparison_name=f"tier5_baseline_{baseline_run_id}_candidate_{run_id}",
                threshold=0.01,
            )
    result["comparison"] = comparison

    # 3) Optional drift detection using stored reference profile
    drift: Optional[Dict[str, Any]] = None
    if db is not None and reference_profile_id is not None:
        ref_profile = _load_reference_profile_from_run(db, reference_profile_id)
        if ref_profile is not None:
            dataset_for_profile = getattr(dataloader, "dataset", dataloader)
            new_profile = compute_dataset_profile(dataset_for_profile, task_spec, sample_size=1000)
            drift = compare_profiles(ref_profile, new_profile)
            if run_id is not None:
                log_profile_to_db(db, run_id, new_profile, profile_name="tier5_eval_dataset")
    result["drift"] = drift

    # 4) Overall status classification
    status = "ok"

    # Baseline regression: treat regressed metrics as failure, neutral as ok.
    if comparison is not None:
        metrics_block = comparison.get("metrics", {})
        # Look for accuracy or loss first, but fall back to any metric
        primary_names = ["accuracy", "loss"]
        primary = None
        for name in primary_names:
            if name in metrics_block:
                primary = metrics_block[name]
                break
        if primary is None and metrics_block:
            # Arbitrary but deterministic: first metric in sorted order
            key = sorted(metrics_block.keys())[0]
            primary = metrics_block[key]
        if isinstance(primary, dict):
            if primary.get("status") == "regressed":
                status = "fail"

    # Drift status escalates warn/fail
    if drift is not None:
        drift_status = drift.get("status")
        if drift_status == "alert":
            status = "fail"
        elif drift_status == "warn" and status == "ok":
            status = "warn"

    result["status"] = status
    return result


__all__ = ["run_tier5_monitoring"]

