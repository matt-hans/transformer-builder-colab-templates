"""
Lightweight input/output drift metrics for text and vision tasks.

This module provides small, interpretable statistics that can be used to
detect dataset/profile drift between a reference window (e.g., training or
validation) and a new window (e.g., production traffic).

The design intentionally favors:
- Simple aggregates (means, histograms) over learned detectors
- Symmetric, bounded divergences (Jensen–Shannon distance)
- Modality-aware routing via TaskSpec.modality
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.stats import entropy

from .task_spec import TaskSpec

_EPS = 1e-12


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen–Shannon distance between two discrete distributions.

    Follows the SciPy convention: sqrt(JS divergence), bounded in [0, 1].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + _EPS
    q = q + _EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    return float(np.sqrt(max(js_div, 0.0)))


def _iterate_dataset(dataset: Any, sample_size: int) -> Iterable[Mapping[str, Any]]:
    """
    Yield up to `sample_size` examples from a dataset-like object.

    Supports sequence-style datasets (len/__getitem__) and generic iterables.
    """
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        n = len(dataset)  # type: ignore[arg-type]
        limit = min(sample_size, n)
        for idx in range(limit):
            yield dataset[idx]
    else:
        for idx, item in enumerate(dataset):
            if idx >= sample_size:
                break
            yield item


def _as_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def compute_dataset_profile(
    dataset: Any,
    task_spec: TaskSpec,
    sample_size: int = 1000,
) -> Dict[str, Any]:
    """
    Compute a simple statistical profile for a dataset.

    For text tasks:
      - Sequence length mean/std
      - Fixed-bin sequence length histogram
      - Top-100 token IDs by frequency (from ``input_ids``)
      - Optional output histogram if labels/predictions present

    For vision tasks:
      - Per-channel mean/std (RGB)
      - Brightness histogram (5 bins over [0, 1])
      - Optional output histogram if labels/predictions present
    """
    if task_spec.modality == "text":
        return _compute_text_profile(dataset, task_spec, sample_size)
    if task_spec.modality == "vision":
        return _compute_vision_profile(dataset, task_spec, sample_size)
    raise ValueError(f"Unsupported modality for drift metrics: {task_spec.modality}")


def _compute_text_profile(
    dataset: Any,
    task_spec: TaskSpec,
    sample_size: int,
) -> Dict[str, Any]:
    lengths: List[int] = []
    token_counts: Counter = Counter()
    output_counts: Counter = Counter()

    for item in _iterate_dataset(dataset, sample_size):
        seq = item.get("input_ids")
        if seq is None:
            # Fallback: single string field "text"
            text = item.get("text")
            if text is None:
                continue
            length_val = len(str(text))
            lengths.append(length_val)
            continue

        seq_tensor = _as_tensor(seq)
        if seq_tensor.dim() == 0:
            continue
        # Treat last dimension as sequence length
        seq_flat = seq_tensor.view(-1)
        lengths.append(int(seq_flat.shape[0]))
        for tok in seq_flat.tolist():
            token_counts[int(tok)] += 1

        # Optional output distribution (labels or predictions)
        label_key = task_spec.target_field or "labels"
        out = item.get("predictions", item.get(label_key))
        if out is not None:
            out_tensor = _as_tensor(out).view(-1)
            for c in out_tensor.tolist():
                output_counts[int(c)] += 1

    if lengths:
        lengths_arr = np.array(lengths, dtype=float)
        seq_length_mean = float(lengths_arr.mean())
        seq_length_std = float(lengths_arr.std())
        # Fixed bins for comparability: 10 bins over [0, 512]
        bins = np.linspace(0, 512, 11)
        hist, _ = np.histogram(np.clip(lengths_arr, 0, 512), bins=bins)
        seq_length_hist = hist.astype(int).tolist()
        seq_length_bins = bins.tolist()
    else:
        seq_length_mean = 0.0
        seq_length_std = 0.0
        seq_length_hist = [0] * 10
        seq_length_bins = np.linspace(0, 512, 11).tolist()

    # Top-100 tokens
    top_tokens = [tok_id for tok_id, _ in token_counts.most_common(100)]

    profile: Dict[str, Any] = {
        "modality": "text",
        "seq_length_mean": seq_length_mean,
        "seq_length_std": seq_length_std,
        "seq_length_hist": seq_length_hist,
        "seq_length_bins": seq_length_bins,
        "top_tokens": top_tokens,
    }

    if output_counts:
        # Build histogram over sorted class IDs for stable comparison
        classes = sorted(output_counts.keys())
        counts = [int(output_counts[c]) for c in classes]
        profile["output_classes"] = classes
        profile["output_hist"] = counts

    return profile


def _compute_vision_profile(
    dataset: Any,
    task_spec: TaskSpec,
    sample_size: int,
) -> Dict[str, Any]:
    channel_vals: List[List[float]] = [[], [], []]
    brightness_vals: List[float] = []
    output_counts: Counter = Counter()

    for item in _iterate_dataset(dataset, sample_size):
        img = item.get("pixel_values")
        if img is None:
            continue
        img_tensor = _as_tensor(img).float()
        if img_tensor.dim() != 3:
            # Expect [C, H, W]; attempt to reshape if necessary
            img_tensor = img_tensor.view(3, -1, -1)

        c, h, w = img_tensor.shape
        if c < 3:
            continue

        for ch in range(3):
            channel_vals[ch].append(float(img_tensor[ch].mean().item()))
        brightness_vals.append(float(img_tensor.mean().item()))

        # Optional output distribution (labels or predictions)
        label_key = task_spec.target_field or "labels"
        out = item.get("predictions", item.get(label_key))
        if out is not None:
            out_tensor = _as_tensor(out).view(-1)
            for c_val in out_tensor.tolist():
                output_counts[int(c_val)] += 1

    if any(channel_vals):
        channel_means = [
            float(np.mean(vals)) if vals else 0.0 for vals in channel_vals
        ]
        channel_stds = [
            float(np.std(vals)) if vals else 0.0 for vals in channel_vals
        ]
    else:
        channel_means = [0.0, 0.0, 0.0]
        channel_stds = [0.0, 0.0, 0.0]

    if brightness_vals:
        brightness_arr = np.array(brightness_vals, dtype=float)
        # 5 bins over [0, 1]
        bins = np.linspace(0.0, 1.0, 6)
        hist, _ = np.histogram(np.clip(brightness_arr, 0.0, 1.0), bins=bins)
        brightness_hist = hist.astype(int).tolist()
        brightness_bins = bins.tolist()
    else:
        brightness_hist = [0] * 5
        brightness_bins = np.linspace(0.0, 1.0, 6).tolist()

    profile: Dict[str, Any] = {
        "modality": "vision",
        "channel_means": channel_means,
        "channel_stds": channel_stds,
        "brightness_hist": brightness_hist,
        "brightness_bins": brightness_bins,
    }

    if output_counts:
        classes = sorted(output_counts.keys())
        counts = [int(output_counts[c]) for c in classes]
        profile["output_classes"] = classes
        profile["output_hist"] = counts

    return profile


def compare_profiles(
    ref_profile: Dict[str, Any],
    new_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two dataset profiles and compute drift scores.

    Returns a dict:
        {
          "drift_scores": { ... metric_name -> float ... },
          "status": "ok" | "warn" | "alert",
          "max_drift": 0.0-1.0
        }
    """
    if ref_profile.get("modality") != new_profile.get("modality"):
        raise ValueError("Cannot compare profiles from different modalities")

    drift_scores: Dict[str, float] = {}

    modality = ref_profile["modality"]
    if modality == "text":
        # Sequence length JS distance
        ref_hist = np.array(ref_profile["seq_length_hist"], dtype=float)
        new_hist = np.array(new_profile["seq_length_hist"], dtype=float)
        drift_scores["seq_length_js"] = _js_distance(ref_hist, new_hist)

        # Token frequency overlap (top-100)
        ref_tokens = set(ref_profile.get("top_tokens", []))
        new_tokens = set(new_profile.get("top_tokens", []))
        if ref_tokens:
            overlap = len(ref_tokens & new_tokens) / float(len(ref_tokens))
        else:
            overlap = 1.0
        drift_scores["token_overlap"] = float(overlap)

    elif modality == "vision":
        # Channel mean distance (Euclidean)
        ref_means = np.array(ref_profile["channel_means"], dtype=float)
        new_means = np.array(new_profile["channel_means"], dtype=float)
        mean_dist = float(np.linalg.norm(ref_means - new_means))
        drift_scores["channel_mean_distance"] = mean_dist

        # Brightness JS distance
        ref_hist = np.array(ref_profile["brightness_hist"], dtype=float)
        new_hist = np.array(new_profile["brightness_hist"], dtype=float)
        drift_scores["brightness_js"] = _js_distance(ref_hist, new_hist)

    # Output distribution drift when available
    if "output_hist" in ref_profile and "output_hist" in new_profile:
        ref_counts = np.array(ref_profile["output_hist"], dtype=float)
        new_counts = np.array(new_profile["output_hist"], dtype=float)
        # Align class order if possible
        if "output_classes" in ref_profile and "output_classes" in new_profile:
            ref_classes = list(ref_profile["output_classes"])
            new_classes = list(new_profile["output_classes"])
            if ref_classes != new_classes:
                # Reindex new_counts to reference class order when possible
                mapping = {c: i for i, c in enumerate(new_classes)}
                aligned = np.zeros_like(ref_counts)
                for idx, c in enumerate(ref_classes):
                    j = mapping.get(c)
                    if j is not None:
                        aligned[idx] = new_counts[j]
                new_counts = aligned

        p = ref_counts + _EPS
        q = new_counts + _EPS
        p /= p.sum()
        q /= q.sum()
        # KL divergence and JS distance for outputs
        drift_scores["output_kl"] = float(entropy(p, q))
        drift_scores["output_js"] = _js_distance(p, q)

    # Status classification uses JS-style distances / bounded metrics.
    if drift_scores:
        # For overlap, larger is better — use (1 - overlap) as "distance".
        distances: List[float] = []
        for name, value in drift_scores.items():
            if name == "token_overlap":
                distances.append(float(max(0.0, 1.0 - value)))
            elif name.endswith("_distance") or name.endswith("_kl"):
                # Approximate normalization for unbounded metrics:
                distances.append(float(min(1.0, value)))
            else:
                distances.append(float(value))
        max_drift = max(distances)
    else:
        max_drift = 0.0

    if max_drift > 0.2:
        status = "alert"
    elif max_drift > 0.1:
        status = "warn"
    else:
        status = "ok"

    return {
        "drift_scores": drift_scores,
        "status": status,
        "max_drift": max_drift,
    }


def log_profile_to_db(
    db: Any,
    run_id: int,
    profile: Dict[str, Any],
    profile_name: str = "dataset_profile",
) -> None:
    """
    Store a profile inside ExperimentDB as a JSON artifact metadata blob.

    This keeps the DB-side responsibility minimal: the profile is embedded in
    the artifact metadata and can be retrieved later via get_artifacts().
    """
    try:
        from pathlib import Path

        # Use a descriptive pseudo-path; the JSON payload is kept in metadata.
        pseudo_path = f"profile:{profile_name}"
        db.log_artifact(
            run_id=run_id,
            artifact_type="profile",
            filepath=pseudo_path,
            metadata={"profile": profile},
        )
    except Exception:
        # Drift metrics are optional; avoid crashing on logging failures.
        return


__all__ = [
    "compute_dataset_profile",
    "compare_profiles",
    "log_profile_to_db",
    "_js_distance",
]

