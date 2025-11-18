---
id: T081
enhancement_id: MO-03
title: Simple Drift Metrics for Inputs and Outputs
status: pending
priority: 3
agent: backend
dependencies: [T066, T079]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [monitoring, tier5, drift-detection, mlops, enhancement1.0]

context_refs:
  - context/project.md

est_tokens: 13000
actual_tokens: null
---

## Description

Create `drift_metrics.py` module with lightweight drift detection for text (sequence length distribution, token frequency) and vision (channel mean/std, brightness histogram) inputs, plus output drift (prediction distribution). Enables early detection of data distribution shifts in production.

## Business Context

**User Story**: As a production ML engineer, I want automated alerts when input data shifts significantly from training distribution, so I can retrain models proactively.

**Why This Matters**: Prevents silent model degradation; enables proactive maintenance

**What It Unblocks**: MO-04 (Tier 5 monitoring), production monitoring dashboards

**Priority Justification**: Priority 3 - Valuable for production but not blocking core features

## Acceptance Criteria

- [ ] `utils/training/drift_metrics.py` module created
- [ ] `compute_dataset_profile(dataset, task_spec, sample_size=1000) -> dict` computes reference stats
- [ ] Text input drift: sequence length (mean, std, histogram), token frequency top-100 changes
- [ ] Vision input drift: per-channel mean/std, brightness histogram (5 bins)
- [ ] Output drift: predicted class histogram (for classification tasks)
- [ ] `compare_profiles(ref_profile, new_profile) -> dict` computes drift scores (KL divergence, Jensen-Shannon)
- [ ] Status classification: "ok" (JS < 0.1), "alert" (JS > 0.2), "warn" (0.1-0.2)
- [ ] Works with both text and vision modalities
- [ ] Unit test with synthetic data validates JS divergence calculation
- [ ] Example notebook cell demonstrates drift detection workflow

## Test Scenarios

**Test Case 1: Text Sequence Length Drift**
- Given: Training data avg length=120, production data avg length=200
- When: compare_profiles computes drift
- Then: Returns drift_score > 0.2, status="alert"

**Test Case 2: Vision Channel Mean Shift**
- Given: Training images mean=[0.5, 0.5, 0.5], production mean=[0.6, 0.4, 0.5]
- When: Drift computed for channel 0
- Then: Significant drift detected on red channel

**Test Case 3: Output Distribution Drift**
- Given: Training predictions: class 0 (40%), class 1 (60%); Production: class 0 (80%), class 1 (20%)
- When: Compare output distributions
- Then: High KL divergence (>0.5), status="alert"

**Test Case 4: No Drift Baseline**
- Given: Reference and new profiles identical
- When: compare_profiles runs
- Then: JS divergence=0.0, status="ok"

**Test Case 5: Token Frequency Drift**
- Given: Training top-10 tokens: ["the", "a", "is", ...], Production top-10: ["COVID", "pandemic", "vaccine", ...]
- When: Compute token frequency overlap
- Then: Low overlap (<50%), drift detected

**Test Case 6: Profile Storage in ExperimentDB**
- Given: Computed profile for run_id=1
- When: Store profile as JSON artifact
- Then: Queryable from ExperimentDB for future comparisons

## Technical Implementation

```python
# utils/training/drift_metrics.py
import numpy as np
from scipy.stats import entropy

def compute_dataset_profile(
    dataset: Dataset,
    task_spec: TaskSpec,
    sample_size: int = 1000
) -> dict:
    """Compute statistical profile of dataset."""
    if task_spec.modality == "text":
        return _compute_text_profile(dataset, sample_size)
    elif task_spec.modality == "vision":
        return _compute_vision_profile(dataset, sample_size)

def _compute_text_profile(dataset, sample_size):
    """Text input drift: sequence lengths, token frequencies."""
    lengths = []
    token_counts = defaultdict(int)

    for i, item in enumerate(dataset):
        if i >= sample_size:
            break
        seq = item["input_ids"]
        lengths.append(len(seq))
        for token_id in seq:
            token_counts[token_id] += 1

    # Top-100 tokens
    top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:100]

    return {
        "seq_length_mean": np.mean(lengths),
        "seq_length_std": np.std(lengths),
        "seq_length_hist": np.histogram(lengths, bins=10)[0].tolist(),
        "top_tokens": [tok_id for tok_id, _ in top_tokens],
        "modality": "text"
    }

def _compute_vision_profile(dataset, sample_size):
    """Vision input drift: channel mean/std, brightness histogram."""
    channel_means = [[], [], []]
    brightness_values = []

    for i, item in enumerate(dataset):
        if i >= sample_size:
            break
        img = item["pixel_values"]  # [C, H, W]
        for c in range(3):
            channel_means[c].append(img[c].mean())
        brightness_values.append(img.mean())

    return {
        "channel_means": [np.mean(channel_means[c]) for c in range(3)],
        "channel_stds": [np.std(channel_means[c]) for c in range(3)],
        "brightness_hist": np.histogram(brightness_values, bins=5)[0].tolist(),
        "modality": "vision"
    }

def compare_profiles(ref_profile: dict, new_profile: dict) -> dict:
    """Compare two dataset profiles, return drift scores."""
    if ref_profile["modality"] != new_profile["modality"]:
        raise ValueError("Cannot compare profiles from different modalities")

    drift_scores = {}

    if ref_profile["modality"] == "text":
        # Sequence length drift (Kolmogorov-Smirnov or JS divergence)
        ref_hist = np.array(ref_profile["seq_length_hist"]) + 1e-10  # Smooth
        new_hist = np.array(new_profile["seq_length_hist"]) + 1e-10
        ref_hist /= ref_hist.sum()
        new_hist /= new_hist.sum()
        drift_scores["seq_length_js"] = jensenshannon(ref_hist, new_hist)

        # Token frequency overlap
        ref_tokens = set(ref_profile["top_tokens"])
        new_tokens = set(new_profile["top_tokens"])
        overlap = len(ref_tokens & new_tokens) / 100.0
        drift_scores["token_overlap"] = overlap

    elif ref_profile["modality"] == "vision":
        # Channel mean drift (Euclidean distance)
        ref_means = np.array(ref_profile["channel_means"])
        new_means = np.array(new_profile["channel_means"])
        drift_scores["channel_mean_distance"] = np.linalg.norm(ref_means - new_means)

        # Brightness histogram drift
        ref_hist = np.array(ref_profile["brightness_hist"]) + 1e-10
        new_hist = np.array(new_profile["brightness_hist"]) + 1e-10
        ref_hist /= ref_hist.sum()
        new_hist /= new_hist.sum()
        drift_scores["brightness_js"] = jensenshannon(ref_hist, new_hist)

    # Classify status
    max_drift = max(drift_scores.values())
    if max_drift > 0.2:
        status = "alert"
    elif max_drift > 0.1:
        status = "warn"
    else:
        status = "ok"

    return {
        "drift_scores": drift_scores,
        "status": status,
        "max_drift": max_drift
    }
```

## Dependencies

**Hard Dependencies**:
- [T066] TaskSpec extension - Modality routing
- [T079] ExperimentDB - Profile storage

**External Dependencies:**
- scipy (already in requirements)
- numpy (already in requirements)

## Design Decisions

**Decision 1: Simple metrics (mean/std, histograms) instead of complex ML**
- **Rationale**: Interpretable, fast to compute, no model training
- **Trade-offs**: Less sensitive than learned drift detectors, but good enough

**Decision 2: Jensen-Shannon divergence instead of KL divergence**
- **Rationale**: JS is symmetric, bounded [0, 1], numerically stable
- **Trade-offs**: Less common than KL, but better for practical thresholds

**Decision 3: Sample 1000 examples instead of full dataset**
- **Rationale**: Fast profiling for large datasets, statistically sufficient
- **Trade-offs**: May miss rare patterns, but configurable

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| High variance in small samples | M - Noisy drift scores | M | Recommend sample_size >= 1000; smooth histograms with +epsilon |
| Thresholds not universal | M - Too many false alarms | M | Document threshold tuning; provide conservative defaults |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Third monitoring tier task (MO-03 from enhancement1.0.md)
**Dependencies:** T066 (TaskSpec), T079 (ExperimentDB)
**Estimated Complexity:** Standard (statistical metrics + multi-modality support)

## Completion Checklist

- [ ] drift_metrics.py module created
- [ ] compute_dataset_profile for text and vision
- [ ] compare_profiles with drift scores
- [ ] Status classification logic
- [ ] All 10 acceptance criteria met
- [ ] All 6 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** Drift metrics compute correctly for text/vision, profile comparison works, thresholds classify drift status, unit tests validate JS divergence.
