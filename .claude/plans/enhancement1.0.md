Here’s a concrete, handoff-ready development plan, grounded in your current repo layout and abstractions. I’ll keep everything aligned to the four priorities you listed and tie tasks directly to specific files/modules where possible.

---

# 0. Context & Current State (What We’re Building On)

From the snapshot:

* **Frontends**

  * `template.ipynb` (Tier 1/2 verification), `training.ipynb` (Tier 3 training) 
  * CLI entrypoints: `cli/run_tiers.py`, `cli/run_training.py` 

* **Core abstractions / engine**

  * `TaskSpec`, `TrainingConfig`, `EvalConfig` and friends in `utils/training/` 
  * `ModelAdapter` and task-specific adapters in `utils/adapters/model_adapter.py` 
  * `dataset_utilities.py`, `eval_runner.py`, `export_utilities.py`, `experiment_db.py` under `utils/training/` (dataset, eval, export, experiment tracking). 

* **Verification stack**

  * Tier 1/2: `utils/tier1_critical_validation.py`, `utils/tier2_advanced_analysis.py`, unified facade in `utils/test_functions.py`. 

* **CLI wiring**

  * `cli/run_tiers.py` already wires a stub LM + `DecoderOnlyLMAdapter` + Tier 1 tests. 
  * `cli/run_training.py` wires `TrainingConfig`, `build_task_spec`, `build_eval_config`, `ExperimentDB`, and a stub LM loader. 

This is already a solid text-first platform. The plan below is about elevating it into a **multi-modal, export-ready, distributed, monitored research platform**.

---

# 1. Multimodal Core (Vision End-to-End)

Goal: Generalize `TaskSpec` + `ModelAdapter` + dataset utilities to support non-text modalities, and ship **one full reference vision path** (e.g., CIFAR-10 / tiny image classification) that passes tiers + training + export.

## 1.1 Design decisions

* Keep **TaskSpec as the single point of truth** for task semantics (input/output shape, modality, metrics).
* Keep **ModelAdapter** as the single point of truth for how an arbitrary `nn.Module` is wired to a `TaskSpec`. 
* Extend dataset layer so that `build_dataloader(task_spec, ...)` can handle both text and images.

---

## 1.2 Tasks – Multimodal Extensions

### [MM-01] Extend `TaskSpec` to support modalities

**Files**

* `utils/training/task_spec.py`
* `docs/DEVELOPER_GUIDE_TASKS_EVAL.md` 

**Implementation**

1. Add fields:

   * `modality: Literal["text", "vision", "audio", "tabular"] = "text"`
   * `input_schema: dict[str, Any]` – e.g., `{ "image_size": [3, 224, 224], "channels_first": True }`
   * `output_schema: dict[str, Any]` – e.g., `{ "num_classes": 10 }`
   * Optional `preprocessing_config: dict[str, Any]` (e.g., normalization, augmentations).
2. Introduce canonical **task types**:

   * `task_type: Literal["lm", "seq2seq", "text_classification", "vision_classification", "vision_multilabel"]`
3. Update any factory helpers (`build_task_spec`, etc.) to populate these fields for existing text tasks, preserving backwards compatibility.
4. Update docstrings & static typing to make `TaskSpec` the multi-modal contract.

**Definition of Done**

* Type check passes (`mypy`) for `task_spec.py` & call sites.
* Existing configs (`lm_tiny`, `cls_tiny`, `seq2seq_tiny`) still work without change.
* Docs updated with examples of text and vision `TaskSpec`.

---

### [MM-02] Add a `VisionClassificationAdapter` to `ModelAdapter` family

**Files**

* `utils/adapters/model_adapter.py` 
* Potentially `utils/adapters/__init__.py` for export.

**Implementation**

1. Define a new adapter:

   ```python
   class VisionClassificationAdapter(ModelAdapter):
       task_type = "vision_classification"

       def forward(self, model, batch, task_spec: TaskSpec):
           # Expect batch["pixel_values"] shaped [B, C, H, W]
           logits = model(batch["pixel_values"])
           return {"logits": logits}
       
       def compute_loss(self, outputs, batch, task_spec):
           labels = batch["labels"]
           return F.cross_entropy(outputs["logits"], labels)
       
       def compute_metrics(self, outputs, batch, task_spec):
           preds = outputs["logits"].argmax(dim=-1)
           accuracy = (preds == batch["labels"]).float().mean().item()
           return {"accuracy": accuracy}
   ```

2. Wire into adapter registry or factory pattern used today (if there is one), e.g., `get_adapter_for(task_spec)` that inspects `task_type` / `modality`.

3. Ensure Tier 1/2 tests can still operate on this adapter (for now, shape tests will just call `.forward` with dummy `pixel_values`).

**Definition of Done**

* Unit test: given a dummy CNN model + synthetic batch `[B,3,32,32]`, adapter returns logits and valid loss/metrics.
* No regressions for existing LM adapter paths.

---

### [MM-03] Extend `dataset_utilities.py` with image loaders

**Files**

* `utils/training/dataset_utilities.py` (currently text-centric). 
* `examples/datasets/` (add image sample).

**Implementation**

1. Add tiny image dataset under `examples/datasets/vision/`:

   * `vision_tiny/` with maybe 16–32 PNG/JPEGs and a `labels.csv` or `labels.json`.

2. Implement a dataset builder:

   ```python
   class TinyVisionDataset(Dataset):
       def __init__(..., image_size=(3, 64, 64), transforms=None):
           ...
   ```

3. Add a new branch in `build_dataloader(task_spec: TaskSpec, ...)`:

   ```python
   if task_spec.modality == "vision" and task_spec.task_type == "vision_classification":
       dataset = TinyVisionDataset(...)
       return DataLoader(dataset, batch_size=..., shuffle=..., num_workers=...)
   ```

4. Use torchvision transforms **optionally**, but keep Colab-friendly (no heavy extra deps by default; consider lazy import behind Tier 3 cell like you do for Lightning/Optuna). 

**Definition of Done**

* A `TrainingConfig(task_name="vision_tiny")` + `build_task_spec` + `build_dataloader` yields a working `DataLoader` with keys `{"pixel_values", "labels"}`.
* Tiny vision task can be run in a small training loop in `training.ipynb`.

---

### [MM-04] Add vision eval into `eval_runner.py`

**Files**

* `utils/training/eval_runner.py`
* `utils/adapters/model_adapter.py` (metrics integration).

**Implementation**

1. Generalize `eval_runner.run_eval` to:

   * Accept `task_spec.modality` and route to appropriate metrics set.
2. Implement simple metric sets for vision:

   * Accuracy
   * (Optional) top-k accuracy for k in [3, 5].
3. Ensure `EvalConfig` can point to the vision dataset and that `cli/run_training.py`’s `build_eval_config` path can handle `task_name="vision_tiny"`. 

**Definition of Done**

* A vision model, when trained for 1–2 epochs on `vision_tiny`, can be evaluated via:

  * `training.ipynb` cell,
  * CLI: `python -m cli.run_training --config configs/example_train_vision.json`.

---

### [MM-05] Wire vision into Tier 1/2 notebooks & CLI

**Files**

* `template.ipynb` (optional, minimal addition)
* `cli/run_tiers.py` 
* `docs/USAGE_GUIDE_COLAB_AND_CLI.md`

**Implementation**

1. Add a **vision mode example**:

   * New config JSON: `configs/example_tiers_vision.json` with `task_name="vision_tiny"` and `modality="vision"`.
2. Extend `cli/run_tiers.py` to:

   * If `task_name` is a vision task, instantiate `VisionClassificationAdapter` and a small stub CNN (analogous to `LMStub`).
3. Optional: one notebook cell showcasing vision Tier 1 (shape, gradients) using the same API.

**Definition of Done**

* `python -m cli.run_tiers --config configs/example_tiers_vision.json` produces Tier 1 shape/gradient results for the vision stub.
* Docs include “How to run a vision task” with CLI + notebook steps.

---

# 2. Deployment & Export Tier (Tier 4)

Goal: Turn `export_utilities.py` into a **first-class export tier** with ONNX/TorchScript/quantization, and add Tier-4 tests that load exported models and validate shape, parity, and basic latency.

## 2.1 Tasks – Export + Validation

### [EX-01] Harden `export_utilities` APIs

**Files**

* `utils/training/export_utilities.py` (ONNX & TorchScript exporters already present). 
* `docs/API_REFERENCE.md` (document user-facing API).

**Implementation**

1. Stabilize a public API surface:

   ```python
   def export_model(
       model: nn.Module,
       adapter: ModelAdapter,
       task_spec: TaskSpec,
       export_dir: Path,
       formats: list[str] = ["torchscript", "onnx"],
       quantization: Optional[str] = None,  # e.g. "dynamic", "static"
   ) -> dict:  # paths + metadata
   ```

2. Make sure the exporter uses a consistent **dummy input generator** derived from `TaskSpec` (for text & vision).

3. Persist a **metadata JSON** alongside exports:

   ```json
   {
     "task_type": "vision_classification",
     "modality": "vision",
     "input_shape": [1, 3, 64, 64],
     "output_shape": [1, 10],
     "exported_at": "...",
     "framework_versions": {...}
   }
   ```

4. Make quantization optional and safe:

   * Default off on Colab.
   * Document tradeoffs and supported backends.

**Definition of Done**

* Export API works for LM and vision stubs.
* Test: call `export_model` on both and confirm files + metadata are produced in a temporary directory.

---

### [EX-02] Implement Tier-4 tests (export validation)

**Files**

* New: `utils/training/tier4_export_validation.py`
* `utils/test_functions.py` (to expose a `run_tier4_export_validation` entry). 
* `cli/run_tiers.py` (optional wiring for Tier 4 mode).

**Implementation**

1. Add a test harness that:

   * Trains (or loads) a small model for a given `TaskSpec`.
   * Calls `export_model` for each requested format.
   * Loads the exported artifact and runs:

     * **Shape check** – input/output compatibility.
     * **Numerical parity** – difference between PyTorch and exported model on N random batches:

       * Compute `max_abs_diff` and `relative_error`; define thresholds per format.
     * **Latency microbenchmark** – time a few forward passes.

2. Return a structured result:

   ```python
   {
     "status": "ok" / "warn" / "fail",
     "formats": {
       "onnx": {"status": "ok", "max_abs_diff": 1e-4, "latency_ms": 3.4},
       "torchscript": {...}
     }
   }
   ```

3. Expose via `utils/test_functions.py` to keep the “single façade” pattern consistent with Tiers 1–3.

**Definition of Done**

* Running Tier-4 on LM stub and vision stub:

  * Produces a human-readable summary in notebook.
  * Returns a programmatic JSON for CLI.

---

### [EX-03] Wire export tier into CLI & config

**Files**

* `cli/run_tiers.py`
* New config: `configs/example_tiers_export.json`
* `docs/USAGE_GUIDE_COLAB_AND_CLI.md`

**Implementation**

1. Add a `mode="EXPORT"` or a `tier="4"` mode to `cli/run_tiers.py`:

   * Build stub model + adapter + task_spec as usual.
   * Invoke Tier-4 helper to run export validation.

2. New example config:

   ```json
   {
     "task_name": "lm_tiny",
     "tier": "4",
     "export": {
       "formats": ["torchscript", "onnx"],
       "quantization": "dynamic"
     }
   }
   ```

3. Document usage:

   * “How to validate your exported model” section in docs.

**Definition of Done**

* `python -m cli.run_tiers --config configs/example_tiers_export.json` runs Tier-4 export validation and prints a summary.
* Example config works out-of-the-box on Colab free GPU/CPU.

---

### [EX-04] Minimal serving examples (FastAPI / Gradio)

**Files**

* New: `examples/serving/fastapi_server.py`
* New: `examples/serving/gradio_demo.py`
* `utils/training/export_utilities.py` (load helpers).

**Implementation**

1. Implement a **generic loader**:

   ```python
   def load_exported_model(export_dir: Path, runtime: Literal["torchscript", "onnx"]) -> Callable:
       ...
   ```

2. Build a small FastAPI server:

   * For text LM: `/generate` endpoint that takes a prompt and returns one-step logits or sampled continuation (even if trivial).
   * For vision: `/predict` endpoint that takes base64 image and returns top-k classes.

3. Build a Gradio demo:

   * Simple UI: textbox for LM, image upload for vision.
   * Uses `load_exported_model` to serve exported artifact.

**Definition of Done**

* Both scripts run locally (or in Colab with `ngrok`) and can hit the exported artifacts created by Tier-4.
* README section in `examples/serving/` describes how to run.

---

# 3. Distributed & Large-Scale Training (Lightning DDP/FSDP)

Goal: Surface PyTorch Lightning’s DDP/FSDP options through `TrainingCoordinator` + CLI, and document/test multi-GPU/TPU usage and robust checkpointing.

The docs already mention Lightning + Optuna as Tier 3 optional deps. 

## 3.1 Tasks – Distributed Training

### [DT-01] Introduce `TrainingCoordinator` & Lightning integration

**Files**

* `utils/training/training_core.py` (implementation of `run_training` / `TrainingCoordinator`).
* `docs/ARCHITECTURE_OVERVIEW_v4.0.0.md` (execution engine section). 

**Implementation**

1. Refactor existing training loop (currently invoked from `run_training` + `tier3_training_utilities`) into a `TrainingCoordinator` class:

   ```python
   class TrainingCoordinator:
       def __init__(self, strategy="auto", precision="bf16-mixed", devices=None, num_nodes=1, ...):
           self._trainer = pl.Trainer(...)
       
       def train(self, model, datamodule_or_loaders, task_spec, experiment_db: Optional[ExperimentDB], ...):
           ...
   ```

2. Keep a simple non-Lightning path as fallback for CPU/very small runs (for users who avoid extra deps), but make Lightning the default path for **distributed configs**.

**Definition of Done**

* Existing single-GPU/single-CPU training continues to work via `TrainingCoordinator` with default arguments.
* Unit test: simple LM training for 1 epoch using the new coordinator.

---

### [DT-02] Surface DDP/FSDP options through config & CLI

**Files**

* `utils/training/training_config.py` (or wherever `TrainingConfig` lives).
* `cli/run_training.py` 
* `docs/USAGE_GUIDE_COLAB_AND_CLI.md` (new “Distributed Training” section).

**Implementation**

1. Extend `TrainingConfig` with fields:

   * `strategy: str | None` (e.g., `"ddp"`, `"fsdp_native"`, `"auto"`).
   * `devices: int | list[int] | "auto"`.
   * `num_nodes: int`.
   * `accumulate_grad_batches`, `precision`, etc., mirroring Lightning’s common flags.
2. Parse them in `cli/run_training.py` from JSON config (optional; default to simple single-device behavior).
3. In `TrainingCoordinator.__init__`, translate `TrainingConfig` into `pl.Trainer(...)` arguments.

**Definition of Done**

* Example config: `configs/example_train_ddp.json` showing a DDP config.
* For users with multi-GPU hardware, `python -m cli.run_training --config configs/example_train_ddp.json` launches Lightning with DDP.

---

### [DT-03] Checkpointing, resume, and experiment tracking integration

**Files**

* `utils/training/training_core.py`
* `utils/training/experiment_db.py` 
* `examples/experiment_tracking_example.py` 

**Implementation**

1. Standardize checkpoint directories:

   * e.g., `./checkpoints/{run_name}/{epoch_or_step}/`.
2. Hook into Lightning callbacks (or manual save) to update:

   * `ExperimentDB` with:

     * `run_name`, `strategy`, `devices`, `best_val_metric`, `checkpoint_path`.
3. Implement resume logic in `run_training` / `TrainingCoordinator`:

   * If `cfg.get("resume_from_checkpoint")`, load from that path and continue training.
4. Make sure this works both for text and vision tasks.

**Definition of Done**

* Example training:

  * Run 1: train 1 epoch, checkpoint saved and registered in `ExperimentDB`.
  * Run 2: resume from checkpoint and see that `epoch` increments, metrics appended, same `run_id` or cross-linked.

---

### [DT-04] Multi-GPU/TPU docs and guardrails

**Files**

* `docs/USAGE_GUIDE_COLAB_AND_CLI.md`
* `docs/ML_ENGINEERING_RISK_ANALYSIS.md` (optional note). 

**Implementation**

1. Add a “Distributed Training Guide” section:

   * Explain strategies (`"auto"`, `"ddp"`, `"fsdp_native"`).
   * Include hardware setup notes (e.g., local multi-GPU vs Colab single-GPU).
   * Provide “safe default” configs.
2. Add clear runtime checks:

   * If user sets `strategy="ddp"` but only 1 device is visible, warn and fall back to single device (or error with actionable message).

**Definition of Done**

* Docs show a copy-pasteable example.
* If misconfigured, the CLI prints clear errors instead of cryptic Lightning stack traces.

---

# 4. Monitoring & Drift / Regression Tier (Tier 5)

Goal: Build a lightweight **production-style monitoring tier** that sits on top of `ExperimentDB` and Tier-3/4 artifacts, providing baseline comparisons, drift detection and regression tests.

`experiment_db.py` and the dashboard example already give a strong starting point. 

## 4.1 Tasks – Monitoring Tier

### [MO-01] Extend `ExperimentDB` schema for comparisons

**Files**

* `utils/training/experiment_db.py` 
* `examples/dashboard_demo.py` and `examples/experiment_tracking_example.py`.

**Implementation**

1. Add tables for:

   * `models` or `runs`: per run, store `run_id`, `task_name`, `modality`, `strategy`, `devices`, `artifact_paths` (checkpoint/export).
   * `run_metrics`: `run_id`, `split`, `metric_name`, `value`, `step/epoch`.
   * `comparisons` (optional): `comparison_id`, `baseline_run_id`, `candidate_run_id`, `created_at`.
2. Provide simple helper APIs:

   ```python
   db.register_run(run_info: dict) -> int  # returns run_id
   db.log_metrics(run_id, metrics: dict, split: str, step: int) -> None
   db.create_comparison(baseline_run_id, candidate_run_id) -> int
   ```

**Definition of Done**

* A training run through `cli/run_training.py` logs at least:

  * Training loss/metric per epoch.
  * Validation metrics per epoch.
  * Artifact paths (checkpoint, exported model if available).

---

### [MO-02] Regression testing utility: baseline vs candidate

**Files**

* New: `utils/training/regression_testing.py`
* `utils/training/eval_runner.py`
* `utils/training/experiment_db.py`

**Implementation**

1. Introduce a function:

   ```python
   def compare_models(
       baseline_model, candidate_model,
       adapter: ModelAdapter,
       task_spec: TaskSpec,
       eval_cfg: EvalConfig,
       db: ExperimentDB | None = None,
       comparison_name: str | None = None,
   ) -> dict:
       ...
   ```

2. Behavior:

   * Evaluate both models on the same held-out eval set.
   * Compute metrics; compute deltas `candidate - baseline`.
   * Optionally log a `comparison` record and associated metrics into `ExperimentDB`.

3. Provide CLI glue:

   * E.g., `scripts/run_regression_test.py` with arguments:

     * `--baseline-run-id`, `--candidate-run-id`, `--dataset-id`, etc.

**Definition of Done**

* Example: two LM checkpoints (baseline vs fine-tuned) produce a JSON report:

  ```json
  {
    "metric": "accuracy",
    "baseline": 0.72,
    "candidate": 0.76,
    "delta": 0.04,
    "status": "improved"
  }
  ```

* Same scaffold works for vision accuracy.

---

### [MO-03] Simple drift metrics for inputs & outputs

**Files**

* New: `utils/training/drift_metrics.py`
* `utils/training/experiment_db.py` (storage)
* `examples/experiment_tracking_example.py`

**Implementation**

Start intentionally small and interpretable:

1. **Input drift (text):**

   * Track:

     * Sequence length distribution (`mean`, `std`, histogram buckets).
     * Token frequency top-k changes vs reference window.

2. **Input drift (vision):**

   * Track:

     * Per-channel mean/std.
     * Brightness histogram.

3. **Output drift:**

   * For classification tasks:

     * Predicted class histogram over a window (“prediction distribution”).

4. Implement APIs:

   ```python
   def compute_dataset_profile(dataset, task_spec, sample_size=1000) -> dict:
       ...

   def compare_profiles(ref_profile: dict, new_profile: dict) -> dict:
       # returns per-feature drift scores
   ```

5. Decide on simple thresholds:

   * e.g., KL divergence > X, or Jensen-Shannon > Y => raise `status="alert"`.

**Definition of Done**

* Profile computation runs on tiny datasets in Colab quickly.
* Example notebook demonstrates:

  * Compute a reference profile on train.
  * Compute new profile on “production-like” sample.
  * Show drift summary in a small table/plot.

---

### [MO-04] Tier-5 “Monitoring & Drift” entrypoint

**Files**

* New: `utils/training/tier5_monitoring.py`
* `utils/test_functions.py`
* `cli/run_tiers.py`

**Implementation**

1. Implement Tier-5 entry similar to other tiers:

   ```python
   def run_tier5_monitoring(
       model,
       adapter,
       task_spec,
       eval_cfg,
       db: ExperimentDB,
       baseline_run_id: int | None = None,
   ) -> dict:
       # 1) Evaluate current model; log metrics
       # 2) If baseline_run_id: run comparison
       # 3) Optionally compute drift profile on eval set
       # 4) Return structured summary
   ```

2. Add CLI config:

   * `configs/example_tiers_monitoring.json` with `tier=5` and `baseline_run_id`.

3. Update docs:

   * “Tier-5 Monitoring & Drift” section describing use cases.

**Definition of Done**

* `python -m cli.run_tiers --config configs/example_tiers_monitoring.json`:

  * Calls Tier-5 entry, logs into `ExperimentDB`, and prints a summary report (e.g., “candidate improved accuracy by +3%, no severe drift detected”).

---

# 5. Recommended Execution Order

To keep complexity manageable and respect dependencies:

1. **Multimodal core (Section 1)**

   * Implement [MM-01] → [MM-05].
   * Outcome: text + tiny vision task both run through Tier 1–3 and CLI.

2. **Deployment/export tier (Section 3)**

   * Implement [EX-01] → [EX-04].
   * Outcome: both text & vision can be exported, validated (Tier-4), and minimally served.

3. **Distributed training (Section 2)**

   * Implement [DT-01] → [DT-04], building on existing training loop and ExperimentDB.
   * Outcome: Lightning DDP/FSDP exposed via `TrainingCoordinator` + CLI; checkpointing is robust.

4. **Monitoring tier (Section 4)**

   * Implement [MO-01] → [MO-04].
   * Outcome: `ExperimentDB` becomes the hub for runs, comparisons, and basic drift monitoring (Tier-5).

---

If you’d like, next step I can do is **draft example config files** (`example_train_vision.json`, `example_tiers_export.json`, `example_train_ddp.json`, `example_tiers_monitoring.json`) and some stub code for the new modules so your dev team can almost literally copy-paste to start implementation.
