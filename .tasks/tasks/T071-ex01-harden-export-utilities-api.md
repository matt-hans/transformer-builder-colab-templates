---
id: T071
enhancement_id: EX-01
title: Harden export_utilities APIs for Production Multi-Format Export
status: pending
priority: 2
agent: backend
dependencies: [T066]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [export, tier4, deployment, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md

docs_refs:
  - docs/API_REFERENCE.md

est_tokens: 13000
actual_tokens: null
---

## Description

Stabilize and productionize the `export_utilities.py` module to support ONNX, TorchScript, and quantized model exports with comprehensive metadata. This creates a unified `export_model` API that generates dummy inputs from `TaskSpec`, exports to multiple formats, and persists metadata JSON for downstream deployment.

The implementation consolidates existing export functions, adds TaskSpec-aware dummy input generation for text and vision modalities, and implements optional quantization with clear tradeoff documentation.

**Technical Approach**: Define public `export_model()` function, implement modality-specific dummy input generators, create metadata schema, add quantization support with safety guardrails.

## Business Context

**User Story**: As an ML engineer, I want to export my trained model to ONNX and TorchScript formats with one function call, so that I can deploy to production without manual export scripting.

**Why This Matters**:
- **Deployment readiness**: Enables production serving (FastAPI, TorchServe, ONNX Runtime)
- **Format flexibility**: Users choose optimal format for their deployment stack
- **Metadata tracking**: Export manifest documents model specs for ops teams

**What It Unblocks**:
- EX-02: Tier 4 export validation tests
- EX-03: CLI export integration
- EX-04: Serving examples (FastAPI, Gradio)

**Priority Justification**: Priority 2 - Foundational for export tier but not blocking multimodal core.

## Acceptance Criteria

- [ ] `export_model(model, adapter, task_spec, export_dir, formats, quantization)` function created
- [ ] Supports formats: `["torchscript", "onnx", "pytorch"]` (state dict)
- [ ] Dummy input generation works for text (input_ids) and vision (pixel_values) modalities
- [ ] Metadata JSON persisted with task_type, modality, input/output shapes, timestamps, framework versions
- [ ] Optional quantization: `"dynamic"` or `"static"` (default None)
- [ ] Quantization disabled by default on Colab (documented tradeoffs)
- [ ] Type hints for all parameters
- [ ] Docstrings with usage examples for text and vision
- [ ] Unit test: export LM stub to all 3 formats, validate files exist
- [ ] Unit test: export SimpleCNN to ONNX, validate metadata.json schema
- [ ] Returns dict with export paths: `{"torchscript": Path(...), "onnx": Path(...), "metadata": Path(...)}`

## Test Scenarios

**Test Case 1: Multi-Format Export for Text Model**
- Given: Trained LM, task_spec with modality="text"
- When: `export_model(model, adapter, task_spec, "exports/", formats=["torchscript", "onnx"])`
- Then: Creates `exports/model.torchscript`, `exports/model.onnx`, `exports/metadata.json`

**Test Case 2: Vision Model ONNX Export**
- Given: SimpleCNN, task_spec with input_schema {"image_size": [3, 64, 64]}
- When: export_model with formats=["onnx"]
- Then: ONNX file has correct input shape [1, 3, 64, 64], opset_version=14

**Test Case 3: Metadata Schema Validation**
- Given: Exported model metadata.json
- When: Parse JSON
- Then: Contains keys: task_type, modality, input_shape, output_shape, exported_at, torch_version, formats

**Test Case 4: Dynamic Quantization**
- Given: LM model, quantization="dynamic"
- When: export_model called
- Then: TorchScript file is smaller than non-quantized (dynamic quant applied to Linear layers)

**Test Case 5: Dummy Input Generation for Vision**
- Given: task_spec with input_schema {"image_size": [3, 224, 224]}
- When: Dummy input generated
- Then: Returns tensor [1, 3, 224, 224] with random values in [0, 1]

**Test Case 6: Export Directory Creation**
- Given: export_dir="exports/run_42/" does not exist
- When: export_model called
- Then: Creates directory structure, exports files to correct location

**Test Case 7: Quantization Warning on Colab**
- Given: quantization="static" on Colab environment
- When: export_model called
- Then: Logs warning: "Static quantization may fail on Colab GPU, use dynamic instead"

**Test Case 8: Return Value Structure**
- Given: export_model with formats=["onnx", "pytorch"]
- When: Function completes
- Then: Returns `{"onnx": Path("exports/model.onnx"), "pytorch": Path("exports/model.pth"), "metadata": Path("exports/metadata.json")}`

## Technical Implementation

**Required Components:**

1. **`utils/training/export_utilities.py`** (primary API)
   ```python
   def export_model(
       model: nn.Module,
       adapter: ModelAdapter,
       task_spec: TaskSpec,
       export_dir: Path | str,
       formats: list[str] = ["torchscript", "onnx"],
       quantization: Optional[str] = None,
   ) -> dict[str, Path]:
       """
       Export model to multiple formats with metadata.

       Args:
           model: Trained PyTorch model
           adapter: ModelAdapter for dummy input generation
           task_spec: TaskSpec with input/output schemas
           export_dir: Directory to save exports
           formats: List of formats: ["torchscript", "onnx", "pytorch"]
           quantization: "dynamic", "static", or None

       Returns:
           Dict mapping format names to export file paths
       """
       export_dir = Path(export_dir)
       export_dir.mkdir(parents=True, exist_ok=True)

       # Generate dummy inputs
       dummy_input = generate_dummy_input(task_spec)

       # Export to each format
       exported = {}
       if "torchscript" in formats:
           ts_path = export_dir / "model.torchscript"
           export_torchscript(model, dummy_input, ts_path, quantization)
           exported["torchscript"] = ts_path

       if "onnx" in formats:
           onnx_path = export_dir / "model.onnx"
           export_onnx(model, dummy_input, onnx_path)
           exported["onnx"] = onnx_path

       if "pytorch" in formats:
           pt_path = export_dir / "model.pth"
           torch.save(model.state_dict(), pt_path)
           exported["pytorch"] = pt_path

       # Save metadata
       metadata = {
           "task_type": task_spec.task_type,
           "modality": task_spec.modality,
           "input_shape": list(dummy_input.shape),
           "output_shape": infer_output_shape(model, dummy_input),
           "exported_at": datetime.now().isoformat(),
           "torch_version": torch.__version__,
           "formats": formats,
           "quantization": quantization,
       }
       metadata_path = export_dir / "metadata.json"
       with open(metadata_path, "w") as f:
           json.dump(metadata, f, indent=2)
       exported["metadata"] = metadata_path

       return exported


   def generate_dummy_input(task_spec: TaskSpec) -> torch.Tensor:
       """Generate dummy input tensor based on task modality."""
       if task_spec.modality == "text":
           max_seq_len = task_spec.input_schema.get("max_seq_len", 128)
           return torch.randint(0, 50257, (1, max_seq_len))
       elif task_spec.modality == "vision":
           image_size = task_spec.input_schema.get("image_size", [3, 224, 224])
           return torch.randn(1, *image_size)
       else:
           raise ValueError(f"Unsupported modality: {task_spec.modality}")
   ```

2. **Helper functions** (export_torchscript, export_onnx)
   ```python
   def export_torchscript(model, dummy_input, path, quantization=None):
       model.eval()
       if quantization == "dynamic":
           quantized_model = torch.quantization.quantize_dynamic(
               model, {torch.nn.Linear}, dtype=torch.qint8
           )
           traced = torch.jit.trace(quantized_model, dummy_input)
       else:
           traced = torch.jit.trace(model, dummy_input)
       torch.jit.save(traced, path)

   def export_onnx(model, dummy_input, path):
       model.eval()
       torch.onnx.export(
           model,
           dummy_input,
           path,
           input_names=["input"],
           output_names=["output"],
           opset_version=14,
           dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
       )
   ```

3. **Documentation in docs/API_REFERENCE.md**

**Validation Commands:**

```bash
python -c "
from pathlib import Path
import torch.nn as nn
from utils.training.export_utilities import export_model
from utils.adapters.model_adapter import VisionClassificationAdapter
from utils.training.task_spec import TaskSpec

# Dummy model
class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16*30*30, 10)
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.flatten(1))

model = DummyCNN()
task_spec = TaskSpec(
    task_name='vision_export_test',
    modality='vision',
    task_type='vision_classification',
    input_schema={'image_size': [3, 32, 32]},
    output_schema={'num_classes': 10}
)
adapter = VisionClassificationAdapter()

# Export
exports = export_model(
    model, adapter, task_spec,
    export_dir='test_exports',
    formats=['torchscript', 'onnx', 'pytorch']
)

# Validate
assert exports['torchscript'].exists()
assert exports['onnx'].exists()
assert exports['metadata'].exists()
print('✓ Export validation passed')
"
```

## Dependencies

**Hard Dependencies**:
- [T066] Extend TaskSpec to Support Modalities - Provides input_schema for dummy inputs

**External Dependencies:**
- PyTorch 2.6+
- ONNX (if ONNX export used, lazy import)

## Design Decisions

**Decision 1: Single export_model function instead of per-format functions**
- **Rationale**: One-call export to multiple formats reduces boilerplate
- **Alternatives**: Separate export_onnx, export_torchscript functions (more modular but verbose)
- **Trade-offs**: More parameters, but convenience outweighs

**Decision 2: Quantization disabled by default**
- **Rationale**: Quantization can fail on Colab GPU, increases complexity
- **Alternatives**: Enable dynamic quantization by default (faster inference but risky)
- **Trade-offs**: Users must opt-in, but avoids unexpected errors

**Decision 3: Metadata as separate JSON file**
- **Rationale**: Easy to parse for deployment scripts, human-readable
- **Alternatives**: Embed in ONNX metadata (not accessible for TorchScript)
- **Trade-offs**: Extra file, but universal across formats

**Decision 4: TaskSpec-driven dummy input generation**
- **Rationale**: Consistent with architecture-agnostic design, avoids hardcoded shapes
- **Alternatives**: Require user to provide dummy input (more flexible but error-prone)
- **Trade-offs**: Assumes input_schema is correctly populated

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| ONNX export fails for complex models | H - No ONNX support | M | Add try/except, log error, skip ONNX in exports dict; document limitations |
| Quantization breaks model accuracy | H - Deployed model wrong | M | Document quantization as experimental; require explicit opt-in |
| Dummy input shape mismatch | H - Export crashes | M | Validate dummy input shape matches model expected input |
| Metadata schema evolution | M - Incompatible metadata across versions | L | Add schema_version field to metadata.json for forward compatibility |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** First export tier task (EX-01 from enhancement1.0.md)
**Dependencies:** T066 (TaskSpec for dummy inputs)
**Estimated Complexity:** Standard (API design + multi-format export + metadata)

## Completion Checklist

**Code Implementation:**
- [ ] export_model function with 6 parameters created
- [ ] generate_dummy_input supports text and vision
- [ ] export_torchscript, export_onnx helpers implemented
- [ ] Metadata JSON generation with 8+ fields
- [ ] Optional quantization with warnings

**Testing:**
- [ ] Unit test: LM export to all 3 formats
- [ ] Unit test: Vision model export to ONNX
- [ ] Unit test: Metadata schema validation
- [ ] Integration test: Quantized export produces smaller file

**Documentation:**
- [ ] Docstrings with Args/Returns/Examples
- [ ] API_REFERENCE.md updated with export_model
- [ ] Quantization tradeoffs documented

**Quality Gates:**
- [ ] All 11 acceptance criteria checked
- [ ] All 8 test scenarios validated
- [ ] 4 design decisions documented
- [ ] 4 risks with mitigations
- [ ] Token estimate (13,000) appropriate

**Definition of Done:**
Task is complete when export_model works for text and vision models, supports 3 formats, generates valid metadata JSON, and quantization is opt-in with warnings.
