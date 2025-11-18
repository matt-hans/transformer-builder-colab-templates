---
id: T070
enhancement_id: MM-05
title: Wire Vision Tasks into Tier 1/2 Notebooks and CLI
status: pending
priority: 2
agent: fullstack
dependencies: [T066, T067, T068, T069]
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [multimodal, cli, notebooks, vision, enhancement1.0]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - docs/USAGE_GUIDE_COLAB_AND_CLI.md

est_tokens: 9000
actual_tokens: null
---

## Description

Integrate vision classification tasks into the existing CLI (`cli/run_tiers.py`) and optionally template.ipynb, enabling users to run Tier 1/2 validation tests on vision models. This task creates example config files, adds vision model stubs for testing, and documents the vision workflow.

The implementation adds a vision mode to run_tiers.py that instantiates a dummy CNN (similar to existing LMStub), creates a VisionClassificationAdapter, and runs shape/gradient tests. This validates the end-to-end multimodal infrastructure.

**Technical Approach**: Create `configs/example_tiers_vision.json` config, add a SimpleCNN stub model, extend run_tiers.py with vision task handling, and add documentation. Optionally add a notebook cell demonstrating vision Tier 1 tests.

**Integration Points**:
- `cli/run_tiers.py` (add vision mode)
- `configs/example_tiers_vision.json` (new config)
- `docs/USAGE_GUIDE_COLAB_AND_CLI.md` (vision workflow docs)
- `template.ipynb` (optional vision demo cell)

## Business Context

**User Story**: As a vision model developer, I want to run Tier 1 shape and gradient tests on my CNN using the same CLI as text models, so that I can validate my architecture before training.

**Why This Matters**:
- **Validates multimodal infrastructure**: Proves MM-01 through MM-04 work end-to-end
- **Onboards vision users**: Provides concrete example workflow for vision tasks
- **Enables debugging**: Vision developers can quickly test model changes

**What It Unblocks**:
- Vision training workflows (users can validate models before expensive training)
- Export tier (EX-01 to EX-04 can use vision models for testing)
- Documentation and tutorials for vision tasks

**Priority Justification**: Priority 2 - Completes multimodal foundation but not blocking other tiers; users can manually create vision TaskSpecs without CLI sugar.

## Acceptance Criteria

- [ ] `configs/example_tiers_vision.json` created with vision task configuration
- [ ] `SimpleCNN` stub model added to cli/run_tiers.py or separate stubs.py module
- [ ] `cli/run_tiers.py` extended with vision task handling (detect task_name="vision_tiny", instantiate adapter and stub)
- [ ] Command `python -m cli.run_tiers --config configs/example_tiers_vision.json` runs Tier 1 tests on vision stub
- [ ] Tier 1 shape tests pass for vision model (multiple batch sizes, image resolutions)
- [ ] Tier 1 gradient tests pass (gradients flow to all Conv/Linear layers)
- [ ] Documentation added to `docs/USAGE_GUIDE_COLAB_AND_CLI.md` with "How to Run Vision Tasks" section
- [ ] Optional: template.ipynb cell added demonstrating vision Tier 1 tests (commented out by default)
- [ ] Type hints added to new vision stub code
- [ ] No regressions in text tier tests (existing LM tests still work)

## Test Scenarios

**Test Case 1: CLI Vision Mode**
- Given: `configs/example_tiers_vision.json` with task_name="vision_tiny"
- When: `python -m cli.run_tiers --config configs/example_tiers_vision.json`
- Then: Instantiates SimpleCNN, runs Tier 1 tests, prints shape/gradient report

**Test Case 2: Shape Robustness for Vision**
- Given: SimpleCNN with input [B, 3, 32, 32], output [B, 10]
- When: Tier 1 shape test with batches [(1, 3, 32, 32), (4, 3, 32, 32), (8, 3, 64, 64)]
- Then: All tests pass, outputs have correct [B, 10] shape

**Test Case 3: Gradient Flow Validation**
- Given: SimpleCNN with 3 Conv layers + 2 Linear layers
- When: Tier 1 gradient test runs backward pass
- Then: All 5 layers have non-None gradients, max_gradient > 1e-6

**Test Case 4: Config Parsing**
- Given: example_tiers_vision.json specifies modality="vision", task_type="vision_classification"
- When: run_tiers.py parses config
- Then: Correctly instantiates VisionClassificationAdapter (not LM adapter)

**Test Case 5: Documentation Walkthrough**
- Given: New user follows "How to Run Vision Tasks" guide in docs
- When: They copy-paste commands
- Then: Successfully runs vision tier tests without errors

**Test Case 6: Optional Notebook Cell**
- Given: template.ipynb has commented vision demo cell
- When: User uncomments and runs cell
- Then: Vision Tier 1 tests execute, output displayed

**Test Case 7: Text CLI Regression**
- Given: Existing `configs/example_tiers_lm.json` (if exists)
- When: `python -m cli.run_tiers --config configs/example_tiers_lm.json`
- Then: Text LM tests still work identically to pre-MM-05 behavior

**Test Case 8: Stub Model Architecture**
- Given: SimpleCNN defined with Conv2d → ReLU → AdaptiveAvgPool2d → Linear
- When: Forward pass with [1, 3, 32, 32] input
- Then: Returns logits [1, 10] without errors

## Technical Implementation

**Required Components:**

1. **`configs/example_tiers_vision.json`**
   ```json
   {
     "task_name": "vision_tiny",
     "modality": "vision",
     "task_type": "vision_classification",
     "input_schema": {
       "image_size": [3, 32, 32],
       "channels_first": true
     },
     "output_schema": {
       "num_classes": 10
     },
     "tier": "1",
     "device": "cuda"
   }
   ```

2. **SimpleCNN stub** (in cli/run_tiers.py or utils/stubs.py)
   ```python
   class SimpleCNN(nn.Module):
       """Minimal CNN for vision tier testing."""
       def __init__(self, num_classes: int = 10):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
           self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
           self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           self.pool = nn.AdaptiveAvgPool2d(1)
           self.fc = nn.Linear(64, num_classes)

       def forward(self, x):
           x = torch.relu(self.conv1(x))
           x = torch.relu(self.conv2(x))
           x = torch.relu(self.conv3(x))
           x = self.pool(x).flatten(1)
           return self.fc(x)
   ```

3. **Extend cli/run_tiers.py**
   ```python
   def main(config_path: str):
       config = load_config(config_path)
       task_spec = TaskSpec(**config)

       # Instantiate model based on modality
       if task_spec.modality == "vision":
           num_classes = task_spec.output_schema.get("num_classes", 10)
           model = SimpleCNN(num_classes=num_classes).to(device)
           adapter = VisionClassificationAdapter()
       else:
           # Existing LM stub logic
           model = LMStub(...).to(device)
           adapter = DecoderOnlyLMAdapter()

       # Run tier tests (already modality-agnostic if using adapter)
       run_tier1_tests(model, task_spec, adapter)
   ```

4. **Documentation update**
   ```markdown
   ## How to Run Vision Tasks

   ### CLI Mode

   ```bash
   python -m cli.run_tiers --config configs/example_tiers_vision.json
   ```

   This will:
   1. Instantiate a SimpleCNN stub model
   2. Run Tier 1 shape and gradient tests
   3. Print validation report

   ### Custom Vision Model

   To test your own CNN:
   1. Export model from Transformer Builder (or define in model.py)
   2. Create TaskSpec with modality="vision"
   3. Run tier tests with your model
   ```

**Validation Commands:**

```bash
# Test CLI vision mode
cd /path/to/repo
python -m cli.run_tiers --config configs/example_tiers_vision.json

# Expected output:
# ✓ SimpleCNN instantiated (num_classes=10)
# ✓ VisionClassificationAdapter selected
# Running Tier 1 tests...
# ✓ Shape robustness: 3/3 tests passed
# ✓ Gradient flow: All layers have gradients
# Vision tier tests PASSED
```

## Dependencies

**Hard Dependencies**:
- [T066] Extend TaskSpec to Support Modalities
- [T067] Add VisionClassificationAdapter
- [T068] Extend Dataset Utilities with Image Loaders
- [T069] Add Vision Evaluation to eval_runner

**Soft Dependencies**:
- Tier 1 test functions (already exist, should work with adapters)

**External Dependencies:**
- None (uses PyTorch standard modules)

## Design Decisions

**Decision 1: Create SimpleCNN instead of using pretrained model**
- **Rationale**: No external dependencies (torchvision.models requires extra download), fast instantiation
- **Alternatives**: Use torchvision.models.resnet18 (more realistic but slower)
- **Trade-offs**: Less realistic architecture, but sufficient for infrastructure testing

**Decision 2: Config-driven vision mode (not command-line flag)**
- **Rationale**: Consistent with existing run_tiers.py pattern, easy to extend
- **Alternatives**: Add --vision flag (more explicit but less flexible)
- **Trade-offs**: Requires JSON config creation, but enables complex task specs

**Decision 3: Optional notebook cell instead of mandatory**
- **Rationale**: template.ipynb focuses on text models (original use case); vision is new feature
- **Alternatives**: Make vision cell mandatory (clutters notebook for text users)
- **Trade-offs**: Users must uncomment cell, but keeps notebook lean

**Decision 4: Document CLI workflow, not notebook workflow**
- **Rationale**: CLI is more suitable for vision (developers), notebook is for beginners (text focus)
- **Alternatives**: Prioritize notebook documentation (inconsistent with target users)
- **Trade-offs**: Less accessible to absolute beginners, but matches user personas

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| SimpleCNN too simple, doesn't catch real bugs | M - False confidence in tier tests | M | Document that stub is minimal; encourage users to test real models |
| Tier 1 tests not adapter-aware yet | H - Vision tests fail | M | Update test_shape_robustness to accept optional adapter parameter (minimal change) |
| Config JSON schema not validated | M - Cryptic errors on malformed config | M | Add JSON schema validation or pydantic model for config parsing |
| Documentation unclear for vision beginners | M - Users confused | L | Include complete example with expected output; link to vision_tiny dataset creation |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** Fifth task in multimodal foundation (MM-05 from enhancement1.0.md)
**Dependencies:** T066-T069 (all multimodal foundation tasks)
**Estimated Complexity:** Simple (config + stub model + integration)

## Completion Checklist

**Code Implementation:**
- [ ] `configs/example_tiers_vision.json` created
- [ ] SimpleCNN stub model implemented
- [ ] cli/run_tiers.py extended with vision mode
- [ ] Adapter selection logic added (vision vs text)

**Testing:**
- [ ] CLI command runs successfully with vision config
- [ ] Tier 1 shape tests pass for SimpleCNN
- [ ] Tier 1 gradient tests pass
- [ ] Regression test: text tier tests still work

**Documentation:**
- [ ] `docs/USAGE_GUIDE_COLAB_AND_CLI.md` updated with vision section
- [ ] Example command with expected output documented
- [ ] Config JSON format explained

**Integration:**
- [ ] Works end-to-end with MM-01 through MM-04 components
- [ ] Optional notebook cell added (if time permits)

**Quality Gates:**
- [ ] All 10 acceptance criteria checked
- [ ] All 8 test scenarios validated
- [ ] 4 design decisions documented
- [ ] 4 risks with mitigations
- [ ] Token estimate (9,000) appropriate

**Definition of Done:**
Task is complete when CLI vision mode runs Tier 1 tests successfully, SimpleCNN stub validates correctly, documentation guides users through vision workflow, and text tier tests have no regressions.
