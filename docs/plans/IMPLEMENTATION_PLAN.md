# Implementation Plan: Production Colab Template Rebuild

**Parent Design:** [2025-01-11-complete-rebuild-design.md](./2025-01-11-complete-rebuild-design.md)
**Start Date:** 2025-01-11
**Target Completion:** 2025-03-01 (8 weeks)
**Current Phase:** Phase 1 - Foundation & Critical Fixes

## Quick Reference

**Current Sprint:** Week 1 (Foundation)
**Next Milestone:** Core infrastructure complete (2025-01-18)
**Blockers:** None

## Phase 1: Foundation & Critical Fixes (Weeks 1-2)

### Week 1: Core Infrastructure

#### Task 1.1: Dependency Management
**Priority:** P0 (Critical)
**Estimated Time:** 2 hours
**Assignee:** Implementation team

**Subtasks:**
- [ ] Create `requirements-colab.txt` with pinned versions
  - Pin numpy==1.26.4 (critical)
  - Pin torch==2.1.2, transformers==4.36.2
  - Pin pytorch-lightning==2.1.0
  - Add all dependencies from design doc
- [ ] Update `template.ipynb` Cell 2 with new installation strategy
  - Upgrade pip first
  - Install numpy separately
  - Install from requirements file
  - Add verification step
- [ ] Test in fresh Colab runtime
  - Verify no dependency conflicts
  - Check import success for all packages
  - Document any version incompatibilities

**Success Criteria:**
- ✓ Cell 2 executes without errors
- ✓ No dependency resolver warnings
- ✓ All imports successful

**Files Modified:**
- `requirements-colab.txt` (NEW)
- `template.ipynb` (Cell 2)

---

#### Task 1.2: Package Structure
**Priority:** P0 (Critical)
**Estimated Time:** 1 hour
**Assignee:** Implementation team

**Subtasks:**
- [ ] Create `utils/__init__.py` with proper exports
  - Import all public classes
  - Define `__all__` list
  - Add version string
  - Add docstring
- [ ] Update Cell 3 in `template.ipynb` for package download
  - Use git clone with depth 1
  - Copy utils/ directory structure
  - Add sys.path.insert for imports
  - Verify package structure
- [ ] Test imports in Colab
  - Test: `from utils import UniversalModelAdapter`
  - Test: `from utils.tokenization import AdaptiveTokenizer`
  - Test: `from utils.ui import SetupWizard`

**Success Criteria:**
- ✓ utils/ is recognized as Python package
- ✓ No ModuleNotFoundError for utils imports
- ✓ All submodules importable

**Files Modified:**
- `utils/__init__.py` (NEW)
- `template.ipynb` (Cell 3)

---

#### Task 1.3: Model Signature Inspector
**Priority:** P0 (Critical)
**Estimated Time:** 4 hours
**Assignee:** Implementation team
**Dependencies:** Task 1.2

**Subtasks:**
- [ ] Create `utils/adapters/__init__.py`
- [ ] Implement `ModelSignatureInspector` class in `utils/adapters/model_adapter.py`
  - `__init__(model)`: Extract signature using inspect module
  - `get_parameters()`: Return list of parameter names
  - `get_required_params()`: Filter required (no default) params
  - `requires_intermediate_outputs()`: Check for mhsa_/residual_/ffn_ prefixes
  - `is_simple_signature()`: Check if only input_ids/attention_mask
- [ ] Write unit tests in `tests/test_model_adapter.py`
  - Test with simple model: `forward(input_ids)`
  - Test with complex model: `forward(input_0_tokens, mhsa_0_output, ...)`
  - Test with attention_mask: `forward(input_ids, attention_mask)`
  - Test parameter extraction accuracy
- [ ] Add docstrings and type hints

**Success Criteria:**
- ✓ Correctly identifies simple vs complex signatures
- ✓ All unit tests pass
- ✓ Works with real generated model from platform

**Files Created:**
- `utils/adapters/__init__.py`
- `utils/adapters/model_adapter.py` (partial, ~100 lines)
- `tests/test_model_adapter.py` (partial, ~50 lines)

**Code Skeleton:**
```python
class ModelSignatureInspector:
    """Analyzes model forward() signature using inspect module"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.signature = inspect.signature(model.forward)
        self.params = list(self.signature.parameters.keys())

    def get_parameters(self) -> List[str]:
        """Return all parameter names"""
        return self.params

    def get_required_params(self) -> List[str]:
        """Return required parameters (no defaults)"""
        return [
            p for p in self.params
            if self.signature.parameters[p].default == inspect.Parameter.empty
        ]

    def requires_intermediate_outputs(self) -> bool:
        """Check if signature needs computed intermediates"""
        intermediate_prefixes = ('mhsa_', 'residual_', 'ffn_', 'attention_', 'mlp_')
        return any(p.startswith(intermediate_prefixes) for p in self.params)

    def is_simple_signature(self) -> bool:
        """Check if signature is simple (input_ids only or with attention_mask)"""
        return set(self.params) <= {'input_ids', 'attention_mask'}
```

---

#### Task 1.4: Computational Graph Executor
**Priority:** P0 (Critical)
**Estimated Time:** 6 hours
**Assignee:** Implementation team
**Dependencies:** Task 1.3

**Subtasks:**
- [ ] Implement `ComputationalGraphExecutor` class
  - `__init__(model, inspector)`: Initialize with model and inspector
  - `_build_dependency_graph()`: Map intermediate outputs to layer dependencies
  - `_compute_intermediate(name, input_ids, attention_mask)`: Compute single intermediate
  - `forward(input_ids, attention_mask)`: Execute full graph with caching
- [ ] Handle different architecture patterns
  - Attention outputs: mhsa_0_output, attention_0_output
  - Residual connections: residual_0_output, residual_1_output
  - FFN outputs: ffn_0_output, mlp_0_output
- [ ] Add caching for intermediate computations
- [ ] Write integration tests
  - Test with GPT-style architecture
  - Test with BERT-style architecture
  - Test with custom architecture
  - Verify outputs match direct model call

**Success Criteria:**
- ✓ Correctly resolves all intermediate dependencies
- ✓ Produces same output as direct model.forward() call
- ✓ Integration tests pass with 3+ architecture types

**Files Modified:**
- `utils/adapters/model_adapter.py` (+200 lines)
- `tests/test_model_adapter.py` (+100 lines)

**Code Skeleton:**
```python
class ComputationalGraphExecutor:
    """Resolves and computes intermediate dependencies"""

    def __init__(self, model: nn.Module, inspector: ModelSignatureInspector):
        self.model = model
        self.inspector = inspector
        self.intermediate_cache = {}
        self.dependency_graph = self._build_dependency_graph()

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Map each intermediate to its dependencies"""
        # Parse parameter names to build execution order
        # Example: mhsa_0_output depends on input_0_tokens
        #          residual_0_output depends on input_0_tokens + mhsa_0_output
        graph = {}
        # ... implementation
        return graph

    def _compute_intermediate(self, name: str, input_ids: torch.Tensor,
                              attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute a single intermediate output"""
        if name in self.intermediate_cache:
            return self.intermediate_cache[name]

        # Extract layer index and type from name
        # e.g., "mhsa_0_output" → layer=0, type="mhsa"
        # Access model.layers[0].mhsa and compute output

        # Cache result
        self.intermediate_cache[name] = output
        return output

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute model with dependency resolution"""
        self.intermediate_cache = {}  # Reset cache

        # Build kwargs with all required parameters
        kwargs = {}
        for param in self.inspector.get_required_params():
            if param == 'input_ids':
                kwargs['input_ids'] = input_ids
            elif param == 'attention_mask':
                kwargs['attention_mask'] = attention_mask
            else:
                # Compute intermediate
                kwargs[param] = self._compute_intermediate(param, input_ids, attention_mask)

        # Call model with all parameters
        return self.model(**kwargs)
```

---

### Week 2: Model Adapter & Tokenization

#### Task 2.1: Universal Model Adapter
**Priority:** P0 (Critical)
**Estimated Time:** 5 hours
**Assignee:** Implementation team
**Dependencies:** Task 1.4

**Subtasks:**
- [ ] Implement `UniversalModelAdapter` as Lightning module
  - Inherit from `pl.LightningModule`
  - `__init__(model, config, tokenizer, learning_rate)`
  - `forward(input_ids, attention_mask, labels)`: Unified interface
  - `training_step(batch, batch_idx)`: Lightning training step
  - `validation_step(batch, batch_idx)`: Lightning validation step
  - `configure_optimizers()`: AdamW optimizer
- [ ] Add loss computation
  - Cross-entropy for language modeling
  - Handle label smoothing (optional)
- [ ] Add metrics logging
  - Training loss, validation loss
  - Perplexity
- [ ] Write integration tests
  - Test training step execution
  - Test validation step execution
  - Test with real generated models
  - Verify Lightning compatibility

**Success Criteria:**
- ✓ Works with ANY generated model signature
- ✓ Lightning Trainer accepts adapter
- ✓ Training/validation steps execute successfully
- ✓ All Tier 1 tests pass with adapter

**Files Modified:**
- `utils/adapters/model_adapter.py` (+100 lines, total ~400 lines)
- `tests/test_model_adapter.py` (+50 lines)

**Code Skeleton:**
```python
class UniversalModelAdapter(pl.LightningModule):
    """Lightning-compatible wrapper for ANY generated model"""

    def __init__(self, generated_model: nn.Module, config: Any,
                 tokenizer: PreTrainedTokenizer, learning_rate: float = 5e-5):
        super().__init__()
        self.model = generated_model
        self.inspector = ModelSignatureInspector(generated_model)
        self.executor = ComputationalGraphExecutor(generated_model, self.inspector)
        self.config = config
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['generated_model', 'tokenizer'])

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """Unified forward interface"""
        # Use executor if complex signature
        if self.inspector.requires_intermediate_outputs():
            logits = self.executor.forward(input_ids, attention_mask)
        else:
            # Simple signature
            if attention_mask is not None:
                logits = self.model(input_ids, attention_mask=attention_mask)
            else:
                logits = self.model(input_ids)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )

        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx):
        output = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train_loss", output["loss"], prog_bar=True)
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        output = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("val_loss", output["loss"], prog_bar=True)
        # Compute perplexity
        perplexity = torch.exp(output["loss"])
        self.log("val_perplexity", perplexity, prog_bar=True)
        return output["loss"]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
```

---

#### Task 2.2: Adaptive Tokenizer - Detection Logic
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Assignee:** Implementation team

**Subtasks:**
- [ ] Create `utils/tokenization/__init__.py`
- [ ] Implement `AdaptiveTokenizer` class in `utils/tokenization/adaptive_tokenizer.py`
  - Define `KNOWN_TOKENIZERS` mapping (vocab_size → HF model name)
  - `detect_strategy(vocab_size, dataset_size)`: Strategy selection logic
  - `load_or_create(vocab_size, dataset, cache_dir)`: Main entry point
  - Add logging for strategy selection
- [ ] Add support for all major tokenizers
  - GPT-2: 50257
  - LLaMA 2: 32000
  - LLaMA 3: 128000
  - BERT: 30522
  - OPT: 250002
  - Phi-2: 49152
  - Qwen: 100277
- [ ] Write unit tests
  - Test strategy detection for known vocab sizes
  - Test strategy detection for unknown vocab sizes
  - Test with various dataset sizes

**Success Criteria:**
- ✓ Correctly identifies pretrained tokenizers
- ✓ Falls back to BPE training when appropriate
- ✓ Falls back to character-level for small datasets
- ✓ All unit tests pass

**Files Created:**
- `utils/tokenization/__init__.py`
- `utils/tokenization/adaptive_tokenizer.py` (partial, ~200 lines)
- `tests/test_tokenization.py` (partial, ~100 lines)

---

#### Task 2.3: Fast BPE Trainer
**Priority:** P0 (Critical)
**Estimated Time:** 4 hours
**Assignee:** Implementation team
**Dependencies:** Task 2.2

**Subtasks:**
- [ ] Implement `FastBPETrainer` class in `utils/tokenization/bpe_trainer.py`
  - `train_on_dataset(texts, vocab_size, special_tokens, cache_dir)`: Main method
  - Use HuggingFace `tokenizers` library
  - Configure BPE trainer with ByteLevel pre-tokenizer
  - Wrap in `PreTrainedTokenizerFast`
  - Save to cache directory
- [ ] Add progress bar for training
- [ ] Optimize for Colab (memory-efficient)
  - Stream text samples
  - Limit training corpus if needed
- [ ] Write integration tests
  - Test with small dataset (100 samples)
  - Test with medium dataset (1K samples)
  - Test with large dataset (10K samples)
  - Verify vocab_size matches target
  - Verify encoding/decoding works

**Success Criteria:**
- ✓ Trains custom BPE in <2 minutes for 10K samples
- ✓ Generated tokenizer has correct vocab_size
- ✓ Encoding/decoding produces valid results
- ✓ Works on Colab T4 GPU without OOM

**Files Created:**
- `utils/tokenization/bpe_trainer.py` (~300 lines)
- `tests/test_tokenization.py` (+100 lines)

---

#### Task 2.4: Character-Level Tokenizer
**Priority:** P1 (High)
**Estimated Time:** 3 hours
**Assignee:** Implementation team

**Subtasks:**
- [ ] Implement `CharacterLevelTokenizer` class in `utils/tokenization/character_tokenizer.py`
  - `__init__(vocab_size, special_tokens)`: Build character vocab
  - `encode(text, max_length)`: Character-level encoding
  - `decode(token_ids)`: Character-level decoding
  - Handle special tokens (<pad>, <unk>, <s>, </s>)
  - Handle padding and truncation
- [ ] Support ASCII + Unicode characters
- [ ] Write unit tests
  - Test encoding simple text
  - Test decoding token IDs
  - Test special token handling
  - Test padding/truncation
  - Test with Unicode characters

**Success Criteria:**
- ✓ Always produces valid tokenizer (fallback)
- ✓ Handles any text input without errors
- ✓ Special tokens work correctly
- ✓ All unit tests pass

**Files Created:**
- `utils/tokenization/character_tokenizer.py` (~200 lines)
- `tests/test_tokenization.py` (+50 lines)

---

#### Task 2.5: Tokenizer Validator
**Priority:** P1 (High)
**Estimated Time:** 2 hours
**Assignee:** Implementation team
**Dependencies:** Tasks 2.2, 2.3, 2.4

**Subtasks:**
- [ ] Implement `TokenizerValidator` class in `utils/tokenization/validator.py`
  - `validate(tokenizer, expected_vocab_size)`: Main validation method
  - Check vocab_size matches expected
  - Check special tokens present
  - Test encode/decode round-trip
  - Report validation results
- [ ] Add helpful error messages
- [ ] Write unit tests
  - Test with valid tokenizer
  - Test with wrong vocab_size
  - Test with missing special tokens

**Success Criteria:**
- ✓ Catches vocab_size mismatches
- ✓ Catches missing special tokens
- ✓ Provides clear error messages
- ✓ All unit tests pass

**Files Created:**
- `utils/tokenization/validator.py` (~100 lines)
- `tests/test_tokenization.py` (+50 lines)

---

#### Task 2.6: Complete Adaptive Tokenizer
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Assignee:** Implementation team
**Dependencies:** Tasks 2.2, 2.3, 2.4, 2.5

**Subtasks:**
- [ ] Complete `load_or_create()` method in `adaptive_tokenizer.py`
  - Integrate all 3 tiers (pretrained, BPE, character)
  - Add tier 4: user upload (optional)
  - Call validator after creation
  - Handle errors gracefully
- [ ] Add caching support
  - Cache pretrained downloads
  - Cache trained BPE tokenizers
- [ ] Write end-to-end tests
  - Test all 4 tiers
  - Test with real vocab sizes from platform
  - Test error handling

**Success Criteria:**
- ✓ Works for ANY vocab_size
- ✓ Selects optimal strategy automatically
- ✓ All tiers functional
- ✓ End-to-end tests pass

**Files Modified:**
- `utils/tokenization/adaptive_tokenizer.py` (+300 lines, total ~500 lines)
- `tests/test_tokenization.py` (+100 lines, total ~300 lines)

---

#### Task 2.7: Lightning DataModule
**Priority:** P0 (Critical)
**Estimated Time:** 3 hours
**Assignee:** Implementation team
**Dependencies:** Task 2.6

**Subtasks:**
- [ ] Implement `AdaptiveTokenizerDataModule` in `utils/tokenization/data_module.py`
  - Inherit from `pl.LightningDataModule`
  - `__init__(dataset, tokenizer, batch_size, max_length)`
  - `setup(stage)`: Tokenize dataset and split train/val
  - `train_dataloader()`: Return training DataLoader
  - `val_dataloader()`: Return validation DataLoader
- [ ] Handle batching and padding
- [ ] Add data augmentation (optional)
- [ ] Write integration tests
  - Test with small dataset
  - Test with Lightning Trainer
  - Verify batch format correct

**Success Criteria:**
- ✓ Lightning Trainer accepts DataModule
- ✓ Batches have correct format (input_ids, attention_mask, labels)
- ✓ Train/val split works correctly
- ✓ Integration tests pass

**Files Created:**
- `utils/tokenization/data_module.py` (~200 lines)
- `tests/test_tokenization.py` (+50 lines)

---

### Week 2 Final Task: Integration Testing
**Priority:** P0 (Critical)
**Estimated Time:** 4 hours
**Assignee:** Implementation team
**Dependencies:** All Week 1-2 tasks

**Subtasks:**
- [ ] Write end-to-end integration test
  - Load real generated model from platform
  - Create adapter with UniversalModelAdapter
  - Create tokenizer with AdaptiveTokenizer
  - Create DataModule
  - Create Lightning Trainer
  - Run 1 epoch of training
  - Verify success
- [ ] Test in fresh Colab runtime
  - Verify all dependencies install correctly
  - Verify package imports work
  - Run integration test
- [ ] Update Tier 1 tests with `_safe_get_model_output`
  - Modify tier1_critical_validation.py
  - Add helper function
  - Update all test functions
  - Run full Tier 1 suite
  - Verify 100% pass rate

**Success Criteria:**
- ✓ End-to-end test passes in Colab
- ✓ All Tier 1 tests pass (100% vs 0% currently)
- ✓ No errors in fresh runtime
- ✓ Training completes successfully

**Files Modified:**
- `utils/tier1_critical_validation.py` (+50 lines)
- `tests/test_integration.py` (NEW, ~150 lines)

---

## Phase 2: Training Pipeline (Weeks 3-4)

### Week 3: Lightning Integration
[Detailed tasks to be added when Phase 1 completes]

High-level tasks:
- Task 3.1: Dataset Loader (HuggingFace, upload, example)
- Task 3.2: Dataset Uploader for Colab
- Task 3.3: Checkpoint Manager with Google Drive
- Task 3.4: Training Coordinator core implementation

### Week 4: Training Features
[Detailed tasks to be added]

High-level tasks:
- Task 4.1: Live training dashboard
- Task 4.2: Early stopping and LR scheduling
- Task 4.3: Checkpoint resumption logic
- Task 4.4: Training integration tests

---

## Phase 3: User Experience & Export (Weeks 5-6)

### Week 5: Setup Wizard
[Detailed tasks to be added]

High-level tasks:
- Task 5.1: SetupWizard base class
- Task 5.2: Step 1 - Model Validation UI
- Task 5.3: Step 2 - Dataset Selection UI
- Task 5.4: Step 3 - Tokenizer Setup UI
- Task 5.5: Step 4 - Training Config UI
- Task 5.6: Step 5 - Confirmation UI
- Task 5.7: Wire all steps with state management

### Week 6: Export & Production
[Detailed tasks to be added]

High-level tasks:
- Task 6.1: ONNX Exporter with validation
- Task 6.2: TorchScript Exporter
- Task 6.3: Quantization support
- Task 6.4: Model Card Generator
- Task 6.5: Export validation tests

---

## Phase 4: Testing & Documentation (Weeks 7-8)

### Week 7: Testing
[Detailed tasks to be added]

High-level tasks:
- Task 7.1: Update Tier 2 tests
- Task 7.2: Comprehensive test suite
- Task 7.3: End-to-end integration tests
- Task 7.4: Performance benchmarks

### Week 8: Documentation & Polish
[Detailed tasks to be added]

High-level tasks:
- Task 8.1: Restructure template.ipynb
- Task 8.2: Write TRAINING_GUIDE.md
- Task 8.3: Write DEPLOYMENT_GUIDE.md
- Task 8.4: Write TROUBLESHOOTING.md
- Task 8.5: Write PLATFORM_RECOMMENDATIONS.md
- Task 8.6: Final testing and UAT

---

## Progress Tracking

### Phase 1 Progress: 0% Complete (0/14 tasks)

**Week 1:**
- [ ] Task 1.1: Dependency Management
- [ ] Task 1.2: Package Structure
- [ ] Task 1.3: Model Signature Inspector
- [ ] Task 1.4: Computational Graph Executor

**Week 2:**
- [ ] Task 2.1: Universal Model Adapter
- [ ] Task 2.2: Adaptive Tokenizer - Detection Logic
- [ ] Task 2.3: Fast BPE Trainer
- [ ] Task 2.4: Character-Level Tokenizer
- [ ] Task 2.5: Tokenizer Validator
- [ ] Task 2.6: Complete Adaptive Tokenizer
- [ ] Task 2.7: Lightning DataModule
- [ ] Integration Testing

### Velocity Tracking
- **Week 1 Planned:** 4 tasks (13 hours)
- **Week 1 Actual:** TBD
- **Week 2 Planned:** 8 tasks (25 hours)
- **Week 2 Actual:** TBD

---

## Daily Standups

### 2025-01-11 (Day 1)
**Completed:**
- Design document approved and committed
- Implementation plan created

**In Progress:**
- Ready to begin Task 1.1 (Dependency Management)

**Blockers:**
- None

**Next:**
- Create requirements-colab.txt
- Update template.ipynb Cell 2
- Test in fresh Colab runtime

---

## Notes & Decisions

### Architecture Decisions
1. **PyTorch Lightning:** Selected for production training framework
2. **4-Tier Tokenization:** Covers all vocab_size scenarios
3. **Wizard-First UX:** Progressive disclosure for new users
4. **Google Drive Checkpoints:** Handle Colab 90-min timeout

### Technical Debt
- None yet (greenfield implementation)

### Known Issues
- Current: 100% test failure rate (will be fixed in Phase 1)
- Current: Dependency conflicts (will be fixed in Task 1.1)
- Current: Import errors (will be fixed in Task 1.2)

---

**Last Updated:** 2025-01-11
**Next Review:** 2025-01-18 (End of Week 1)
