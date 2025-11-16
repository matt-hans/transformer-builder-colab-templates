# Acceptance Criteria Templates

## Standard Acceptance Criteria

### Code Quality
- [ ] Follows PEP 8 style (4-space indentation, snake_case functions, CamelCase classes)
- [ ] Type hints on all public functions
- [ ] Docstrings with Args/Returns/Raises sections
- [ ] No hardcoded values (use config/constants)

### Testing
- [ ] Works in Colab free tier (12GB GPU, tested manually)
- [ ] Handles missing dependencies gracefully (try/except with warnings)
- [ ] Returns structured data (DataFrame or dict, not just prints)
- [ ] Validates inputs (model type, config attributes)

### Documentation
- [ ] Clear markdown cell explaining feature in notebook
- [ ] Example usage code snippet
- [ ] Common errors documented with fixes
- [ ] Token budget impact noted (if context file)

### Integration
- [ ] Backward compatible with existing code
- [ ] Doesn't break zero-installation strategy (template.ipynb)
- [ ] Works with architecture-agnostic design (GPT/BERT/T5)
- [ ] Proper error handling with user-friendly messages

## Validation Commands

### Notebook Validation
```bash
# Manual execution in Colab (no automated testing for notebooks)
# 1. Upload to Colab
# 2. Runtime → Restart runtime
# 3. Run All Cells
# 4. Verify no errors, check outputs
```

### Python Package Validation
```bash
# Local development setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch numpy pandas matplotlib seaborn scipy jupyter

# Run notebook locally
jupyter lab template.ipynb

# Test utilities programmatically
python -c "
from types import SimpleNamespace
from utils.test_functions import test_shape_robustness
config = SimpleNamespace(vocab_size=50257, max_seq_len=128, max_batch_size=8)
# model = ... (needs actual model instance)
print('Import successful')
"
```

### Integration Testing
```bash
# Test with real Transformer Builder export
# 1. Generate model at transformer-builder.com
# 2. Copy Gist ID
# 3. Open template.ipynb in Colab
# 4. Paste Gist ID, run all cells
# 5. Verify Tier 1 + 2 tests pass
```

## Test Scenario Format

Use Given/When/Then format for all test scenarios:

```
Given: [Initial conditions, setup state]
When: [Action performed by user/system]
Then: [Expected outcome, success criteria]
```

Example:
```
Given: User has trained model for 5 epochs with validation loss improving
When: Validation loss increases for 3 consecutive epochs (early stopping patience)
Then: Training stops at epoch 8, best model from epoch 5 restored automatically
```

## Definition of Done

A task is complete when:

1. **Code implemented** and committed with conventional commit message
2. **Tests passing** (manual Colab execution for notebooks, local test for utils)
3. **Documentation added** (markdown cells in notebook or docstrings in code)
4. **Acceptance criteria verified** (all checkboxes checked)
5. **No regressions** (existing tests still pass)
6. **Token budgets met** (context files under limits if applicable)
7. **Peer review** (self-review checklist completed)
