# Development Guide

This guide covers development workflows, best practices, and maintenance procedures for the Transformer Builder Colab Templates project.

## Table of Contents

- [Requirements Management](#requirements-management)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [CI/CD](#cicd)

## Requirements Management

### Overview

The project uses a three-file requirements strategy to support different use cases:

1. **`requirements.txt`** - Local development with exact version pins (`==`)
2. **`requirements-training.txt`** - Training notebook exact pins (`==`)
3. **`requirements-colab-v3.4.0.txt`** - Colab with range pins (`>=`)

### Requirements Sync Validation

All three requirements files are automatically validated by CI to ensure they stay in sync. The validation checks:

- ✅ All packages in `requirements-training.txt` exist in `requirements-colab-v3.4.0.txt` (training section)
- ✅ Version compatibility (exact pins satisfy range pins)
- ✅ All imports in `utils/` are declared in requirements files
- ✅ No missing third-party dependencies

**Validation script:** `scripts/check_requirements_sync.py`

**CI workflow:** `.github/workflows/validate-requirements.yml`

### Updating Requirements

Follow this workflow when adding or updating dependencies:

#### Step 1: Add Package Locally

```bash
# Install the package with exact version
pip install new-package==1.2.3

# Update requirements.txt
pip freeze | grep new-package >> requirements.txt
```

#### Step 2: Update Training Requirements (if needed)

If the package is used in training utilities (Tier 3):

```bash
# Add to requirements-training.txt with exact version
echo "new-package==1.2.3" >> requirements-training.txt
```

#### Step 3: Update Colab Requirements (if needed)

If the package is used in training notebooks:

1. Open `requirements-colab-v3.4.0.txt`
2. Add the package to the **TRAINING.IPYNB - AUTOMATIC INSTALLATION** section
3. Use range pin format: `new-package>=1.2.0`

```txt
# ==============================================================================
# TRAINING.IPYNB - AUTOMATIC INSTALLATION
# ==============================================================================
# Installs in fresh runtime:
pytorch-lightning>=2.4.0
optuna>=3.0.0
torchmetrics>=1.3.0
wandb>=0.15.0
new-package>=1.2.0  # Add new package here
```

#### Step 4: Validate Sync

Run the validation script to ensure all files are in sync:

```bash
python scripts/check_requirements_sync.py
```

Expected output on success:

```
✅ SUCCESS: All requirements files are in sync!

✓ training.txt ⊆ colab.txt
✓ Version compatibility verified
✓ All imports declared
```

#### Step 5: Commit Changes

Commit all three files together:

```bash
git add requirements.txt requirements-training.txt requirements-colab-v3.4.0.txt
git commit -m "feat(deps): add new-package==1.2.3 for XYZ feature"
```

### Handling Validation Errors

#### Error: Missing Package in Colab Training Section

```
❌ Missing packages in requirements-colab training section:
   wandb

   Fix: Add to TRAINING.IPYNB section in requirements-colab-v3.4.0.txt
```

**Solution:** Add the package to the training section in `requirements-colab-v3.4.0.txt`:

```bash
# Edit requirements-colab-v3.4.0.txt
# Add under TRAINING.IPYNB section:
wandb>=0.15.0
```

#### Error: Version Conflict

```
❌ Version conflicts detected:

   Package: torch
   requirements-training.txt: torch==1.8.0
   requirements-colab.txt: torch>=2.0.0

   Fix: Update requirements-training.txt:
   sed -i 's/torch==1.8.0/torch>=2.0.0/' requirements-training.txt
```

**Solution:** Update the exact version in `requirements-training.txt` to satisfy the range:

```bash
# Update to compatible version
sed -i 's/torch==1.8.0/torch==2.0.0/' requirements-training.txt

# Or use the suggested command from error message
sed -i 's/torch==1.8.0/torch>=2.0.0/' requirements-training.txt
```

#### Error: Undeclared Imports

```
❌ Undeclared imports found in utils/:
   captum, requests

   Fix: Add missing packages:
   pip install captum requests
   pip freeze | grep -E '(captum|requests)' >> requirements.txt
```

**Solution:** Add the missing packages to requirements:

```bash
# Install packages
pip install captum requests

# Add to requirements.txt
pip freeze | grep -E '(captum|requests)' >> requirements.txt

# If training-specific, also add to requirements-training.txt
echo "captum==0.7.0" >> requirements-training.txt
```

### Package Categories

The validation script recognizes several package categories:

#### Standard Library (Exempt)

Packages from Python's standard library (e.g., `os`, `sys`, `json`) are automatically excluded from requirements checks.

#### Internal Modules (Exempt)

Project-internal modules (e.g., `utils`, `training`, `tokenization`) are excluded from requirements checks.

#### Optional Packages (Exempt)

Optional dependencies that may not be installed in all environments:

- `google` - Google Colab specific
- `ipython` - Jupyter/IPython specific (provided by jupyter package)
- `pynvml` - Optional GPU monitoring
- `onnx` / `onnxruntime` - Optional model export
- `psutil` - Optional system monitoring
- `captum` - Optional feature attribution (Tier 2 tests)
- `requests` - Optional for Gist loader (has fallback)
- `pytorch_lightning` - Training-specific (in requirements-training.txt)

#### Intentionally Omitted Packages

Packages that are explicitly not included (documented in CLAUDE.md):

- `datasets` - Tests use synthetic data generation
- `huggingface-hub` - Models loaded from Gist, not Hub

### CI/CD Integration

The requirements validation runs automatically:

1. **On push to `main`**: Validates all requirements files
2. **On pull requests**: Blocks merge if validation fails
3. **Path triggers**: Only runs when requirements or utils/ files change

**Workflow file:** `.github/workflows/validate-requirements.yml`

**Exit codes:**
- `0` - All checks pass
- `1` - Validation failures detected

### Best Practices

1. **Always validate before committing**: Run `python scripts/check_requirements_sync.py` locally
2. **Use exact pins in development**: `requirements.txt` should have `==` for reproducibility
3. **Use range pins for Colab**: `requirements-colab-v3.4.0.txt` uses `>=` for flexibility
4. **Document intentional omissions**: Update OPTIONAL_PACKAGES in the validation script
5. **Keep training deps separate**: Training-only packages go in `requirements-training.txt`
6. **Update all three files together**: Commit requirements changes atomically

### Troubleshooting

#### Validation script fails to run

```bash
# Install validation dependencies
pip install packaging

# Run script
python scripts/check_requirements_sync.py
```

#### CI workflow fails but local validation passes

1. Check that you've committed all three requirements files
2. Ensure the validation script is executable: `chmod +x scripts/check_requirements_sync.py`
3. Verify you're using Python 3.10+ (same as CI)

#### False positives for optional packages

If a package is incorrectly flagged as missing but is actually optional:

1. Open `scripts/check_requirements_sync.py`
2. Add the package to `OPTIONAL_PACKAGES` set
3. Commit the change with explanation

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/transformer-builder-colab-templates.git
cd transformer-builder-colab-templates

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utils --cov-report=html

# Run specific test file
pytest tests/test_tier1_validation.py -v
```

---

## Testing

### Test Structure

- **Unit tests**: `tests/unit/`
- **Integration tests**: `tests/integration/`
- **End-to-end tests**: `tests/e2e/`

### Running Validation Checks

```bash
# Requirements sync validation
python scripts/check_requirements_sync.py

# Type checking with mypy
mypy utils/ --config-file mypy.ini

# Code formatting check
black --check utils/

# Linting
flake8 utils/
```

---

## CI/CD

### GitHub Actions Workflows

1. **Requirements Validation** (`.github/workflows/validate-requirements.yml`)
   - Validates requirements files sync
   - Runs on push to main and PRs
   - Blocks merge if validation fails

2. **Test Suite** (`.github/workflows/test.yml`) - _Coming soon_
   - Runs pytest suite
   - Reports coverage
   - Tests multiple Python versions

### Pre-commit Hooks

Install pre-commit hooks for automatic validation:

```bash
# Copy pre-commit hook
cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The pre-commit hook runs:
- Secret scanning (blocks API keys)
- Requirements validation
- Code formatting checks

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for full contribution guidelines.

### Quick Checklist

- [ ] Code follows PEP 8 style guide
- [ ] Type hints added for public functions
- [ ] Tests added/updated for changes
- [ ] Requirements files updated and validated
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow Conventional Commits format

---

## Additional Resources

- [CLAUDE.md](../CLAUDE.md) - Project architecture and conventions
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [USAGE_GUIDE_COLAB_AND_CLI.md](USAGE_GUIDE_COLAB_AND_CLI.md) - User guide
