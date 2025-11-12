# Repository Guidelines

This repository provides Colab-ready notebooks and utilities to validate and benchmark Transformer Builder exports.

## Project Structure & Module Organization
- `template.ipynb` — Main Colab template; loads a model/config and runs Tier 1 validation.
- `utils/test_functions.py` — Importable test utilities (shape, gradients, stability, memory, speed).
- `examples/` — Optional example notebooks.
- `README.md`, `LICENSE` — Documentation and licensing.

## Build, Test, and Development Commands
- Create env and install basics:
  `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install torch numpy pandas matplotlib seaborn scipy jupyter`
- Run the notebook locally:
  `jupyter lab template.ipynb`  (or: `jupyter notebook template.ipynb`)
- Use tests in a Python session:
  ```python
  from types import SimpleNamespace
  from utils.test_functions import test_shape_robustness
  model = ...  # your nn.Module
  config = SimpleNamespace(vocab_size=50257, max_seq_len=128, max_batch_size=8)
  print(test_shape_robustness(model, config))
  ```

## Coding Style & Naming Conventions
- Python: PEP 8; 4-space indentation; type hints where practical.
- Names: `snake_case` for functions/variables, `CamelCase` for classes, public test helpers use `test_*` prefix.
- Utils: deterministic, side-effect free functions that return `pandas.DataFrame`/`dict` and accept `model` and `config`.
- Notebooks: clear markdown headings, idempotent cells, minimal hidden state.

## Testing Guidelines
- Primary path: run Tier 1 cells in `template.ipynb` (prints tables/plots).
- Direct usage: import functions from `utils/test_functions.py` as in the snippet above.
- Optional dependencies: some analyses use SciPy; functions should degrade gracefully when unavailable.

## Commit & Pull Request Guidelines
- Use Conventional Commits (observed): `feat:`, `fix:`, `chore:`.
- PRs include: summary, motivation, screenshots/sample outputs for notebook changes, and linked issues.
- Keep diffs focused; avoid committing large datasets or secrets; clear heavy notebook outputs unless needed for documentation.

## Security & Configuration Tips
- The template may fetch code from GitHub Gists; review downloaded code before execution.
- Do not commit credentials; prefer environment variables for tokens/keys.
- For offline/strict runs, rely on the checked-in `utils/test_functions.py` rather than fetching remote files.

