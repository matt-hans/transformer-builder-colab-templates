"""
Tests for metrics_utils.calculate_perplexity.
"""

import os, importlib.util, math
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mu_path = os.path.join(repo_root, 'utils', 'training', 'metrics_utils.py')
spec = importlib.util.spec_from_file_location('metrics_utils', mu_path)
mu = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mu)  # type: ignore
calculate_perplexity = mu.calculate_perplexity
import math


def test_calculate_perplexity_basic():
    assert math.isclose(calculate_perplexity(0.0), 1.0, rel_tol=1e-9)
    assert math.isclose(calculate_perplexity(1.0), math.e, rel_tol=1e-9)


def test_calculate_perplexity_clipping():
    # Extremely large loss should be clipped to 20
    assert calculate_perplexity(1000) == math.exp(20.0)
    assert calculate_perplexity(20.0) == math.exp(20.0)
