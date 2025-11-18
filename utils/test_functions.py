"""
Unified import facade for all testing tiers.

This module re-exports all test functions for backward compatibility.
Individual tiers can be imported directly for better modularity:

- tier1_critical_validation: Core validation tests
- tier2_advanced_analysis: Advanced diagnostic tests
- tier3_training_utilities: Training and benchmarking tests

Usage:
    # Import from facade (backward compatible)
    from test_functions import test_shape_robustness, test_gradient_flow

    # Import from tier modules directly
    from tier1_critical_validation import test_shape_robustness
    from tier2_advanced_analysis import test_attention_patterns
    from tier3_training_utilities import test_fine_tuning
"""

# Re-export all functions from tier modules
from .tier1_critical_validation import (
    test_shape_robustness,
    test_gradient_flow,
    test_output_stability,
    test_parameter_initialization,
    test_memory_footprint,
    test_inference_speed,
)

from .tier2_advanced_analysis import (
    test_attention_patterns,
    test_attribution_analysis,
    test_robustness,
)

from .tier3_training_utilities import (
    test_fine_tuning,
    test_hyperparameter_search,
    test_benchmark_comparison,
)
from .training.tier4_export_validation import run_tier4_export_validation

# Import for utility functions
import torch.nn as nn
from typing import Any

__all__ = [
    # Tier 1: Critical Validation
    'test_shape_robustness',
    'test_gradient_flow',
    'test_output_stability',
    'test_parameter_initialization',
    'test_memory_footprint',
    'test_inference_speed',
    # Tier 2: Advanced Analysis
    'test_attention_patterns',
    'test_attribution_analysis',
    'test_robustness',
    # Tier 3: Training Utilities
    'test_fine_tuning',
    'test_hyperparameter_search',
    'test_benchmark_comparison',
    # Tier 4: Export Validation
    'run_tier4_export_validation',
    # Utility functions
    'run_all_tier1_tests',
    'run_all_tier2_tests',
    'run_all_tests',
]


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def run_all_tier1_tests(model: nn.Module, config: Any) -> None:
    """
    Run all Tier 1 tests in sequence.

    Provides a comprehensive validation suite for critical model functionality.
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL TIER 1 TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Shape Robustness", lambda: test_shape_robustness(model, config)),
        ("Gradient Flow", lambda: test_gradient_flow(model, config)),
        ("Output Stability", lambda: test_output_stability(model, config)),
        ("Parameter Initialization", lambda: test_parameter_initialization(model)),
        ("Memory Footprint", lambda: test_memory_footprint(model, config)),
        ("Inference Speed", lambda: test_inference_speed(model, config)),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        try:
            result = test_func()
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
        print()


def run_all_tier2_tests(model: nn.Module, config: Any) -> None:
    """
    Run all Tier 2 tests in sequence.

    Provides advanced analysis of attention patterns, attribution, and robustness.
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL TIER 2 TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Attention Patterns", lambda: test_attention_patterns(model, config)),
        ("Attribution Analysis", lambda: test_attribution_analysis(model, config)),
        ("Robustness Testing", lambda: test_robustness(model, config)),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        try:
            result = test_func()
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
        print()


def run_all_tests(model: nn.Module, config: Any) -> None:
    """
    Run complete test suite (Tier 1 + Tier 2).

    Note: Tier 3 tests require additional setup and are not included here.
    """
    run_all_tier1_tests(model, config)
    run_all_tier2_tests(model, config)
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
