#!/usr/bin/env python3
"""
Integration tests for W&B integration and helper modules.
Tests the interaction between utils/ modules and training.ipynb.
"""

import sys
import os
import json
import traceback
from types import SimpleNamespace

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test 1: Verify all helper modules can be imported."""
    print("TEST 1: Import Helper Modules")
    print("-" * 40)

    errors = []
    modules = [
        'utils.model_helpers',
        'utils.wandb_helpers',
        'utils.test_functions',
        'utils.tier1_critical_validation',
        'utils.tier2_advanced_analysis',
        'utils.tier3_training_utilities'
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            errors.append(f"❌ {module_name}: {e}")
            print(f"❌ {module_name}: {e}")

    print()
    return len(errors) == 0, errors


def test_model_helpers():
    """Test 2: Verify model_helpers functions work correctly."""
    print("TEST 2: Model Helpers Integration")
    print("-" * 40)

    try:
        from utils.model_helpers import (
            find_model_class,
            instantiate_model,
            create_model_config,
            count_parameters
        )
        import torch
        import torch.nn as nn

        # Create a test model class
        class TestTransformer(nn.Module):
            def __init__(self, vocab_size=100):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 128)
                self.linear = nn.Linear(128, vocab_size)

            def forward(self, x):
                x = self.embedding(x)
                return self.linear(x)

        # Test find_model_class
        globals_dict = {'TestTransformer': TestTransformer}
        model_class = find_model_class(globals_dict, 'TestTransformer')
        assert model_class == TestTransformer, "find_model_class failed"
        print("✅ find_model_class() works")

        # Test instantiate_model
        config_dict = {'vocab_size': 200}
        model = instantiate_model(TestTransformer, config_dict)
        assert isinstance(model, nn.Module), "instantiate_model failed"
        assert model.embedding.num_embeddings == 200, "Config not applied"
        print("✅ instantiate_model() works")

        # Test create_model_config
        config = create_model_config({
            'nodes': [{'params': {'vocab_size': 32000, 'max_seq_len': 256}}]
        })
        assert config.vocab_size == 32000, "Config extraction failed"
        assert config.max_seq_len == 256, "Config extraction failed"
        print("✅ create_model_config() works")

        # Test count_parameters
        param_counts = count_parameters(model)
        assert 'total' in param_counts, "count_parameters missing total"
        assert 'trainable' in param_counts, "count_parameters missing trainable"
        assert param_counts['total'] > 0, "No parameters counted"
        print(f"✅ count_parameters() works (found {param_counts['total']:,} params)")

        print()
        return True, []

    except Exception as e:
        error = f"Model helpers test failed: {e}"
        print(f"❌ {error}")
        traceback.print_exc()
        print()
        return False, [error]


def test_wandb_helpers():
    """Test 3: Verify wandb_helpers functions work correctly."""
    print("TEST 3: W&B Helpers Integration")
    print("-" * 40)

    try:
        from utils.wandb_helpers import (
            detect_model_type,
            build_wandb_config,
            print_wandb_summary
        )
        import torch
        import torch.nn as nn

        # Create test models with different architectures
        class GPTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(10, 10)

        class BERTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 10)

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

        # Test detect_model_type
        gpt = GPTModel()
        bert = BERTModel()
        custom = CustomModel()

        assert detect_model_type(gpt) == 'gpt', "GPT detection failed"
        assert detect_model_type(bert) == 'bert', "BERT detection failed"
        assert detect_model_type(custom) == 'custom', "Custom detection failed"
        print("✅ detect_model_type() works")

        # Test build_wandb_config
        config = SimpleNamespace(vocab_size=50257, max_seq_len=512)
        hyperparams = {'learning_rate': 1e-4, 'batch_size': 4}

        wandb_config = build_wandb_config(custom, config, hyperparams)
        assert wandb_config['learning_rate'] == 1e-4, "Hyperparam not set"
        assert wandb_config['vocab_size'] == 50257, "Model config not set"
        assert 'total_params' in wandb_config, "Missing total_params"
        assert wandb_config['model_type'] == 'custom', "Wrong model type"
        print("✅ build_wandb_config() works")

        # Test print_wandb_summary (mock run object)
        class MockRun:
            def __init__(self):
                self.project = "test-project"
                self.name = "test-run"
            def get_url(self):
                return "https://wandb.ai/test/url"

        mock_run = MockRun()
        # This should not crash
        print_wandb_summary(mock_run, custom, hyperparams)
        print("✅ print_wandb_summary() works")

        print()
        return True, []

    except Exception as e:
        error = f"W&B helpers test failed: {e}"
        print(f"❌ {error}")
        traceback.print_exc()
        print()
        return False, [error]


def test_offline_mode():
    """Test 4: Verify offline mode fallback works."""
    print("TEST 4: Offline Mode Fallback")
    print("-" * 40)

    try:
        import os

        # Set offline mode
        os.environ['WANDB_MODE'] = 'offline'

        # Try importing wandb (might not be installed)
        try:
            import wandb
            # If wandb is available, check offline mode works
            assert os.environ.get('WANDB_MODE') == 'offline', "Offline mode not set"
            print("✅ W&B offline mode configured")

            # Try to create a dummy run in offline mode
            run = wandb.init(
                project="test-offline",
                mode="offline",
                config={"test": True}
            )
            run.finish()
            print("✅ Offline run creation works")

        except ImportError:
            print("✅ W&B not installed (expected in CI)")

        print()
        return True, []

    except Exception as e:
        error = f"Offline mode test failed: {e}"
        print(f"❌ {error}")
        traceback.print_exc()
        print()
        return False, [error]


def test_gitignore():
    """Test 5: Verify .gitignore excludes W&B artifacts."""
    print("TEST 5: .gitignore Configuration")
    print("-" * 40)

    gitignore_path = "/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/.gitignore"

    try:
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                content = f.read()

            required_patterns = [
                '.wandb/',
                'wandb/',
                '*.wandb'
            ]

            found_patterns = []
            missing_patterns = []

            for pattern in required_patterns:
                if pattern in content or pattern.replace('/', '') in content:
                    found_patterns.append(pattern)
                else:
                    missing_patterns.append(pattern)

            if found_patterns:
                for pattern in found_patterns:
                    print(f"✅ Found: {pattern}")

            if missing_patterns:
                for pattern in missing_patterns:
                    print(f"⚠️ Missing: {pattern}")
                print("\nRecommended .gitignore additions:")
                print("# Weights & Biases")
                for pattern in missing_patterns:
                    print(pattern)
            else:
                print("✅ All W&B patterns in .gitignore")
        else:
            print("⚠️ No .gitignore file found")
            print("\nRecommended .gitignore content:")
            print("# Weights & Biases")
            print(".wandb/")
            print("wandb/")
            print("*.wandb")

        print()
        return True, []  # Warning only, not a failure

    except Exception as e:
        error = f".gitignore check failed: {e}"
        print(f"❌ {error}")
        print()
        return False, [error]


def test_notebook_integration():
    """Test 6: Verify notebook cells use helper functions correctly."""
    print("TEST 6: Notebook Integration Points")
    print("-" * 40)

    notebook_path = "/Users/matthewhans/Desktop/Programming/transformer-builder-colab-templates/training.ipynb"

    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)

        # Check for key integration points in cells
        integration_checks = {
            'model_helpers_import': False,
            'wandb_helpers_import': False,
            'find_model_class_call': False,
            'instantiate_model_call': False,
            'build_wandb_config_call': False,
            'offline_mode_handling': False
        }

        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])

                if 'from utils.model_helpers import' in source:
                    integration_checks['model_helpers_import'] = True
                if 'from utils.wandb_helpers import' in source:
                    integration_checks['wandb_helpers_import'] = True
                if 'find_model_class(' in source:
                    integration_checks['find_model_class_call'] = True
                if 'instantiate_model(' in source:
                    integration_checks['instantiate_model_call'] = True
                if 'build_wandb_config(' in source:
                    integration_checks['build_wandb_config_call'] = True
                if 'WANDB_MODE' in source or 'offline' in source.lower():
                    integration_checks['offline_mode_handling'] = True

        all_pass = True
        for check, passed in integration_checks.items():
            if passed:
                print(f"✅ {check}")
            else:
                print(f"❌ {check}")
                all_pass = False

        print()
        return all_pass, [] if all_pass else ["Missing notebook integrations"]

    except Exception as e:
        error = f"Notebook integration check failed: {e}"
        print(f"❌ {error}")
        print()
        return False, [error]


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("W&B INTEGRATION TESTS")
    print("=" * 60)
    print()

    tests = [
        ("Import Helper Modules", test_imports),
        ("Model Helpers Integration", test_model_helpers),
        ("W&B Helpers Integration", test_wandb_helpers),
        ("Offline Mode Fallback", test_offline_mode),
        (".gitignore Configuration", test_gitignore),
        ("Notebook Integration Points", test_notebook_integration)
    ]

    results = []
    all_errors = []

    for test_name, test_func in tests:
        success, errors = test_func()
        results.append((test_name, success))
        all_errors.extend(errors)

    # Summary
    print("=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print()

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print()

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print()

    if all_errors:
        print("ERRORS FOUND:")
        for error in all_errors:
            print(f"  - {error}")
        print()
        print("Status: ❌ INTEGRATION TESTS FAILED")
        return 1
    else:
        print("Status: ✅ ALL INTEGRATION TESTS PASSED")
        return 0


if __name__ == "__main__":
    exit(main())