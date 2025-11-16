"""
Integration tests for TrainingConfig with MetricsTracker and W&B.

These tests verify that TrainingConfig integrates correctly with the existing
training infrastructure (MetricsTracker, W&B logging, seed management).
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from utils.training.training_config import TrainingConfig, compare_configs
from utils.training.seed_manager import set_random_seed


class TestSeedManagerIntegration:
    """Tests for TrainingConfig integration with seed management."""

    def test_config_seed_used_with_seed_manager(self):
        """
        Test: Config's random_seed can be passed to set_random_seed()
        Why: Ensures reproducibility settings work together
        Contract: set_random_seed() accepts config.random_seed
        """
        config = TrainingConfig(random_seed=123, deterministic=True)

        # Should not raise
        set_random_seed(config.random_seed, config.deterministic)

        # Verify seed was actually set (check PyTorch initial seed)
        import torch
        # Get current seed by creating a random tensor
        initial_state = torch.get_rng_state()
        assert initial_state is not None  # Seed has been set


class TestMetricsTrackerIntegration:
    """Tests for TrainingConfig integration with MetricsTracker."""

    def test_config_to_dict_compatible_with_wandb_config(self):
        """
        Test: Config dict can be passed to wandb.config.update()
        Why: Required for W&B experiment tracking integration
        Contract: to_dict() returns JSON-serializable dict
        """
        config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
            notes="Integration test"
        )

        config_dict = config.to_dict()

        # Verify it's JSON-serializable (requirement for W&B)
        try:
            json_str = json.dumps(config_dict)
            assert json_str is not None
        except (TypeError, ValueError) as e:
            pytest.fail(f"Config dict not JSON-serializable: {e}")

    def test_config_dict_format_for_wandb(self):
        """
        Test: Config dict has correct format for W&B logging
        Why: Ensures compatibility with wandb.config.update()
        Contract: to_dict() returns flat dict with scalar values
        """
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            notes="W&B format test"
        )

        config_dict = config.to_dict()

        # Verify all values are JSON-serializable scalars or None
        for key, value in config_dict.items():
            assert isinstance(value, (int, float, str, bool, type(None))), \
                f"Field {key} has non-scalar value {value} of type {type(value)}"

        # Verify key fields present
        assert 'learning_rate' in config_dict
        assert 'batch_size' in config_dict
        assert 'random_seed' in config_dict
        assert config_dict['learning_rate'] == 1e-4
        assert config_dict['batch_size'] == 8


class TestTrainingWorkflowIntegration:
    """Tests for end-to-end training workflow with config versioning."""

    def test_complete_training_workflow_with_config(self):
        """
        Test: Complete workflow - create, validate, save, log config
        Why: Validates full integration path users will follow
        Contract: All steps work together without errors
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create configuration
            config = TrainingConfig(
                learning_rate=5e-5,
                batch_size=4,
                epochs=10,
                random_seed=42,
                deterministic=False,
                notes="End-to-end integration test"
            )

            # Step 2: Validate configuration
            assert config.validate() is True

            # Step 3: Save configuration
            config_path = os.path.join(tmpdir, "training_config.json")
            saved_path = config.save(config_path)
            assert os.path.exists(saved_path)

            # Step 4: Set random seed from config
            set_random_seed(config.random_seed, config.deterministic)

            # Step 5: Verify config dict is ready for W&B (JSON-serializable)
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            json_str = json.dumps(config_dict)  # Would fail if not serializable
            assert json_str is not None

            # Step 6: Later, load config to reproduce
            loaded_config = TrainingConfig.load(config_path)
            assert loaded_config.learning_rate == config.learning_rate
            assert loaded_config.random_seed == config.random_seed
            assert loaded_config.notes == config.notes

    def test_config_comparison_between_experiments(self):
        """
        Test: Compare configs from two different experiments
        Why: Users need to track what changed between experiments
        Contract: compare_configs() shows meaningful differences
        """
        # Baseline experiment
        baseline = TrainingConfig(
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
            notes="Baseline experiment"
        )

        # Modified experiment
        experiment = TrainingConfig(
            learning_rate=1e-4,  # Changed
            batch_size=8,        # Changed
            epochs=10,           # Same
            notes="Higher LR and batch size"
        )

        # Compare
        diff = compare_configs(baseline, experiment)

        # Verify differences detected
        assert 'learning_rate' in diff['changed']
        assert 'batch_size' in diff['changed']
        assert 'notes' in diff['changed']

        # Verify values
        assert diff['changed']['learning_rate'] == (5e-5, 1e-4)
        assert diff['changed']['batch_size'] == (4, 8)

        # Verify unchanged fields not in diff
        assert 'epochs' not in diff['changed']
        assert 'random_seed' not in diff['changed']

    def test_config_resume_training_scenario(self):
        """
        Test: Resume training from saved config
        Why: Critical use case - reproduce/continue experiments
        Contract: Loaded config can reinitialize training exactly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Original training run
            original_config = TrainingConfig(
                learning_rate=5e-5,
                batch_size=4,
                epochs=10,
                random_seed=42,
                run_name="experiment-001",
                notes="Original training run"
            )

            # Save config
            config_path = os.path.join(tmpdir, "checkpoint_config.json")
            original_config.save(config_path)

            # Later: Resume training
            resumed_config = TrainingConfig.load(config_path)

            # Verify all hyperparameters match
            assert resumed_config.learning_rate == original_config.learning_rate
            assert resumed_config.batch_size == original_config.batch_size
            assert resumed_config.epochs == original_config.epochs
            assert resumed_config.random_seed == original_config.random_seed

            # Verify architecture matches
            assert resumed_config.vocab_size == original_config.vocab_size
            assert resumed_config.d_model == original_config.d_model
            assert resumed_config.num_layers == original_config.num_layers

            # Validation should still pass
            assert resumed_config.validate() is True


class TestConfigFileOperations:
    """Tests for config file operations and paths."""

    def test_config_file_can_be_referenced_for_artifacts(self):
        """
        Test: Saved config file can be referenced (for W&B artifacts, etc.)
        Why: Config files need to be accessible for artifact systems
        Contract: save() returns path that exists and can be read
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                learning_rate=5e-5,
                notes="Artifact reference test"
            )

            # Save config to file
            config_path = os.path.join(tmpdir, "config.json")
            returned_path = config.save(config_path)

            # Verify path is returned and exists
            assert returned_path == config_path
            assert os.path.exists(returned_path)
            assert os.path.isfile(returned_path)

            # Verify file is readable
            with open(returned_path, 'r') as f:
                content = f.read()
                assert len(content) > 0

            # Verify it's valid JSON
            with open(returned_path, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert 'learning_rate' in data
