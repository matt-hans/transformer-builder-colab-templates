"""
Tests for training configuration versioning system.

This test suite validates the TrainingConfig dataclass and utilities for
saving, loading, and comparing training configurations. Tests follow TDD
principles with meaningful scenarios covering requirements, edge cases, and
error handling.

Test categories:
- Config creation and defaults
- Validation (valid and invalid inputs)
- Save/load persistence
- Config comparison and diffing
- Integration with W&B
- Edge cases (optional fields, corrupted files)
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import the module we're testing
# (Will fail initially - that's expected in TDD)
from utils.training.training_config import (
    TrainingConfig,
    compare_configs,
)


class TestConfigCreation:
    """Tests for TrainingConfig instantiation and defaults."""

    def test_config_creation_with_defaults(self):
        """
        Test: Create config with default values
        Why: Validates default configuration is complete and sensible
        Contract: Config object created with all required fields
        """
        config = TrainingConfig()

        # Verify core hyperparameters have sensible defaults
        assert config.learning_rate == 5e-5
        assert config.batch_size == 4
        assert config.epochs == 10
        assert config.random_seed == 42

        # Verify model architecture defaults
        assert config.vocab_size == 50257  # GPT-2 default
        assert config.max_seq_len == 128
        assert config.d_model == 768
        assert config.num_layers == 12
        assert config.num_heads == 12

        # Verify metadata fields exist
        assert hasattr(config, 'created_at')
        assert hasattr(config, 'config_version')
        assert config.config_version == "1.0"

    def test_config_creation_with_custom_values(self):
        """
        Test: Create config with custom hyperparameters
        Why: Users need to specify their own values
        Contract: All custom values stored correctly
        """
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            epochs=20,
            vocab_size=32000,
            d_model=512,
            num_layers=6,
            num_heads=8,
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.epochs == 20
        assert config.vocab_size == 32000
        assert config.d_model == 512
        assert config.num_layers == 6
        assert config.num_heads == 8


class TestConfigValidation:
    """Tests for configuration validation (green and red paths)."""

    def test_validation_passes_valid_config(self):
        """
        Test: Valid config passes validation
        Why: Ensures validation accepts correct configurations
        Contract: validate() returns True without exceptions
        """
        config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
            d_model=768,
            num_heads=12,  # 768 % 12 = 0, valid
        )

        # Should not raise
        result = config.validate()
        assert result is True

    def test_validation_negative_learning_rate(self):
        """
        Test: Config with negative learning rate raises error
        Why: Catch invalid hyperparameters early, prevent training failures
        Contract: ValueError with message "learning_rate must be positive"
        """
        config = TrainingConfig(learning_rate=-0.001)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "learning_rate must be positive" in str(exc_info.value)

    def test_validation_zero_learning_rate(self):
        """
        Test: Learning rate of 0 is invalid
        Why: Zero learning rate means no training
        Contract: ValueError raised
        """
        config = TrainingConfig(learning_rate=0.0)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "learning_rate must be positive" in str(exc_info.value)

    def test_validation_invalid_batch_size_zero(self):
        """
        Test: Batch size of 0 raises error
        Why: Prevent training failures from empty batches
        Contract: ValueError with message "batch_size must be >= 1"
        """
        config = TrainingConfig(batch_size=0)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "batch_size must be >= 1" in str(exc_info.value)

    def test_validation_invalid_batch_size_negative(self):
        """
        Test: Negative batch size raises error
        Why: Nonsensical value should be caught
        Contract: ValueError raised
        """
        config = TrainingConfig(batch_size=-1)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "batch_size must be >= 1" in str(exc_info.value)

    def test_validation_invalid_epochs(self):
        """
        Test: Zero or negative epochs raises error
        Why: Must train for at least 1 epoch
        Contract: ValueError with message "epochs must be >= 1"
        """
        config = TrainingConfig(epochs=0)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "epochs must be >= 1" in str(exc_info.value)

    def test_validation_warmup_ratio_out_of_range(self):
        """
        Test: Warmup ratio outside [0, 1] raises error
        Why: Warmup ratio is a percentage
        Contract: ValueError with message about valid range
        """
        # Test > 1
        config = TrainingConfig(warmup_ratio=1.5)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "warmup_ratio must be in [0, 1]" in str(exc_info.value)

        # Test < 0
        config = TrainingConfig(warmup_ratio=-0.1)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "warmup_ratio must be in [0, 1]" in str(exc_info.value)

    def test_validation_d_model_not_divisible_by_heads(self):
        """
        Test: d_model not divisible by num_heads raises error
        Why: Transformer architecture requirement (head_dim = d_model / num_heads)
        Contract: ValueError mentioning divisibility requirement
        """
        config = TrainingConfig(
            d_model=768,
            num_heads=5,  # 768 % 5 != 0
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        assert "d_model" in error_msg
        assert "divisible by num_heads" in error_msg
        assert "768" in error_msg
        assert "5" in error_msg

    def test_validation_invalid_vocab_size(self):
        """
        Test: Vocab size < 1 raises error
        Why: Must have at least one token
        Contract: ValueError raised
        """
        config = TrainingConfig(vocab_size=0)

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "vocab_size must be >= 1" in str(exc_info.value)

    def test_validation_invalid_validation_split(self):
        """
        Test: Validation split outside [0, 0.5] raises error
        Why: Unrealistic to use >50% for validation
        Contract: ValueError with valid range
        """
        # Test > 0.5
        config = TrainingConfig(validation_split=0.8)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "validation_split must be in [0, 0.5]" in str(exc_info.value)

        # Test < 0
        config = TrainingConfig(validation_split=-0.1)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "validation_split must be in [0, 0.5]" in str(exc_info.value)


class TestConfigSaveLoad:
    """Tests for saving and loading configurations."""

    def test_config_save_and_load(self):
        """
        Test: Save config to JSON, load it back, verify equality
        Why: Core requirement - config persistence and reproduction
        Contract: Loaded config equals original config
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                learning_rate=1e-4,
                batch_size=8,
                epochs=20,
                vocab_size=32000,
                d_model=512,
                num_layers=6,
                notes="Test experiment",
            )

            # Save config
            save_path = os.path.join(tmpdir, "test_config.json")
            returned_path = config.save(save_path)

            assert returned_path == save_path
            assert os.path.exists(save_path)

            # Load config
            loaded_config = TrainingConfig.load(save_path)

            # Verify all hyperparameters match
            assert loaded_config.learning_rate == config.learning_rate
            assert loaded_config.batch_size == config.batch_size
            assert loaded_config.epochs == config.epochs
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.d_model == config.d_model
            assert loaded_config.num_layers == config.num_layers
            assert loaded_config.notes == config.notes
            assert loaded_config.random_seed == config.random_seed

    def test_config_save_auto_generated_filename(self):
        """
        Test: Auto-generated filename has timestamp
        Why: Prevent overwrites, track config evolution
        Contract: Filename matches pattern config_YYYYMMDD_HHMMSS.json
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory for auto-generated file
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                config = TrainingConfig()

                # Save without specifying path (auto-generate)
                save_path = config.save()

                # Verify file exists
                assert os.path.exists(save_path)

                # Verify filename pattern: config_YYYYMMDD_HHMMSS.json
                filename = os.path.basename(save_path)
                assert filename.startswith("config_")
                assert filename.endswith(".json")

                # Extract timestamp part
                timestamp_part = filename[7:-5]  # Remove "config_" and ".json"
                assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS

                # Verify it's a valid timestamp
                datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")

            finally:
                os.chdir(original_cwd)

    def test_config_save_creates_valid_json(self):
        """
        Test: Saved config is valid JSON with correct structure
        Why: Ensure serialization produces parseable output
        Contract: JSON file can be read and has expected fields
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                learning_rate=5e-5,
                batch_size=4,
                notes="JSON validation test",
            )

            save_path = os.path.join(tmpdir, "valid.json")
            config.save(save_path)

            # Load as raw JSON
            with open(save_path, 'r') as f:
                data = json.load(f)

            # Verify structure
            assert isinstance(data, dict)
            assert 'learning_rate' in data
            assert 'batch_size' in data
            assert 'epochs' in data
            assert 'random_seed' in data
            assert 'vocab_size' in data
            assert 'created_at' in data
            assert 'config_version' in data
            assert 'notes' in data

            # Verify values
            assert data['learning_rate'] == 5e-5
            assert data['batch_size'] == 4
            assert data['notes'] == "JSON validation test"

    def test_load_nonexistent_file(self):
        """
        Test: Attempt to load from missing file raises FileNotFoundError
        Why: Graceful error handling for user mistakes
        Contract: FileNotFoundError with clear message
        """
        with pytest.raises(FileNotFoundError):
            TrainingConfig.load("/nonexistent/path/config.json")

    def test_load_corrupted_json(self):
        """
        Test: Load from invalid JSON raises appropriate error
        Why: Handle manual edits gracefully
        Contract: ValueError with helpful message about JSON corruption
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted JSON file
            corrupted_path = os.path.join(tmpdir, "corrupted.json")
            with open(corrupted_path, 'w') as f:
                f.write("{ invalid json content }")

            # Attempt to load - should raise ValueError with JSON error details
            with pytest.raises(ValueError, match="Invalid JSON"):
                TrainingConfig.load(corrupted_path)


class TestConfigToDict:
    """Tests for dictionary conversion."""

    def test_config_to_dict(self):
        """
        Test: Convert config to dictionary
        Why: Required for W&B integration
        Contract: Returns dict with all fields
        """
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            epochs=20,
        )

        config_dict = config.to_dict()

        # Verify it's a dict
        assert isinstance(config_dict, dict)

        # Verify key fields present
        assert 'learning_rate' in config_dict
        assert 'batch_size' in config_dict
        assert 'epochs' in config_dict
        assert 'random_seed' in config_dict
        assert 'vocab_size' in config_dict
        assert 'created_at' in config_dict

        # Verify values
        assert config_dict['learning_rate'] == 1e-4
        assert config_dict['batch_size'] == 8
        assert config_dict['epochs'] == 20


class TestConfigComparison:
    """Tests for comparing configurations."""

    def test_compare_configs_no_diff(self):
        """
        Test: Compare identical configs
        Why: Baseline for diff tool
        Contract: Returns empty changed/added/removed dicts
        """
        config1 = TrainingConfig(
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
        )

        config2 = TrainingConfig(
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
        )

        diff = compare_configs(config1, config2)

        # No differences (metadata fields like created_at are skipped)
        assert len(diff['changed']) == 0
        assert len(diff['added']) == 0
        assert len(diff['removed']) == 0

    def test_compare_configs_with_changes(self):
        """
        Test: Compare configs with different hyperparameters
        Why: Core diff functionality
        Contract: Returns dict with changed fields showing old -> new
        """
        config1 = TrainingConfig(
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
        )

        config2 = TrainingConfig(
            learning_rate=1e-4,  # Changed
            batch_size=8,        # Changed
            epochs=10,           # Same
        )

        diff = compare_configs(config1, config2)

        # Verify changed fields
        assert 'learning_rate' in diff['changed']
        assert 'batch_size' in diff['changed']

        # Verify old -> new values
        assert diff['changed']['learning_rate'] == (5e-5, 1e-4)
        assert diff['changed']['batch_size'] == (4, 8)

        # Epochs unchanged
        assert 'epochs' not in diff['changed']

    def test_compare_configs_skips_metadata_fields(self):
        """
        Test: Comparison skips metadata fields like created_at, run_name
        Why: These fields are expected to differ between runs
        Contract: Metadata not included in diff
        """
        config1 = TrainingConfig(
            learning_rate=5e-5,
            run_name="run-1",
        )

        config2 = TrainingConfig(
            learning_rate=5e-5,
            run_name="run-2",  # Different run name
        )

        diff = compare_configs(config1, config2)

        # run_name should be skipped
        assert 'run_name' not in diff['changed']
        assert 'created_at' not in diff['changed']


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_config_with_optional_fields_none(self):
        """
        Test: Config with optional fields set to None
        Why: Ensure optional fields handled correctly
        Contract: Config created successfully, validation passes
        """
        config = TrainingConfig(
            wandb_entity=None,
            dataset_subset=None,
            max_train_samples=None,
            max_val_samples=None,
            run_name=None,
        )

        # Should not raise
        config.validate()

        # Verify None values preserved
        assert config.wandb_entity is None
        assert config.dataset_subset is None
        assert config.max_train_samples is None

    def test_config_roundtrip_preserves_types(self):
        """
        Test: Save/load roundtrip preserves data types
        Why: Ensure no type coercion issues (e.g., int -> float)
        Contract: Types match after load
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                learning_rate=5e-5,      # float
                batch_size=4,            # int
                deterministic=False,     # bool
                notes="test",            # str
            )

            save_path = os.path.join(tmpdir, "types.json")
            config.save(save_path)
            loaded = TrainingConfig.load(save_path)

            # Verify types preserved
            assert isinstance(loaded.learning_rate, float)
            assert isinstance(loaded.batch_size, int)
            assert isinstance(loaded.deterministic, bool)
            assert isinstance(loaded.notes, str)

    def test_validation_multiple_errors_reported(self):
        """
        Test: Multiple validation errors reported together
        Why: User should see all issues at once, not one-by-one
        Contract: ValueError contains multiple error messages
        """
        config = TrainingConfig(
            learning_rate=-0.001,  # Invalid
            batch_size=0,          # Invalid
            epochs=0,              # Invalid
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_msg = str(exc_info.value)

        # All three errors should be in the message
        assert "learning_rate must be positive" in error_msg
        assert "batch_size must be >= 1" in error_msg
        assert "epochs must be >= 1" in error_msg
