"""
Tests for TrainingConfigBuilder fluent API and presets.

This test suite validates the builder pattern implementation including:
- Fluent API method chaining
- Progressive validation
- Preset factory methods
- Immutability guarantees
- Error handling and edge cases

Test strategy:
- Test each with_* method independently
- Test method chaining
- Test all 5 presets
- Test progressive validation catches errors early
- Test immutability (builder methods return new instances)
"""

import pytest
from typing import Any

from utils.training.training_config import TrainingConfig, TrainingConfigBuilder


class TestBuilderBasics:
    """Tests for basic builder functionality."""

    def test_builder_initialization_empty(self):
        """
        Test: Create empty builder
        Why: Ensure builder can start with no parameters
        Contract: Builder created successfully
        """
        builder = TrainingConfigBuilder()
        assert builder is not None
        assert isinstance(builder._config, dict)

    def test_builder_initialization_with_kwargs(self):
        """
        Test: Create builder with initial values
        Why: Support starting builder with some predefined values
        Contract: Values stored in internal config dict
        """
        builder = TrainingConfigBuilder(learning_rate=1e-4, batch_size=8)
        assert builder._config['learning_rate'] == 1e-4
        assert builder._config['batch_size'] == 8

    def test_builder_immutability(self):
        """
        Test: Builder methods return new instances (immutable pattern)
        Why: Prevent accidental state mutation, enable safe method chaining
        Contract: Each method call returns a different builder instance
        """
        builder1 = TrainingConfigBuilder()
        builder2 = builder1.with_training(learning_rate=1e-4)

        # Different instances
        assert builder1 is not builder2

        # Original unchanged
        assert 'learning_rate' not in builder1._config
        assert builder2._config['learning_rate'] == 1e-4

    def test_builder_method_chaining(self):
        """
        Test: Fluent API supports method chaining
        Why: Core requirement for builder pattern ergonomics
        Contract: Chain multiple methods and call build()
        """
        config = (TrainingConfigBuilder()
            .with_model(d_model=512, num_heads=8)
            .with_training(learning_rate=1e-4, epochs=5)
            .with_optimizer(weight_decay=0.1)
            .build()
        )

        assert isinstance(config, TrainingConfig)
        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.learning_rate == 1e-4
        assert config.epochs == 5
        assert config.weight_decay == 0.1


class TestWithModel:
    """Tests for with_model() method."""

    def test_with_model_valid_params(self):
        """
        Test: Set valid model architecture parameters
        Why: Ensure model config works with valid inputs
        Contract: All parameters stored correctly
        """
        config = (TrainingConfigBuilder()
            .with_model(
                model_name="test-gpt",
                model_type="gpt",
                vocab_size=32000,
                max_seq_len=256,
                d_model=512,
                num_layers=6,
                num_heads=8,
                d_ff=2048,
                dropout=0.1
            )
            .build()
        )

        assert config.model_name == "test-gpt"
        assert config.model_type == "gpt"
        assert config.vocab_size == 32000
        assert config.max_seq_len == 256
        assert config.d_model == 512
        assert config.num_layers == 6
        assert config.num_heads == 8
        assert config.d_ff == 2048
        assert config.dropout == 0.1

    def test_with_model_invalid_vocab_size(self):
        """
        Test: vocab_size < 1 raises ValueError
        Why: Progressive validation should catch errors early
        Contract: ValueError raised immediately in with_model()
        """
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            TrainingConfigBuilder().with_model(vocab_size=0)

    def test_with_model_invalid_d_model_num_heads_divisibility(self):
        """
        Test: d_model not divisible by num_heads raises ValueError
        Why: Transformer architecture requirement
        Contract: ValueError raised with clear message
        """
        with pytest.raises(ValueError, match="d_model.*divisible by num_heads"):
            TrainingConfigBuilder().with_model(d_model=768, num_heads=5)

    def test_with_model_invalid_model_type(self):
        """
        Test: Invalid model_type raises ValueError
        Why: Only support known architecture families
        Contract: ValueError with list of valid types
        """
        with pytest.raises(ValueError, match="Invalid model_type"):
            TrainingConfigBuilder().with_model(model_type="unknown")  # type: ignore

    def test_with_model_invalid_dropout(self):
        """
        Test: dropout outside [0, 1] raises ValueError
        Why: Dropout is a probability
        Contract: ValueError raised
        """
        with pytest.raises(ValueError, match="dropout must be in"):
            TrainingConfigBuilder().with_model(dropout=1.5)


class TestWithTraining:
    """Tests for with_training() method."""

    def test_with_training_valid_params(self):
        """
        Test: Set valid training parameters
        Why: Core training config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_training(
                learning_rate=1e-4,
                batch_size=16,
                epochs=20,
                validation_split=0.2,
                early_stopping_patience=3,
                max_train_samples=1000,
                max_val_samples=100
            )
            .build()
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.epochs == 20
        assert config.validation_split == 0.2
        assert config.early_stopping_patience == 3
        assert config.max_train_samples == 1000
        assert config.max_val_samples == 100

    def test_with_training_invalid_learning_rate(self):
        """
        Test: Negative or zero learning rate raises ValueError
        Why: Learning rate must be positive
        Contract: ValueError raised early
        """
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfigBuilder().with_training(learning_rate=-0.001)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfigBuilder().with_training(learning_rate=0.0)

    def test_with_training_invalid_batch_size(self):
        """
        Test: batch_size < 1 raises ValueError
        Why: Must have at least one sample per batch
        Contract: ValueError raised
        """
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainingConfigBuilder().with_training(batch_size=0)

    def test_with_training_invalid_validation_split(self):
        """
        Test: validation_split outside [0, 0.5] raises ValueError
        Why: Can't use >50% for validation
        Contract: ValueError with range
        """
        with pytest.raises(ValueError, match="validation_split must be in"):
            TrainingConfigBuilder().with_training(validation_split=0.8)


class TestWithOptimizer:
    """Tests for with_optimizer() method."""

    def test_with_optimizer_valid_params(self):
        """
        Test: Set valid optimizer parameters
        Why: Optimizer config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_optimizer(
                weight_decay=0.1,
                max_grad_norm=2.0,
                warmup_ratio=0.2,
                gradient_accumulation_steps=4
            )
            .build()
        )

        assert config.weight_decay == 0.1
        assert config.max_grad_norm == 2.0
        assert config.warmup_ratio == 0.2
        assert config.gradient_accumulation_steps == 4
        # Lightning compatibility field
        assert config.accumulate_grad_batches == 4

    def test_with_optimizer_invalid_weight_decay(self):
        """
        Test: Negative weight_decay raises ValueError
        Why: Weight decay must be non-negative
        Contract: ValueError raised
        """
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            TrainingConfigBuilder().with_optimizer(weight_decay=-0.01)

    def test_with_optimizer_invalid_max_grad_norm(self):
        """
        Test: Non-positive max_grad_norm raises ValueError
        Why: Gradient clipping norm must be positive
        Contract: ValueError raised
        """
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            TrainingConfigBuilder().with_optimizer(max_grad_norm=0.0)

    def test_with_optimizer_invalid_warmup_ratio(self):
        """
        Test: warmup_ratio outside [0, 1] raises ValueError
        Why: Warmup ratio is a fraction
        Contract: ValueError with range
        """
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            TrainingConfigBuilder().with_optimizer(warmup_ratio=1.5)


class TestWithHardware:
    """Tests for with_hardware() method."""

    def test_with_hardware_valid_params(self):
        """
        Test: Set valid hardware parameters
        Why: Hardware config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_hardware(
                use_amp=True,
                compile_mode="default",
                compile_fullgraph=False,
                compile_dynamic=True,
                strategy="ddp",
                devices=4,
                num_nodes=2,
                precision="bf16-mixed"
            )
            .build()
        )

        assert config.use_amp is True
        assert config.compile_mode == "default"
        assert config.compile_fullgraph is False
        assert config.compile_dynamic is True
        assert config.strategy == "ddp"
        assert config.devices == 4
        assert config.num_nodes == 2
        assert config.precision == "bf16-mixed"

    def test_with_hardware_invalid_compile_mode(self):
        """
        Test: Invalid compile_mode raises ValueError
        Why: Only support known compilation modes
        Contract: ValueError with valid modes
        """
        with pytest.raises(ValueError, match="Invalid compile_mode"):
            TrainingConfigBuilder().with_hardware(compile_mode="invalid")

    def test_with_hardware_invalid_strategy(self):
        """
        Test: Invalid strategy raises ValueError
        Why: Only support known distributed strategies
        Contract: ValueError with valid strategies
        """
        with pytest.raises(ValueError, match="Invalid strategy"):
            TrainingConfigBuilder().with_hardware(strategy="invalid")

    def test_with_hardware_invalid_devices_type(self):
        """
        Test: Invalid devices type raises ValueError
        Why: devices must be int, str, or list of ints
        Contract: ValueError with type info
        """
        with pytest.raises(ValueError, match="devices must be"):
            TrainingConfigBuilder().with_hardware(devices=3.14)  # type: ignore

    def test_with_hardware_invalid_precision(self):
        """
        Test: Invalid precision string raises ValueError
        Why: Only support known precision modes
        Contract: ValueError with valid precisions
        """
        with pytest.raises(ValueError, match="Invalid precision"):
            TrainingConfigBuilder().with_hardware(precision="invalid")

    def test_with_hardware_devices_as_list(self):
        """
        Test: devices accepts list of GPU IDs
        Why: Support multi-GPU with specific device selection
        Contract: List stored correctly
        """
        config = (TrainingConfigBuilder()
            .with_hardware(devices=[0, 1, 3])
            .build()
        )
        assert config.devices == [0, 1, 3]


class TestWithLogging:
    """Tests for with_logging() method."""

    def test_with_logging_valid_params(self):
        """
        Test: Set valid logging parameters
        Why: Logging config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_logging(
                wandb_project="my-project",
                wandb_entity="my-team",
                run_name="experiment-1",
                notes="Testing builder"
            )
            .build()
        )

        assert config.wandb_project == "my-project"
        assert config.wandb_entity == "my-team"
        assert config.run_name == "experiment-1"
        assert config.notes == "Testing builder"


class TestWithCheckpointing:
    """Tests for with_checkpointing() method."""

    def test_with_checkpointing_valid_params(self):
        """
        Test: Set valid checkpointing parameters
        Why: Checkpointing config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_checkpointing(
                checkpoint_dir="./ckpts",
                save_every_n_epochs=2,
                keep_best_only=True
            )
            .build()
        )

        assert config.checkpoint_dir == "./ckpts"
        assert config.save_every_n_epochs == 2
        assert config.keep_best_only is True

    def test_with_checkpointing_invalid_save_frequency(self):
        """
        Test: save_every_n_epochs < 1 raises ValueError
        Why: Must save at least every epoch
        Contract: ValueError raised
        """
        with pytest.raises(ValueError, match="save_every_n_epochs must be >= 1"):
            TrainingConfigBuilder().with_checkpointing(save_every_n_epochs=0)


class TestWithExport:
    """Tests for with_export() method."""

    def test_with_export_valid_params(self):
        """
        Test: Set valid export parameters
        Why: Export config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_export(
                export_bundle=True,
                export_formats=["onnx", "torchscript"],
                export_dir="./exports"
            )
            .build()
        )

        assert config.export_bundle is True
        assert config.export_formats == ["onnx", "torchscript"]
        assert config.export_dir == "./exports"

    def test_with_export_invalid_format(self):
        """
        Test: Invalid export format raises ValueError
        Why: Only support known export formats
        Contract: ValueError with valid formats
        """
        with pytest.raises(ValueError, match="Invalid export format"):
            TrainingConfigBuilder().with_export(export_formats=["invalid"])


class TestWithReproducibility:
    """Tests for with_reproducibility() method."""

    def test_with_reproducibility_valid_params(self):
        """
        Test: Set valid reproducibility parameters
        Why: Reproducibility config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_reproducibility(random_seed=123, deterministic=True)
            .build()
        )

        assert config.random_seed == 123
        assert config.deterministic is True


class TestWithDataset:
    """Tests for with_dataset() method."""

    def test_with_dataset_valid_params(self):
        """
        Test: Set valid dataset parameters
        Why: Dataset config works correctly
        Contract: All parameters stored
        """
        config = (TrainingConfigBuilder()
            .with_dataset(
                dataset_name="wikitext-v1",
                dataset_split="train",
                task_name="lm_tiny",
                eval_dataset_id="wikitext-v1"
            )
            .build()
        )

        assert config.dataset_name == "wikitext-v1"
        assert config.dataset_split == "train"
        assert config.task_name == "lm_tiny"
        assert config.eval_dataset_id == "wikitext-v1"


class TestBuilderBuild:
    """Tests for build() method."""

    def test_build_creates_valid_config(self):
        """
        Test: build() creates valid TrainingConfig
        Why: Final config must pass validation
        Contract: Returns TrainingConfig instance
        """
        config = (TrainingConfigBuilder()
            .with_model(d_model=512, num_heads=8)
            .with_training(learning_rate=1e-4, epochs=5)
            .build()
        )

        assert isinstance(config, TrainingConfig)
        # Validation was called internally
        assert config.d_model == 512

    def test_build_runs_full_validation(self):
        """
        Test: build() runs TrainingConfig.validate()
        Why: Catch any issues before returning config
        Contract: Invalid config raises ValueError from validate()
        """
        # Create builder with conflicting params that pass progressive validation
        # but fail TrainingConfig.validate()
        builder = TrainingConfigBuilder(
            d_model=768,
            num_heads=12,
            learning_rate=-0.001  # Invalid but not caught until build()
        )

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            builder.build()


class TestPresetQuickPrototype:
    """Tests for quick_prototype() preset."""

    def test_quick_prototype_creates_valid_config(self):
        """
        Test: quick_prototype() creates valid configuration
        Why: Preset must produce working config
        Contract: Returns builder that builds successfully
        """
        config = TrainingConfigBuilder.quick_prototype().build()

        assert isinstance(config, TrainingConfig)
        assert config.model_name == "quick-prototype"
        assert config.epochs == 3
        assert config.num_layers == 6
        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.compile_mode is None  # No compilation for speed

    def test_quick_prototype_customizable(self):
        """
        Test: quick_prototype() can be customized
        Why: Users should be able to override preset values
        Contract: Method chaining works on preset
        """
        config = (TrainingConfigBuilder.quick_prototype()
            .with_training(epochs=5)
            .with_logging(run_name="custom-quick-test")
            .build()
        )

        assert config.epochs == 5  # Overridden
        assert config.run_name == "custom-quick-test"
        assert config.d_model == 512  # Preset value retained


class TestPresetBaseline:
    """Tests for baseline() preset."""

    def test_baseline_creates_valid_config(self):
        """
        Test: baseline() creates valid configuration
        Why: Preset must produce working config
        Contract: Returns builder that builds successfully
        """
        config = TrainingConfigBuilder.baseline().build()

        assert isinstance(config, TrainingConfig)
        assert config.model_name == "baseline-transformer"
        assert config.epochs == 10
        assert config.num_layers == 12
        assert config.d_model == 768
        assert config.num_heads == 12
        assert config.compile_mode == "default"
        assert config.precision == "bf16-mixed"

    def test_baseline_gpt2_scale(self):
        """
        Test: baseline() uses GPT-2 small scale architecture
        Why: Standard baseline should match known architecture
        Contract: 125M parameter scale (12 layers, 768 dims)
        """
        config = TrainingConfigBuilder.baseline().build()

        assert config.d_model == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.d_ff == 3072
        assert config.vocab_size == 50257  # GPT-2 tokenizer


class TestPresetProduction:
    """Tests for production() preset."""

    def test_production_creates_valid_config(self):
        """
        Test: production() creates valid configuration
        Why: Preset must produce working config
        Contract: Returns builder that builds successfully
        """
        config = TrainingConfigBuilder.production().build()

        assert isinstance(config, TrainingConfig)
        assert config.model_name == "production-transformer"
        assert config.epochs == 20
        assert config.deterministic is True  # Reproducible
        assert config.export_bundle is True  # Generate artifacts
        assert config.compile_mode == "reduce-overhead"

    def test_production_export_enabled(self):
        """
        Test: production() enables export bundle generation
        Why: Production preset should generate deployment artifacts
        Contract: export_bundle=True, formats include onnx and torchscript
        """
        config = TrainingConfigBuilder.production().build()

        assert config.export_bundle is True
        assert "onnx" in config.export_formats
        assert "torchscript" in config.export_formats


class TestPresetDistributed:
    """Tests for distributed() preset."""

    def test_distributed_creates_valid_config(self):
        """
        Test: distributed() creates valid configuration
        Why: Preset must produce working config
        Contract: Returns builder that builds successfully
        """
        config = TrainingConfigBuilder.distributed().build()

        assert isinstance(config, TrainingConfig)
        assert config.model_name == "distributed-transformer"
        assert config.strategy == "ddp"
        assert config.devices == 4
        assert config.num_nodes == 1

    def test_distributed_large_model_scale(self):
        """
        Test: distributed() uses larger model for multi-GPU
        Why: Distributed training typically used for larger models
        Contract: Larger than baseline (1024 dims, 24 layers)
        """
        config = TrainingConfigBuilder.distributed().build()

        assert config.d_model == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.d_ff == 4096

    def test_distributed_customizable_devices(self):
        """
        Test: distributed() devices can be customized
        Why: Support different cluster sizes
        Contract: Can override device count
        """
        config = (TrainingConfigBuilder.distributed()
            .with_hardware(devices=8, strategy="fsdp_native")
            .build()
        )

        assert config.devices == 8
        assert config.strategy == "fsdp_native"


class TestPresetLowMemory:
    """Tests for low_memory() preset."""

    def test_low_memory_creates_valid_config(self):
        """
        Test: low_memory() creates valid configuration
        Why: Preset must produce working config
        Contract: Returns builder that builds successfully
        """
        config = TrainingConfigBuilder.low_memory().build()

        assert isinstance(config, TrainingConfig)
        assert config.model_name == "low-memory-transformer"
        assert config.batch_size == 2  # Very small
        assert config.gradient_accumulation_steps == 8  # Compensate for small batch
        assert config.compile_mode is None  # No compilation (saves memory)

    def test_low_memory_small_model(self):
        """
        Test: low_memory() uses small model architecture
        Why: Fit in limited memory environments
        Contract: Small footprint (384 dims, 6 layers)
        """
        config = TrainingConfigBuilder.low_memory().build()

        assert config.d_model == 384
        assert config.num_layers == 6
        assert config.num_heads == 6
        assert config.d_ff == 1536

    def test_low_memory_dataset_limits(self):
        """
        Test: low_memory() limits dataset size
        Why: Reduce memory footprint
        Contract: max_train_samples and max_val_samples set
        """
        config = TrainingConfigBuilder.low_memory().build()

        assert config.max_train_samples == 10000
        assert config.max_val_samples == 1000
        assert config.keep_best_only is True  # Save storage


class TestPresetComparison:
    """Tests comparing different presets."""

    def test_presets_produce_different_configs(self):
        """
        Test: Each preset produces a distinct configuration
        Why: Presets should serve different use cases
        Contract: Configs differ in key parameters
        """
        quick = TrainingConfigBuilder.quick_prototype().build()
        baseline = TrainingConfigBuilder.baseline().build()
        production = TrainingConfigBuilder.production().build()
        distributed = TrainingConfigBuilder.distributed().build()
        low_mem = TrainingConfigBuilder.low_memory().build()

        # Model sizes differ
        assert quick.d_model < baseline.d_model
        assert low_mem.d_model < baseline.d_model
        assert distributed.d_model > baseline.d_model

        # Epoch counts differ
        assert quick.epochs < baseline.epochs < production.epochs

        # Export enabled only for production
        assert production.export_bundle is True
        assert baseline.export_bundle is False

        # Deterministic mode differs
        assert production.deterministic is True
        assert baseline.deterministic is False


class TestProgressiveValidation:
    """Tests for progressive validation (errors caught early)."""

    def test_progressive_validation_model_divisibility(self):
        """
        Test: d_model/num_heads divisibility checked in with_model()
        Why: Catch errors early, not in build()
        Contract: ValueError raised immediately
        """
        # Error caught in with_model(), not build()
        with pytest.raises(ValueError, match="divisible"):
            TrainingConfigBuilder().with_model(d_model=768, num_heads=5)

    def test_progressive_validation_multiple_calls(self):
        """
        Test: Validation works across multiple method calls
        Why: Ensure validation uses cumulative state
        Contract: Divisibility checked even when d_model and num_heads set separately
        """
        # First call: set d_model
        builder = TrainingConfigBuilder().with_model(d_model=768)

        # Second call: set incompatible num_heads - should fail
        with pytest.raises(ValueError, match="divisible"):
            builder.with_model(num_heads=5)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_builder_partial_config_builds_with_defaults(self):
        """
        Test: Build config with only some fields set (use defaults for rest)
        Why: Builder should merge with TrainingConfig defaults
        Contract: Config created successfully with defaults
        """
        config = (TrainingConfigBuilder()
            .with_training(learning_rate=1e-4)
            .build()
        )

        # Custom value
        assert config.learning_rate == 1e-4

        # Defaults from TrainingConfig
        assert config.batch_size == 4  # Default
        assert config.epochs == 10  # Default

    def test_builder_empty_builds_with_all_defaults(self):
        """
        Test: Empty builder creates config with all defaults
        Why: Should work like TrainingConfig() with no args
        Contract: Config created successfully
        """
        config = TrainingConfigBuilder().build()

        assert isinstance(config, TrainingConfig)
        assert config.learning_rate == 5e-5  # Default
        assert config.batch_size == 4  # Default

    def test_builder_overwrite_same_field(self):
        """
        Test: Later with_* calls override earlier values for same field
        Why: Support incremental refinement
        Contract: Last value wins
        """
        config = (TrainingConfigBuilder()
            .with_training(learning_rate=1e-4)
            .with_training(learning_rate=5e-5)  # Override
            .build()
        )

        assert config.learning_rate == 5e-5

    def test_builder_none_values_ignored(self):
        """
        Test: Passing None to with_* methods does not update config
        Why: Support conditional parameter setting
        Contract: None values skipped
        """
        config = (TrainingConfigBuilder()
            .with_training(learning_rate=1e-4, batch_size=None)  # type: ignore
            .build()
        )

        assert config.learning_rate == 1e-4
        # batch_size should be default (None was ignored)
        assert config.batch_size == 4
