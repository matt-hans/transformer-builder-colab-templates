"""
Training Configuration Management for Reproducibility.

This module provides a versioned configuration system for transformer training,
enabling exact reproduction of experiments. Configurations include all
hyperparameters, model architecture, dataset settings, and experiment tracking
metadata.

Key features:
- TrainingConfig dataclass with comprehensive validation
- JSON serialization with timestamped versioning
- Configuration comparison and diff utilities
- W&B integration for experiment tracking
- Type-safe configuration with sensible defaults

Example usage:
    >>> # Create configuration
    >>> config = TrainingConfig(
    ...     learning_rate=5e-5,
    ...     batch_size=4,
    ...     epochs=10,
    ...     notes="Baseline experiment"
    ... )
    >>>
    >>> # Validate before training
    >>> config.validate()
    >>>
    >>> # Save for later reproduction
    >>> config.save()  # Auto-generates timestamped filename
    >>>
    >>> # Load and reproduce
    >>> loaded = TrainingConfig.load("config_20250115_143022.json")
    >>>
    >>> # Compare configurations
    >>> diff = compare_configs(config_v1, config_v2)
    >>> print(diff['changed'])  # See what changed

Architecture:
    TrainingConfig uses Python's dataclasses for type safety and automatic
    serialization. All fields have defaults to enable incremental configuration.
    Validation uses explicit checks with accumulated error reporting.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Dict, Tuple, Any, Union, List
import json
import logging
from datetime import datetime
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Complete training configuration for reproducibility.

    This dataclass captures all settings needed to reproduce a training run:
    hyperparameters, model architecture, dataset configuration, reproducibility
    settings (seed), experiment tracking metadata, and checkpointing options.

    All fields have sensible defaults based on common transformer training
    practices. Validation ensures configurations are internally consistent and
    within valid ranges before training begins.

    Attributes:
        # Reproducibility
        random_seed: Random seed for reproducibility (default: 42)
        deterministic: Enable fully deterministic mode, slower but bit-exact (default: False)

        # Hyperparameters
        learning_rate: Learning rate for optimizer (default: 5e-5)
        batch_size: Training batch size (default: 4)
        epochs: Number of training epochs (default: 10)
        warmup_ratio: Fraction of steps for learning rate warmup (default: 0.1)
        weight_decay: L2 regularization coefficient (default: 0.01)
        max_grad_norm: Gradient clipping threshold (default: 1.0)

        # Training Features
        use_amp: Enable automatic mixed precision training (default: True)
        gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)
        early_stopping_patience: Epochs to wait before early stopping (default: 5)
        validation_split: Fraction of data for validation (default: 0.1)

        # Model Architecture
        model_name: Human-readable model identifier (default: "custom-transformer")
        model_type: Architecture family (default: "gpt")
        vocab_size: Vocabulary size (default: 50257, GPT-2 tokenizer)
        max_seq_len: Maximum sequence length (default: 128)
        d_model: Model dimension (default: 768)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        d_ff: Feed-forward dimension (default: 3072)
        dropout: Dropout probability (default: 0.1)

        # Dataset
        dataset_name: Dataset identifier (default: "wikitext-103-v1")
        dataset_split: Split to use for training (default: "train")
        dataset_subset: Optional subset name (default: None)
        max_train_samples: Limit training samples for quick experiments (default: None)
        max_val_samples: Limit validation samples (default: None)

        # Checkpointing
        checkpoint_dir: Directory for saving checkpoints (default: Colab Drive path)
        save_every_n_epochs: Checkpoint frequency (default: 5)
        keep_best_only: Only keep best checkpoint, delete others (default: False)

        # Experiment Tracking
        wandb_project: W&B project name (default: "transformer-builder-training")
        wandb_entity: W&B entity/team (default: None)
        run_name: Experiment run name (default: None, auto-generated)

        # Metadata
        created_at: ISO timestamp when config was created (auto-generated)
        config_version: Config schema version (default: "1.0")
        notes: Freeform notes about this experiment (default: "")

    Example:
        >>> config = TrainingConfig(
        ...     learning_rate=1e-4,
        ...     batch_size=8,
        ...     epochs=20,
        ...     vocab_size=32000,
        ...     notes="Testing increased batch size"
        ... )
        >>> config.validate()
        >>> config.save("experiments/config_baseline.json")
    """

    # === Reproducibility ===
    random_seed: int = 42
    deterministic: bool = False

    # === Hyperparameters ===
    learning_rate: float = 5e-5
    batch_size: int = 4
    epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # === Training Features ===
    use_amp: bool = True  # Mixed precision
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    validation_split: float = 0.1

    # === Compilation Settings (v3.5.0) ===
    compile_mode: Optional[str] = None  # None=disabled, "default"|"reduce-overhead"|"max-autotune"
    compile_fullgraph: bool = False     # Require single graph (strict, may fail)
    compile_dynamic: bool = True        # Support dynamic shapes (safer for variable seq lengths)

    # === Distributed / Precision Settings ===
    # Lightning strategy: "auto", "ddp", "fsdp_native", or None for vanilla
    strategy: Optional[str] = "auto"
    # Devices can be an int, "auto", a list of ints, or None
    devices: Optional[Union[int, str, List[int]]] = "auto"
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    # Precision string passed to Lightning; mapped downstream to AMP utilities
    precision: str = "bf16-mixed"

    # === Model Architecture ===
    model_name: str = "custom-transformer"
    model_type: Literal["gpt", "bert", "t5", "custom"] = "gpt"
    vocab_size: int = 50257
    max_seq_len: int = 128
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1

    # === Dataset / Task Selection ===
    # Optional high-level task selector used by TaskSpec/EvalConfig builders
    task_name: str = "lm_tiny"
    # Optional dataset preset identifier for evaluation; if None, derived from task_name
    eval_dataset_id: Optional[str] = None

    # Optional checkpoint to resume from (Lightning ckpt path)
    resume_from_checkpoint: Optional[str] = None

    # Legacy dataset fields (kept for backwards compatibility and power users)
    dataset_name: str = "wikitext-103-v1"
    dataset_split: str = "train"
    dataset_subset: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    # === Checkpointing ===
    checkpoint_dir: str = "/content/drive/MyDrive/transformer-checkpoints"
    save_every_n_epochs: int = 5
    keep_best_only: bool = False

    # === Experiment Tracking ===
    wandb_project: str = "transformer-builder-training"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None

    # === Production Inference Artifacts (v3.5) ===
    export_bundle: bool = False  # Generate full deployment bundle
    export_formats: List[str] = field(default_factory=lambda: ["onnx", "torchscript"])
    export_dir: str = "exports"

    # === Metadata ===
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_version: str = "1.0"
    notes: str = ""

    def validate(self) -> bool:
        """
        Validate configuration values for correctness and consistency.

        This method performs comprehensive validation of all configuration
        parameters, checking:
        - Numeric values are positive where required
        - Ratios/percentages are in valid ranges [0, 1]
        - Architectural constraints (e.g., d_model divisible by num_heads)
        - Required minimums (vocab_size >= 1, etc.)

        All validation errors are accumulated and reported together, so users
        can fix multiple issues at once rather than encountering them one by one.

        Returns:
            bool: True if validation passes

        Raises:
            ValueError: If any validation checks fail. The error message contains
                all validation failures formatted as a bulleted list.

        Example:
            >>> config = TrainingConfig(learning_rate=-0.001, batch_size=0)
            >>> config.validate()
            ValueError: Configuration validation failed:
              - learning_rate must be positive
              - batch_size must be >= 1
        """
        errors = []

        # Validate hyperparameters - numeric ranges
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")

        if self.epochs < 1:
            errors.append("epochs must be >= 1")

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append("warmup_ratio must be in [0, 1]")

        if self.validation_split < 0 or self.validation_split > 0.5:
            errors.append("validation_split must be in [0, 0.5]")

        # Validate model architecture constraints
        if self.vocab_size < 1:
            errors.append("vocab_size must be >= 1")

        if self.max_seq_len < 1:
            errors.append("max_seq_len must be >= 1")

        # Critical transformer constraint: d_model must be divisible by num_heads
        # This ensures each head gets an integer dimension (d_model // num_heads)
        if self.d_model < 1 or self.d_model % self.num_heads != 0:
            errors.append(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

        # Report all errors together
        if errors:
            # Log config values for debugging production issues
            logger = logging.getLogger(__name__)
            logger.error(
                f"Configuration validation failed for config with:\n"
                f"  learning_rate={self.learning_rate}, batch_size={self.batch_size}, "
                f"epochs={self.epochs}\n"
                f"  warmup_ratio={self.warmup_ratio}, validation_split={self.validation_split}\n"
                f"  d_model={self.d_model}, num_heads={self.num_heads}\n"
                f"  vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len}\n"
                f"Errors:\n" + "\n".join(f"  - {e}" for e in errors)
            )

            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_message)

        return True

    def save(self, path: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.

        Serializes the complete configuration to a JSON file for later loading.
        If no path is specified, auto-generates a timestamped filename in the
        current directory to prevent accidental overwrites.

        Args:
            path: Optional custom path. If None, auto-generates filename as
                config_YYYYMMDD_HHMMSS.json in current directory.

        Returns:
            str: Absolute path where config was saved

        Example:
            >>> config = TrainingConfig()
            >>> # Auto-generate timestamped filename
            >>> path = config.save()
            >>> print(path)  # config_20250115_143022.json
            >>>
            >>> # Or specify custom path
            >>> config.save("experiments/baseline_config.json")

        Note:
            The saved JSON is human-readable and can be manually edited, but
            be careful to maintain valid JSON syntax and pass validation when
            loading.
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"config_{timestamp}.json"

        # Convert to dictionary and serialize with error handling
        try:
            with open(path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except PermissionError as e:
            raise IOError(
                f"Permission denied writing configuration to {path}. "
                f"Check file/directory permissions. Original error: {e}"
            )
        except OSError as e:
            raise IOError(
                f"Failed to write configuration to {path}. "
                f"Possible causes: disk full, invalid path, I/O error. "
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error saving configuration to {path}: {e}"
            )

        print(f"✅ Configuration saved to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """
        Load configuration from JSON file.

        Deserializes a previously saved configuration from JSON. The loaded
        configuration can be used to reproduce a previous training run exactly.

        Args:
            path: Path to JSON config file

        Returns:
            TrainingConfig: Loaded configuration instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
            TypeError: If JSON contains fields not in TrainingConfig schema

        Example:
            >>> # Load previous experiment config
            >>> config = TrainingConfig.load("config_20250115_143022.json")
            >>> config.validate()
            >>> # Now use config for training...

        Note:
            After loading, it's recommended to call validate() to ensure the
            loaded config is still valid (in case of manual edits or schema
            version changes).
        """
        # Load and parse JSON with comprehensive error handling
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Expected a JSON file created by TrainingConfig.save(). "
                f"Check that the file exists and the path is correct."
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in configuration file {path}.\n"
                f"The file may be corrupted or not valid JSON syntax.\n"
                f"JSON error: {e}"
            )
        except PermissionError as e:
            raise IOError(
                f"Permission denied reading configuration from {path}. "
                f"Check file permissions. Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error reading configuration from {path}: {e}"
            )

        # Instantiate from dict with type checking
        try:
            config = cls(**config_dict)
        except TypeError as e:
            raise ValueError(
                f"Invalid configuration structure in {path}.\n"
                f"The JSON may be from an incompatible version or contain "
                f"unexpected fields.\n"
                f"Type error: {e}"
            )

        print(f"✅ Configuration loaded from {path}")
        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Useful for W&B integration, logging, and programmatic access to config
        values. The returned dict contains all fields, including metadata.

        Returns:
            dict: Configuration as dictionary with all fields

        Example:
            >>> config = TrainingConfig()
            >>> config_dict = config.to_dict()
            >>> wandb.config.update(config_dict)
        """
        return asdict(self)


def compare_configs(
    config1: TrainingConfig,
    config2: TrainingConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Compare two configurations and return differences.

    This utility identifies what changed between two experiment configurations,
    making it easy to track experiment variations. Comparison skips metadata
    fields (created_at, run_name) that are expected to differ between runs.

    Args:
        config1: First configuration (baseline)
        config2: Second configuration (comparison)

    Returns:
        dict: Dictionary with three keys:
            - 'changed': Fields that differ, maps field -> (old_value, new_value)
            - 'added': Fields present in config2 but not config1
            - 'removed': Fields present in config1 but not config2

    Example:
        >>> baseline = TrainingConfig(learning_rate=5e-5, batch_size=4)
        >>> experiment = TrainingConfig(learning_rate=1e-4, batch_size=8)
        >>> diff = compare_configs(baseline, experiment)
        >>> print(diff['changed'])
        {
            'learning_rate': (5e-5, 1e-4),
            'batch_size': (4, 8)
        }

    Note:
        Metadata fields (created_at, run_name) are automatically excluded
        from comparison as they're expected to differ between runs.
        Use print_config_diff() to display differences in human-readable format.
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()

    all_keys = set(dict1.keys()) | set(dict2.keys())

    differences: Dict[str, Dict[str, Any]] = {
        'changed': {},
        'added': {},
        'removed': {},
    }

    # Fields to skip in comparison (expected to differ between runs)
    skip_fields = {'created_at', 'run_name'}

    for key in all_keys:
        # Skip metadata fields
        if key in skip_fields:
            continue

        # Check if key exists in both dicts (not just if value is None)
        key_in_1 = key in dict1
        key_in_2 = key in dict2

        if not key_in_1:
            # Key only exists in config2
            differences['added'][key] = dict2[key]
        elif not key_in_2:
            # Key only exists in config1
            differences['removed'][key] = dict1[key]
        else:
            # Key exists in both, check if values differ
            v1 = dict1[key]
            v2 = dict2[key]
            if v1 != v2:
                differences['changed'][key] = (v1, v2)

    return differences


def print_config_diff(differences: Dict[str, Dict[str, Any]]) -> None:
    """
    Pretty-print configuration differences to stdout.

    Displays changes in a human-readable format with unicode symbols
    for quick visual inspection of what changed between configurations.

    Args:
        differences: Dict returned by compare_configs() with keys:
                     'changed', 'added', 'removed'

    Example:
        >>> diff = compare_configs(config1, config2)
        >>> print_config_diff(diff)
        🔍 Configuration Differences:
          learning_rate: 5e-5 → 1e-4
          batch_size: 4 → 8
    """
    if differences['changed']:
        print("🔍 Configuration Differences:")
        for key, (old, new) in differences['changed'].items():
            print(f"  {key}: {old} → {new}")
    elif not differences['added'] and not differences['removed']:
        print("✅ Configurations are identical (excluding metadata)")

    if differences['added']:
        print("➕ Added fields:")
        for key, value in differences['added'].items():
            print(f"  {key}: {value}")

    if differences['removed']:
        print("➖ Removed fields:")
        for key, value in differences['removed'].items():
            print(f"  {key}: {value}")


# =============================================================================
# Builder Pattern for Fluent Configuration Construction
# =============================================================================


class TrainingConfigBuilder:
    """
    Fluent builder for constructing TrainingConfig with progressive validation.

    The builder pattern provides an ergonomic API for creating complex training
    configurations through method chaining. Each method validates inputs early,
    catching errors before training begins. The builder is immutable - each
    method returns a new builder instance.

    Key features:
    - Fluent API with method chaining
    - Progressive validation (catch errors as you build)
    - Organized methods by concern (model, training, hardware, etc.)
    - Preset factories for common configurations
    - Backward compatible with direct TrainingConfig construction

    Example:
        >>> # Build custom configuration with fluent API
        >>> config = (TrainingConfigBuilder()
        ...     .with_model(d_model=512, num_layers=6, num_heads=8, vocab_size=32000)
        ...     .with_training(learning_rate=1e-4, batch_size=16, epochs=20)
        ...     .with_optimizer(weight_decay=0.1, warmup_ratio=0.05)
        ...     .with_hardware(use_amp=True, compile_mode="default")
        ...     .with_logging(wandb_project="my-project", run_name="experiment-1")
        ...     .build()
        ... )
        >>>
        >>> # Or use a preset and customize
        >>> config = (TrainingConfigBuilder.baseline()
        ...     .with_training(epochs=30)  # Override just epochs
        ...     .with_logging(run_name="custom-run")
        ...     .build()
        ... )

    Presets:
        - quick_prototype(): Fast iteration, small model, 3 epochs
        - baseline(): Standard config, balanced settings
        - production(): High quality, full training
        - distributed(): Multi-GPU with DDP/FSDP
        - low_memory(): Resource-constrained environments

    Architecture:
        The builder stores partial config as a dict, validates incrementally,
        then constructs the final TrainingConfig in build(). Immutability
        ensures thread safety and prevents accidental state mutation.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize builder with optional starting values.

        Args:
            **kwargs: Initial configuration values (optional)

        Example:
            >>> # Start fresh
            >>> builder = TrainingConfigBuilder()
            >>>
            >>> # Start with some values
            >>> builder = TrainingConfigBuilder(learning_rate=1e-4, batch_size=8)
        """
        self._config: Dict[str, Any] = kwargs.copy()

    def _copy_with(self, **updates: Any) -> 'TrainingConfigBuilder':
        """
        Create new builder with updated values (immutable pattern).

        Args:
            **updates: Fields to update in the new builder

        Returns:
            New builder instance with merged values
        """
        new_config = self._config.copy()
        new_config.update(updates)
        return TrainingConfigBuilder(**new_config)

    def with_model(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[Literal["gpt", "bert", "t5", "custom"]] = None,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        d_model: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure model architecture parameters.

        Args:
            model_name: Human-readable model identifier
            model_type: Architecture family (gpt, bert, t5, custom)
            vocab_size: Vocabulary size (must be >= 1)
            max_seq_len: Maximum sequence length (must be >= 1)
            d_model: Model dimension (must be divisible by num_heads)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability (must be in [0, 1])

        Returns:
            New builder with model parameters set

        Raises:
            ValueError: If validation fails for any parameter

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_model(
            ...         model_name="gpt-small",
            ...         model_type="gpt",
            ...         vocab_size=50257,
            ...         d_model=512,
            ...         num_layers=6,
            ...         num_heads=8
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if model_name is not None:
            updates['model_name'] = model_name

        if model_type is not None:
            if model_type not in ("gpt", "bert", "t5", "custom"):
                raise ValueError(
                    f"Invalid model_type '{model_type}'. "
                    "Must be one of: gpt, bert, t5, custom"
                )
            updates['model_type'] = model_type

        if vocab_size is not None:
            if vocab_size < 1:
                raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
            updates['vocab_size'] = vocab_size

        if max_seq_len is not None:
            if max_seq_len < 1:
                raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
            updates['max_seq_len'] = max_seq_len

        if d_model is not None:
            if d_model < 1:
                raise ValueError(f"d_model must be >= 1, got {d_model}")
            updates['d_model'] = d_model

        if num_layers is not None:
            if num_layers < 1:
                raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            updates['num_layers'] = num_layers

        if num_heads is not None:
            if num_heads < 1:
                raise ValueError(f"num_heads must be >= 1, got {num_heads}")
            updates['num_heads'] = num_heads

        if d_ff is not None:
            if d_ff < 1:
                raise ValueError(f"d_ff must be >= 1, got {d_ff}")
            updates['d_ff'] = d_ff

        if dropout is not None:
            if not (0.0 <= dropout <= 1.0):
                raise ValueError(f"dropout must be in [0, 1], got {dropout}")
            updates['dropout'] = dropout

        # Validate d_model/num_heads divisibility if both present
        new_config = self._config.copy()
        new_config.update(updates)
        if 'd_model' in new_config and 'num_heads' in new_config:
            if new_config['d_model'] % new_config['num_heads'] != 0:
                raise ValueError(
                    f"d_model ({new_config['d_model']}) must be divisible by "
                    f"num_heads ({new_config['num_heads']})"
                )

        return self._copy_with(**updates)

    def with_training(
        self,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        validation_split: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure core training parameters.

        Args:
            learning_rate: Learning rate (must be > 0)
            batch_size: Training batch size (must be >= 1)
            epochs: Number of training epochs (must be >= 1)
            validation_split: Fraction of data for validation (must be in [0, 0.5])
            early_stopping_patience: Epochs to wait before early stopping
            max_train_samples: Limit training samples for quick experiments
            max_val_samples: Limit validation samples

        Returns:
            New builder with training parameters set

        Raises:
            ValueError: If validation fails for any parameter

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_training(
            ...         learning_rate=5e-5,
            ...         batch_size=8,
            ...         epochs=10,
            ...         validation_split=0.1
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if learning_rate is not None:
            if learning_rate <= 0:
                raise ValueError(f"learning_rate must be positive, got {learning_rate}")
            updates['learning_rate'] = learning_rate

        if batch_size is not None:
            if batch_size < 1:
                raise ValueError(f"batch_size must be >= 1, got {batch_size}")
            updates['batch_size'] = batch_size

        if epochs is not None:
            if epochs < 1:
                raise ValueError(f"epochs must be >= 1, got {epochs}")
            updates['epochs'] = epochs

        if validation_split is not None:
            if not (0.0 <= validation_split <= 0.5):
                raise ValueError(
                    f"validation_split must be in [0, 0.5], got {validation_split}"
                )
            updates['validation_split'] = validation_split

        if early_stopping_patience is not None:
            if early_stopping_patience < 0:
                raise ValueError(
                    f"early_stopping_patience must be >= 0, got {early_stopping_patience}"
                )
            updates['early_stopping_patience'] = early_stopping_patience

        if max_train_samples is not None:
            if max_train_samples < 1:
                raise ValueError(f"max_train_samples must be >= 1, got {max_train_samples}")
            updates['max_train_samples'] = max_train_samples

        if max_val_samples is not None:
            if max_val_samples < 1:
                raise ValueError(f"max_val_samples must be >= 1, got {max_val_samples}")
            updates['max_val_samples'] = max_val_samples

        return self._copy_with(**updates)

    def with_optimizer(
        self,
        weight_decay: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        warmup_ratio: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure optimizer and gradient settings.

        Args:
            weight_decay: L2 regularization coefficient (must be >= 0)
            max_grad_norm: Gradient clipping threshold (must be > 0)
            warmup_ratio: Fraction of steps for LR warmup (must be in [0, 1])
            gradient_accumulation_steps: Steps to accumulate gradients (must be >= 1)

        Returns:
            New builder with optimizer parameters set

        Raises:
            ValueError: If validation fails for any parameter

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_optimizer(
            ...         weight_decay=0.01,
            ...         max_grad_norm=1.0,
            ...         warmup_ratio=0.1,
            ...         gradient_accumulation_steps=4
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if weight_decay is not None:
            if weight_decay < 0:
                raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
            updates['weight_decay'] = weight_decay

        if max_grad_norm is not None:
            if max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
            updates['max_grad_norm'] = max_grad_norm

        if warmup_ratio is not None:
            if not (0.0 <= warmup_ratio <= 1.0):
                raise ValueError(f"warmup_ratio must be in [0, 1], got {warmup_ratio}")
            updates['warmup_ratio'] = warmup_ratio

        if gradient_accumulation_steps is not None:
            if gradient_accumulation_steps < 1:
                raise ValueError(
                    f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}"
                )
            updates['gradient_accumulation_steps'] = gradient_accumulation_steps
            # Also update accumulate_grad_batches for Lightning compatibility
            updates['accumulate_grad_batches'] = gradient_accumulation_steps

        return self._copy_with(**updates)

    def with_scheduler(
        self,
        use_lr_schedule: Optional[bool] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure learning rate scheduler.

        Note: Currently the training pipeline uses a fixed warmup + cosine decay
        schedule. This method is provided for future extensibility.

        Args:
            use_lr_schedule: Enable LR schedule (warmup + cosine decay)

        Returns:
            New builder with scheduler parameters set

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_scheduler(use_lr_schedule=True)
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if use_lr_schedule is not None:
            updates['use_lr_schedule'] = use_lr_schedule

        return self._copy_with(**updates)

    def with_hardware(
        self,
        use_amp: Optional[bool] = None,
        compile_mode: Optional[str] = None,
        compile_fullgraph: Optional[bool] = None,
        compile_dynamic: Optional[bool] = None,
        strategy: Optional[str] = None,
        devices: Optional[Union[int, str, List[int]]] = None,
        num_nodes: Optional[int] = None,
        precision: Optional[str] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure hardware and performance settings.

        Args:
            use_amp: Enable automatic mixed precision training
            compile_mode: PyTorch 2.0 compilation mode
                ("default", "reduce-overhead", "max-autotune", or None to disable)
            compile_fullgraph: Require single graph (strict mode)
            compile_dynamic: Support dynamic shapes (safer for variable seq lengths)
            strategy: Lightning strategy ("auto", "ddp", "fsdp_native", or None)
            devices: GPU devices (int, "auto", or list of ints)
            num_nodes: Number of nodes for distributed training
            precision: Precision mode ("bf16-mixed", "16-mixed", "32", etc.)

        Returns:
            New builder with hardware parameters set

        Raises:
            ValueError: If validation fails for any parameter

        Example:
            >>> # Single GPU with compilation
            >>> config = (TrainingConfigBuilder()
            ...     .with_hardware(
            ...         use_amp=True,
            ...         compile_mode="default",
            ...         devices=1
            ...     )
            ...     .build()
            ... )
            >>>
            >>> # Multi-GPU with DDP
            >>> config = (TrainingConfigBuilder.distributed()
            ...     .with_hardware(devices=4, strategy="ddp")
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if use_amp is not None:
            updates['use_amp'] = use_amp

        if compile_mode is not None:
            valid_modes = {None, "default", "reduce-overhead", "max-autotune"}
            if compile_mode not in valid_modes:
                raise ValueError(
                    f"Invalid compile_mode '{compile_mode}'. "
                    f"Must be one of: {valid_modes}"
                )
            updates['compile_mode'] = compile_mode

        if compile_fullgraph is not None:
            updates['compile_fullgraph'] = compile_fullgraph

        if compile_dynamic is not None:
            updates['compile_dynamic'] = compile_dynamic

        if strategy is not None:
            valid_strategies = {None, "auto", "ddp", "fsdp_native", "dp"}
            if strategy not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{strategy}'. "
                    f"Must be one of: {valid_strategies}"
                )
            updates['strategy'] = strategy

        if devices is not None:
            # Validate devices is int, str, or list of ints
            if not (isinstance(devices, (int, str)) or
                    (isinstance(devices, list) and all(isinstance(d, int) for d in devices))):
                raise ValueError(
                    f"devices must be int, 'auto', or list of ints, got {type(devices)}"
                )
            if isinstance(devices, int) and devices < 1:
                raise ValueError(f"devices must be >= 1, got {devices}")
            updates['devices'] = devices

        if num_nodes is not None:
            if num_nodes < 1:
                raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
            updates['num_nodes'] = num_nodes

        if precision is not None:
            # Validate precision string format
            valid_precisions = {
                "32", "16", "bf16",
                "16-mixed", "bf16-mixed",
                "32-true", "16-true", "bf16-true"
            }
            if precision not in valid_precisions:
                raise ValueError(
                    f"Invalid precision '{precision}'. "
                    f"Must be one of: {valid_precisions}"
                )
            updates['precision'] = precision

        # Warn about conflicting settings
        new_config = self._config.copy()
        new_config.update(updates)
        if new_config.get('use_amp') and new_config.get('precision') == '32':
            logger = logging.getLogger(__name__)
            logger.warning(
                "Conflicting settings: use_amp=True but precision='32'. "
                "Consider setting precision='16-mixed' or 'bf16-mixed' for AMP."
            )

        return self._copy_with(**updates)

    def with_logging(
        self,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        run_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure experiment tracking and logging.

        Args:
            wandb_project: W&B project name
            wandb_entity: W&B entity/team
            run_name: Experiment run name
            notes: Freeform notes about this experiment

        Returns:
            New builder with logging parameters set

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_logging(
            ...         wandb_project="my-transformers",
            ...         run_name="experiment-v1",
            ...         notes="Testing higher learning rate"
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if wandb_project is not None:
            updates['wandb_project'] = wandb_project

        if wandb_entity is not None:
            updates['wandb_entity'] = wandb_entity

        if run_name is not None:
            updates['run_name'] = run_name

        if notes is not None:
            updates['notes'] = notes

        return self._copy_with(**updates)

    def with_checkpointing(
        self,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: Optional[int] = None,
        keep_best_only: Optional[bool] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure checkpointing behavior.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            save_every_n_epochs: Checkpoint frequency (epochs)
            keep_best_only: Only keep best checkpoint, delete others
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            New builder with checkpointing parameters set

        Raises:
            ValueError: If validation fails for any parameter

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_checkpointing(
            ...         checkpoint_dir="./checkpoints",
            ...         save_every_n_epochs=2,
            ...         keep_best_only=True
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if checkpoint_dir is not None:
            updates['checkpoint_dir'] = checkpoint_dir

        if save_every_n_epochs is not None:
            if save_every_n_epochs < 1:
                raise ValueError(
                    f"save_every_n_epochs must be >= 1, got {save_every_n_epochs}"
                )
            updates['save_every_n_epochs'] = save_every_n_epochs

        if keep_best_only is not None:
            updates['keep_best_only'] = keep_best_only

        if resume_from_checkpoint is not None:
            # Validate path exists if provided
            checkpoint_path = Path(resume_from_checkpoint)
            if not checkpoint_path.exists():
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Checkpoint path does not exist: {resume_from_checkpoint}. "
                    "Training will fail if this path is not created before training starts."
                )
            updates['resume_from_checkpoint'] = resume_from_checkpoint

        return self._copy_with(**updates)

    def with_export(
        self,
        export_bundle: Optional[bool] = None,
        export_formats: Optional[List[str]] = None,
        export_dir: Optional[str] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure production inference artifact export.

        Args:
            export_bundle: Generate full deployment bundle after training
            export_formats: Export formats (e.g., ["onnx", "torchscript"])
            export_dir: Directory for export artifacts

        Returns:
            New builder with export parameters set

        Raises:
            ValueError: If validation fails for any parameter

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_export(
            ...         export_bundle=True,
            ...         export_formats=["onnx", "torchscript"],
            ...         export_dir="./exports"
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if export_bundle is not None:
            updates['export_bundle'] = export_bundle

        if export_formats is not None:
            valid_formats = {"onnx", "torchscript", "pytorch"}
            for fmt in export_formats:
                if fmt not in valid_formats:
                    raise ValueError(
                        f"Invalid export format '{fmt}'. "
                        f"Must be one of: {valid_formats}"
                    )
            updates['export_formats'] = export_formats

        if export_dir is not None:
            updates['export_dir'] = export_dir

        return self._copy_with(**updates)

    def with_reproducibility(
        self,
        random_seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure reproducibility settings.

        Args:
            random_seed: Random seed for reproducibility
            deterministic: Enable fully deterministic mode (slower, bit-exact)

        Returns:
            New builder with reproducibility parameters set

        Example:
            >>> # Fast mode (default)
            >>> config = (TrainingConfigBuilder()
            ...     .with_reproducibility(random_seed=42, deterministic=False)
            ...     .build()
            ... )
            >>>
            >>> # Deterministic mode for publication results
            >>> config = (TrainingConfigBuilder()
            ...     .with_reproducibility(random_seed=42, deterministic=True)
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if random_seed is not None:
            updates['random_seed'] = random_seed

        if deterministic is not None:
            updates['deterministic'] = deterministic

        return self._copy_with(**updates)

    def with_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        dataset_subset: Optional[str] = None,
        task_name: Optional[str] = None,
        eval_dataset_id: Optional[str] = None,
    ) -> 'TrainingConfigBuilder':
        """
        Configure dataset and task settings.

        Args:
            dataset_name: Dataset identifier (legacy field)
            dataset_split: Split to use for training (legacy field)
            dataset_subset: Optional subset name (legacy field)
            task_name: High-level task selector (used by TaskSpec)
            eval_dataset_id: Dataset preset for evaluation

        Returns:
            New builder with dataset parameters set

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_dataset(
            ...         task_name="lm_tiny",
            ...         eval_dataset_id="wikitext-v1"
            ...     )
            ...     .build()
            ... )
        """
        updates: Dict[str, Any] = {}

        if dataset_name is not None:
            updates['dataset_name'] = dataset_name

        if dataset_split is not None:
            updates['dataset_split'] = dataset_split

        if dataset_subset is not None:
            updates['dataset_subset'] = dataset_subset

        if task_name is not None:
            updates['task_name'] = task_name

        if eval_dataset_id is not None:
            updates['eval_dataset_id'] = eval_dataset_id

        return self._copy_with(**updates)

    def build(self) -> TrainingConfig:
        """
        Construct final TrainingConfig with full validation.

        Returns:
            Validated TrainingConfig instance

        Raises:
            ValueError: If final configuration is invalid

        Example:
            >>> config = (TrainingConfigBuilder()
            ...     .with_model(d_model=512, num_heads=8)
            ...     .with_training(learning_rate=1e-4, epochs=10)
            ...     .build()
            ... )
        """
        # Create config with merged defaults
        config = TrainingConfig(**self._config)

        # Run full validation
        config.validate()

        return config

    # =========================================================================
    # Preset Factory Methods
    # =========================================================================

    @classmethod
    def quick_prototype(cls) -> 'TrainingConfigBuilder':
        """
        Quick prototype preset: Fast iteration, small model, minimal epochs.

        Use case:
        - Rapid prototyping and debugging
        - Testing new ideas quickly
        - Verifying training pipeline works
        - CI/CD smoke tests

        Configuration:
        - Model: 6 layers, 512 dims, 8 heads (12M params)
        - Training: 3 epochs, batch size 8
        - Hardware: AMP enabled, no compilation
        - Checkpointing: Every epoch, keep best only

        Expected runtime: ~5-10 minutes on single GPU

        Returns:
            Builder configured for quick prototyping

        Example:
            >>> # Use preset as-is
            >>> config = TrainingConfigBuilder.quick_prototype().build()
            >>>
            >>> # Customize specific parameters
            >>> config = (TrainingConfigBuilder.quick_prototype()
            ...     .with_training(epochs=5)
            ...     .with_logging(run_name="quick-test-v2")
            ...     .build()
            ... )
        """
        return cls(
            # Model: Small but functional
            model_name="quick-prototype",
            d_model=512,
            num_layers=6,
            num_heads=8,
            d_ff=2048,
            max_seq_len=128,
            vocab_size=50257,

            # Training: Minimal epochs for speed
            learning_rate=5e-4,  # Higher LR for faster convergence
            batch_size=8,
            epochs=3,
            warmup_ratio=0.05,  # Short warmup
            validation_split=0.1,
            early_stopping_patience=2,

            # Optimizer: Standard settings
            weight_decay=0.01,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,

            # Hardware: AMP only, no compilation (faster startup)
            use_amp=True,
            compile_mode=None,
            strategy="auto",
            devices="auto",

            # Checkpointing: Keep best only to save space
            save_every_n_epochs=1,
            keep_best_only=True,

            # Logging
            wandb_project="transformer-builder-training",
            notes="Quick prototype configuration for fast iteration",

            # Reproducibility: Fast mode
            random_seed=42,
            deterministic=False,
        )

    @classmethod
    def baseline(cls) -> 'TrainingConfigBuilder':
        """
        Baseline preset: Standard config, balanced settings.

        Use case:
        - Default starting point for most experiments
        - Balanced speed/quality tradeoff
        - Comparing against standard settings
        - Production training on single GPU

        Configuration:
        - Model: 12 layers, 768 dims, 12 heads (125M params, GPT-2 scale)
        - Training: 10 epochs, batch size 4
        - Hardware: AMP + torch.compile "default" mode
        - Checkpointing: Every 5 epochs

        Expected runtime: ~2-4 hours on V100/A100

        Returns:
            Builder configured for baseline training

        Example:
            >>> # Standard baseline
            >>> config = TrainingConfigBuilder.baseline().build()
            >>>
            >>> # Baseline with higher batch size
            >>> config = (TrainingConfigBuilder.baseline()
            ...     .with_training(batch_size=16)
            ...     .with_optimizer(gradient_accumulation_steps=4)
            ...     .build()
            ... )
        """
        return cls(
            # Model: GPT-2 small scale
            model_name="baseline-transformer",
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
            max_seq_len=128,
            vocab_size=50257,

            # Training: Standard hyperparameters
            learning_rate=5e-5,
            batch_size=4,
            epochs=10,
            warmup_ratio=0.1,
            validation_split=0.1,
            early_stopping_patience=5,

            # Optimizer: Conservative settings
            weight_decay=0.01,
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,

            # Hardware: AMP + compilation for performance
            use_amp=True,
            compile_mode="default",  # 10-15% speedup
            compile_dynamic=True,
            strategy="auto",
            devices="auto",
            precision="bf16-mixed",

            # Checkpointing: Standard frequency
            save_every_n_epochs=5,
            keep_best_only=False,

            # Logging
            wandb_project="transformer-builder-training",
            notes="Baseline configuration with balanced settings",

            # Reproducibility: Fast mode
            random_seed=42,
            deterministic=False,
        )

    @classmethod
    def production(cls) -> 'TrainingConfigBuilder':
        """
        Production preset: High quality, full training, maximum performance.

        Use case:
        - Final training runs for publication/deployment
        - Maximum model quality
        - Full dataset training
        - Reproducible results

        Configuration:
        - Model: 12 layers, 768 dims, 12 heads (GPT-2 scale)
        - Training: 20 epochs, batch size 8, larger effective batch via accumulation
        - Hardware: AMP + torch.compile "reduce-overhead", deterministic mode
        - Export: Full deployment bundle generated
        - Checkpointing: Every 2 epochs, all checkpoints saved

        Expected runtime: ~8-12 hours on A100

        Returns:
            Builder configured for production training

        Example:
            >>> # Production training
            >>> config = TrainingConfigBuilder.production().build()
            >>>
            >>> # Production with custom export
            >>> config = (TrainingConfigBuilder.production()
            ...     .with_export(export_formats=["onnx", "torchscript"])
            ...     .with_logging(run_name="final-model-v1")
            ...     .build()
            ... )
        """
        return cls(
            # Model: Production scale
            model_name="production-transformer",
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
            max_seq_len=256,  # Longer sequences
            vocab_size=50257,

            # Training: Full epochs, larger effective batch
            learning_rate=3e-5,  # More conservative
            batch_size=8,
            epochs=20,
            warmup_ratio=0.1,
            validation_split=0.1,
            early_stopping_patience=10,

            # Optimizer: Production settings
            weight_decay=0.01,
            max_grad_norm=1.0,
            gradient_accumulation_steps=4,  # Effective batch = 32

            # Hardware: Maximum performance
            use_amp=True,
            compile_mode="reduce-overhead",  # 15-20% speedup
            compile_dynamic=True,
            strategy="auto",
            devices="auto",
            precision="bf16-mixed",

            # Checkpointing: Save all for analysis
            save_every_n_epochs=2,
            keep_best_only=False,

            # Export: Generate deployment bundle
            export_bundle=True,
            export_formats=["onnx", "torchscript"],
            export_dir="exports",

            # Logging
            wandb_project="transformer-builder-production",
            notes="Production configuration for final training runs",

            # Reproducibility: Deterministic for reproducibility
            random_seed=42,
            deterministic=True,  # Bit-exact reproducibility
        )

    @classmethod
    def distributed(cls) -> 'TrainingConfigBuilder':
        """
        Distributed preset: Multi-GPU training with DDP/FSDP.

        Use case:
        - Large models that don't fit on single GPU
        - Multi-GPU clusters
        - Faster training with data parallelism
        - Scaling to multiple nodes

        Configuration:
        - Model: Large scale (configurable)
        - Training: Optimized for distributed
        - Hardware: DDP strategy, 4 GPUs default
        - Gradient accumulation for large effective batch

        Expected runtime: Depends on cluster size

        Warning:
            DDP/FSDP may cause issues in Jupyter/Colab notebooks.
            Use CLI training (run_training.py) for distributed setups.

        Returns:
            Builder configured for distributed training

        Example:
            >>> # 4-GPU DDP training
            >>> config = (TrainingConfigBuilder.distributed()
            ...     .with_hardware(devices=4, strategy="ddp")
            ...     .build()
            ... )
            >>>
            >>> # FSDP for very large models
            >>> config = (TrainingConfigBuilder.distributed()
            ...     .with_hardware(devices=8, strategy="fsdp_native")
            ...     .with_model(d_model=1024, num_layers=24)
            ...     .build()
            ... )
        """
        return cls(
            # Model: Larger scale for distributed
            model_name="distributed-transformer",
            d_model=1024,
            num_layers=24,
            num_heads=16,
            d_ff=4096,
            max_seq_len=256,
            vocab_size=50257,

            # Training: Distributed-friendly batch sizes
            learning_rate=1e-4,
            batch_size=8,  # Per GPU
            epochs=15,
            warmup_ratio=0.1,
            validation_split=0.1,
            early_stopping_patience=8,

            # Optimizer: Larger effective batch
            weight_decay=0.01,
            max_grad_norm=1.0,
            gradient_accumulation_steps=2,  # Effective batch = 64 (8 * 4 GPUs * 2)

            # Hardware: DDP configuration
            use_amp=True,
            compile_mode="default",
            strategy="ddp",  # DistributedDataParallel
            devices=4,  # 4 GPUs
            num_nodes=1,
            precision="bf16-mixed",
            accumulate_grad_batches=2,

            # Checkpointing: Moderate frequency
            save_every_n_epochs=3,
            keep_best_only=False,

            # Logging
            wandb_project="transformer-builder-distributed",
            notes="Distributed training configuration for multi-GPU",

            # Reproducibility: Fast mode (deterministic mode can be slower on multi-GPU)
            random_seed=42,
            deterministic=False,
        )

    @classmethod
    def low_memory(cls) -> 'TrainingConfigBuilder':
        """
        Low memory preset: Resource-constrained environments (Colab free, small GPUs).

        Use case:
        - Google Colab free tier (12-16GB GPU)
        - Consumer GPUs (RTX 3060, etc.)
        - CPU-only training
        - Memory-limited environments

        Configuration:
        - Model: Small (6 layers, 384 dims)
        - Training: Small batches, gradient accumulation
        - Hardware: AMP, no compilation (save memory)
        - Checkpointing: Keep best only

        Expected runtime: ~1-2 hours on Colab T4

        Returns:
            Builder configured for low memory

        Example:
            >>> # Colab free tier
            >>> config = TrainingConfigBuilder.low_memory().build()
            >>>
            >>> # Even more aggressive memory savings
            >>> config = (TrainingConfigBuilder.low_memory()
            ...     .with_training(batch_size=2)
            ...     .with_optimizer(gradient_accumulation_steps=8)
            ...     .with_model(max_seq_len=64)
            ...     .build()
            ... )
        """
        return cls(
            # Model: Small footprint
            model_name="low-memory-transformer",
            d_model=384,
            num_layers=6,
            num_heads=6,
            d_ff=1536,
            max_seq_len=128,
            vocab_size=50257,

            # Training: Small batches
            learning_rate=5e-5,
            batch_size=2,  # Very small batch
            epochs=10,
            warmup_ratio=0.1,
            validation_split=0.1,
            early_stopping_patience=5,
            max_train_samples=10000,  # Limit dataset size
            max_val_samples=1000,

            # Optimizer: Gradient accumulation for larger effective batch
            weight_decay=0.01,
            max_grad_norm=1.0,
            gradient_accumulation_steps=8,  # Effective batch = 16

            # Hardware: AMP for memory savings, no compilation
            use_amp=True,
            compile_mode=None,  # Compilation uses extra memory
            strategy="auto",
            devices=1,
            precision="16-mixed",  # FP16 (more compatible than bf16 on old GPUs)

            # Checkpointing: Minimal storage
            save_every_n_epochs=5,
            keep_best_only=True,  # Save space

            # Logging
            wandb_project="transformer-builder-training",
            notes="Low memory configuration for resource-constrained environments",

            # Reproducibility: Fast mode
            random_seed=42,
            deterministic=False,
        )


# Public API
__all__ = [
    'TrainingConfig',
    'TrainingConfigBuilder',
    'compare_configs',
    'print_config_diff',
]

# -----------------------------------------------------------------------------
# Builders for new abstractions (TaskSpec / EvalConfig)
# -----------------------------------------------------------------------------

def build_task_spec(training_config: 'TrainingConfig'):
    """
    Build a TaskSpec from the provided TrainingConfig.

    Notes:
        - Imports are local to avoid import cycles during static analysis/tests.
        - Defaults resolve from `task_name` to built-in tiny presets.
    """
    from .task_spec import get_default_task_specs, TaskSpec

    presets = get_default_task_specs()
    name = training_config.task_name
    if name not in presets:
        raise ValueError(f"Unknown task_name '{name}'. Available: {list(presets.keys())}")
    return presets[name]


def build_eval_config(training_config: 'TrainingConfig'):
    """
    Build an EvalConfig using TrainingConfig defaults.

    Derives dataset_id from `eval_dataset_id` if set, otherwise from `task_name`.
    """
    from .eval_config import EvalConfig

    dataset_id = training_config.eval_dataset_id or f"{training_config.task_name}_v1"
    return EvalConfig(
        dataset_id=dataset_id,
        split="validation",
        max_eval_examples=training_config.max_val_samples or 512,
        batch_size=training_config.batch_size,
        num_workers=0,
        max_seq_length=training_config.max_seq_len,
        eval_interval_steps=100,
        eval_on_start=True,
    )
