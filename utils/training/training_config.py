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
from typing import Optional, Literal, Dict, Tuple, Any
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

    # === Dataset ===
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


# Public API
__all__ = [
    'TrainingConfig',
    'compare_configs',
    'print_config_diff',
]
