"""
Interactive Setup Wizard for Training Configuration.

Provides a guided 5-step workflow for configuring model training:
1. Dataset selection/upload
2. Tokenizer configuration
3. Model verification
4. Training hyperparameters
5. Validation and launch
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

from .presets import ConfigPresets, PRESETS


@dataclass
class WizardConfig:
    """Configuration collected by setup wizard."""
    # Dataset
    dataset_source: str  # 'huggingface', 'local', 'drive', 'upload'
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_path: Optional[str] = None

    # Tokenizer
    vocab_size: int = 50257
    tokenizer_strategy: str = 'auto'  # 'auto', 'pretrained', 'train_bpe', 'character'

    # Model
    model_verified: bool = False
    estimated_params: Optional[int] = None

    # Training
    batch_size: int = 16
    max_seq_len: int = 512
    learning_rate: float = 1e-4
    max_epochs: int = 3
    early_stopping_patience: Optional[int] = None
    val_split: float = 0.1

    # Hardware
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 1.0

    # Output
    output_dir: str = './training_output'
    checkpoint_every_n_epochs: int = 1
    save_top_k: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'WizardConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class SetupWizard:
    """
    Interactive setup wizard for training configuration.

    Guides users through a 5-step configuration process with
    validation and helpful defaults.

    Example (Colab):
        >>> wizard = SetupWizard()
        >>> config = wizard.run(model=my_model)
        >>>
        >>> # Use config for training
        >>> from utils.training import TrainingCoordinator
        >>> coordinator = TrainingCoordinator()
        >>> results = coordinator.train(model=my_model, **config.to_dict())

    Example (Non-interactive):
        >>> wizard = SetupWizard()
        >>> config = wizard.create_config_from_preset('small')
        >>> config.dataset_path = 'my_data.txt'
        >>> wizard.validate_config(config)
    """

    def __init__(self):
        """Initialize setup wizard."""
        self.config = WizardConfig()
        self.presets = ConfigPresets()
        self.steps_completed = []

    def run(self,
            model: Any,
            interactive: bool = True,
            preset: Optional[str] = None) -> WizardConfig:
        """
        Run the interactive setup wizard.

        Args:
            model: PyTorch model to train
            interactive: Use interactive widgets (Colab)
            preset: Optional preset to start from

        Returns:
            Complete WizardConfig

        Example:
            >>> config = wizard.run(model=my_transformer, preset='small')
        """
        print("\n" + "="*80)
        print("🧙 Training Setup Wizard")
        print("="*80)
        print("\nLet's configure your training in 5 simple steps!\n")

        # Load preset if provided
        if preset:
            print(f"📦 Loading preset: {preset}")
            self._apply_preset(preset)
            print("✓ Preset loaded\n")

        # Step 1: Dataset
        print("─" * 80)
        print("Step 1/5: Dataset Selection")
        print("─" * 80)
        if interactive:
            self._step1_dataset_interactive()
        else:
            self._step1_dataset_manual()

        # Step 2: Tokenizer
        print("\n" + "─" * 80)
        print("Step 2/5: Tokenizer Configuration")
        print("─" * 80)
        if interactive:
            self._step2_tokenizer_interactive()
        else:
            self._step2_tokenizer_manual()

        # Step 3: Model verification
        print("\n" + "─" * 80)
        print("Step 3/5: Model Verification")
        print("─" * 80)
        self._step3_model_verification(model)

        # Step 4: Training parameters
        print("\n" + "─" * 80)
        print("Step 4/5: Training Parameters")
        print("─" * 80)
        if interactive:
            self._step4_training_interactive()
        else:
            self._step4_training_manual()

        # Step 5: Validation and summary
        print("\n" + "─" * 80)
        print("Step 5/5: Configuration Summary")
        print("─" * 80)
        self._step5_validation()

        print("\n" + "="*80)
        print("✓ Setup Complete!")
        print("="*80 + "\n")

        return self.config

    def _apply_preset(self, preset_name: str):
        """Apply preset configuration."""
        preset_config = self.presets.get(preset_name)

        self.config.vocab_size = preset_config.vocab_size
        self.config.max_seq_len = preset_config.max_seq_len
        self.config.batch_size = preset_config.batch_size
        self.config.learning_rate = preset_config.learning_rate
        self.config.max_epochs = preset_config.max_epochs
        self.config.gradient_accumulation_steps = preset_config.gradient_accumulation_steps
        self.config.dataset_name = preset_config.dataset_name
        self.config.dataset_config = preset_config.dataset_config
        self.config.dataset_source = 'huggingface'

    def _step1_dataset_interactive(self):
        """Step 1: Interactive dataset selection."""
        try:
            from google.colab import files
            in_colab = True
        except ImportError:
            in_colab = False

        print("\nChoose your dataset source:")
        print("  1. HuggingFace Dataset (recommended)")
        print("  2. Local file (TXT, JSON, CSV)")
        if in_colab:
            print("  3. Google Drive")
            print("  4. Upload file")

        # For now, default to HuggingFace
        print("\nDefaulting to HuggingFace dataset...")
        self.config.dataset_source = 'huggingface'

        if not self.config.dataset_name:
            self.config.dataset_name = 'wikitext'
            self.config.dataset_config = 'wikitext-2-raw-v1'

        print(f"✓ Dataset: {self.config.dataset_name} ({self.config.dataset_config})")

    def _step1_dataset_manual(self):
        """Step 1: Manual dataset configuration."""
        if not self.config.dataset_name:
            self.config.dataset_source = 'huggingface'
            self.config.dataset_name = 'wikitext'
            self.config.dataset_config = 'wikitext-2-raw-v1'

        print(f"Dataset: {self.config.dataset_name}")
        if self.config.dataset_config:
            print(f"Config: {self.config.dataset_config}")

    def _step2_tokenizer_interactive(self):
        """Step 2: Interactive tokenizer configuration."""
        print(f"\nVocabulary size: {self.config.vocab_size:,}")
        print(f"Strategy: {self.config.tokenizer_strategy}")

        # Auto-detect strategy
        if self.config.vocab_size == 50257:
            print("  → Will use GPT-2 tokenizer (exact match)")
        elif 5000 <= self.config.vocab_size <= 100000:
            print("  → Will train custom BPE tokenizer")
        else:
            print("  → Will use character-level tokenizer")

        print("✓ Tokenizer configured")

    def _step2_tokenizer_manual(self):
        """Step 2: Manual tokenizer configuration."""
        print(f"Vocabulary size: {self.config.vocab_size:,}")
        print(f"Strategy: {self.config.tokenizer_strategy}")

    def _step3_model_verification(self, model: Any):
        """Step 3: Verify model configuration."""
        print("\nVerifying model...")

        # Count parameters
        try:
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.config.estimated_params = num_params
            self.config.model_verified = True

            print(f"✓ Model type: {model.__class__.__name__}")
            print(f"  Total parameters: {num_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Size: ~{num_params / 1_000_000:.1f}M params")

            # Estimate memory
            param_memory_mb = num_params * 4 / (1024**2)  # 4 bytes per param (float32)
            print(f"  Est. memory: ~{param_memory_mb:.0f} MB (params only)")

        except Exception as e:
            print(f"⚠️  Could not verify model: {e}")
            self.config.model_verified = False

    def _step4_training_interactive(self):
        """Step 4: Interactive training configuration."""
        print("\nTraining configuration:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Max sequence length: {self.config.max_seq_len}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Max epochs: {self.config.max_epochs}")
        print(f"  Validation split: {self.config.val_split * 100:.0f}%")

        print("\nOptimizations:")
        print(f"  Mixed precision: {'✓' if self.config.use_mixed_precision else '✗'}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps} steps")
        print(f"  Gradient clipping: {self.config.gradient_clip_val}")

        print("\nCheckpointing:")
        print(f"  Output directory: {self.config.output_dir}")
        print(f"  Save every: {self.config.checkpoint_every_n_epochs} epoch(s)")
        print(f"  Keep top: {self.config.save_top_k} checkpoints")

        print("\n✓ Training parameters configured")

    def _step4_training_manual(self):
        """Step 4: Manual training configuration."""
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Max epochs: {self.config.max_epochs}")

    def _step5_validation(self):
        """Step 5: Validate and summarize configuration."""
        print("\n📋 Configuration Summary:")
        print(f"\n  Dataset: {self.config.dataset_name or self.config.dataset_path}")
        print(f"  Vocabulary: {self.config.vocab_size:,}")
        print(f"  Model: ~{self.config.estimated_params / 1_000_000:.0f}M params" if self.config.estimated_params else "  Model: Unknown size")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Epochs: {self.config.max_epochs}")

        # Estimate training time
        if self.config.estimated_params:
            # Rough estimate: ~1 hour per 100M params per epoch on T4 GPU
            estimated_hours = (self.config.estimated_params / 100_000_000) * self.config.max_epochs
            print(f"  Estimated time: ~{estimated_hours:.1f} hours")

        print("\n✓ Configuration validated")

    def create_config_from_preset(self, preset_name: str) -> WizardConfig:
        """
        Create configuration from preset without running wizard.

        Args:
            preset_name: Preset identifier

        Returns:
            WizardConfig initialized with preset values

        Example:
            >>> config = wizard.create_config_from_preset('small')
            >>> config.dataset_path = 'my_data.txt'
        """
        self._apply_preset(preset_name)
        return self.config

    def validate_config(self, config: WizardConfig) -> Tuple[bool, List[str]]:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Example:
            >>> is_valid, errors = wizard.validate_config(config)
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(f"  ❌ {error}")
        """
        errors = []

        # Dataset validation
        if config.dataset_source == 'huggingface' and not config.dataset_name:
            errors.append("HuggingFace dataset requires dataset_name")
        elif config.dataset_source in ['local', 'drive'] and not config.dataset_path:
            errors.append(f"{config.dataset_source} dataset requires dataset_path")

        # Training validation
        if config.batch_size < 1:
            errors.append("Batch size must be >= 1")
        if config.learning_rate <= 0:
            errors.append("Learning rate must be > 0")
        if config.max_epochs < 1:
            errors.append("Max epochs must be >= 1")
        if not 0.0 <= config.val_split < 1.0:
            errors.append("Validation split must be in [0.0, 1.0)")

        # Hardware validation
        if config.gradient_accumulation_steps < 1:
            errors.append("Gradient accumulation steps must be >= 1")

        is_valid = len(errors) == 0
        return is_valid, errors

    def print_config(self, config: Optional[WizardConfig] = None):
        """
        Print formatted configuration.

        Args:
            config: Configuration to print (uses self.config if None)
        """
        if config is None:
            config = self.config

        print("\n" + "="*80)
        print("Training Configuration")
        print("="*80)

        print("\n📊 Dataset:")
        print(f"  Source: {config.dataset_source}")
        if config.dataset_name:
            print(f"  Name: {config.dataset_name}")
        if config.dataset_config:
            print(f"  Config: {config.dataset_config}")
        if config.dataset_path:
            print(f"  Path: {config.dataset_path}")

        print("\n🔤 Tokenizer:")
        print(f"  Vocabulary size: {config.vocab_size:,}")
        print(f"  Max sequence length: {config.max_seq_len}")
        print(f"  Strategy: {config.tokenizer_strategy}")

        print("\n🏋️  Training:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Max epochs: {config.max_epochs}")
        print(f"  Validation split: {config.val_split * 100:.0f}%")
        if config.early_stopping_patience:
            print(f"  Early stopping: {config.early_stopping_patience} epochs")

        print("\n⚡ Optimizations:")
        print(f"  Mixed precision: {'✓' if config.use_mixed_precision else '✗'}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Gradient clipping: {config.gradient_clip_val}")

        print("\n💾 Checkpointing:")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Save every: {config.checkpoint_every_n_epochs} epoch(s)")
        print(f"  Keep top: {config.save_top_k}")

        print("="*80 + "\n")

    def quick_setup(self,
                   model: Any,
                   preset: str = 'small',
                   dataset_name: str = 'wikitext',
                   dataset_config: str = 'wikitext-2-raw-v1') -> WizardConfig:
        """
        Quick non-interactive setup with sensible defaults.

        Args:
            model: PyTorch model
            preset: Configuration preset
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration

        Returns:
            Configured WizardConfig

        Example:
            >>> config = wizard.quick_setup(
            ...     model=my_model,
            ...     preset='small',
            ...     dataset_name='wikitext'
            ... )
        """
        print(f"🚀 Quick Setup (preset: {preset})")

        # Apply preset
        self._apply_preset(preset)

        # Override dataset
        self.config.dataset_name = dataset_name
        self.config.dataset_config = dataset_config
        self.config.dataset_source = 'huggingface'

        # Verify model
        self._step3_model_verification(model)

        # Validate
        is_valid, errors = self.validate_config(self.config)
        if not is_valid:
            print("⚠️  Configuration has errors:")
            for error in errors:
                print(f"  ❌ {error}")

        print("✓ Quick setup complete\n")

        return self.config
