"""
Configuration Presets for Common Training Scenarios.

Provides quick-start configurations for different model sizes and use cases.
"""

from typing import Dict, Any, Literal
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    name: str
    description: str

    # Model configuration
    vocab_size: int
    max_seq_len: int

    # Training parameters
    batch_size: int
    learning_rate: float
    max_epochs: int
    warmup_steps: int

    # Dataset
    dataset_name: str
    dataset_config: str

    # Hardware
    gradient_accumulation_steps: int
    precision: str

    # Estimated metrics
    estimated_time_hours: float
    estimated_params_millions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Predefined configurations
PRESETS = {
    'tiny': TrainingConfig(
        name='Tiny (Debug/Testing)',
        description='Ultra-fast training for testing and debugging. Trains in ~1 hour on Colab free tier.',
        vocab_size=10000,
        max_seq_len=256,
        batch_size=32,
        learning_rate=5e-4,
        max_epochs=3,
        warmup_steps=100,
        dataset_name='wikitext',
        dataset_config='wikitext-2-raw-v1',
        gradient_accumulation_steps=1,
        precision='16',
        estimated_time_hours=1.0,
        estimated_params_millions=10
    ),

    'small': TrainingConfig(
        name='Small (Educational)',
        description='Good for learning and experimentation. Similar to GPT-2 small (117M params).',
        vocab_size=50257,
        max_seq_len=512,
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=5,
        warmup_steps=500,
        dataset_name='wikitext',
        dataset_config='wikitext-103-raw-v1',
        gradient_accumulation_steps=2,
        precision='16',
        estimated_time_hours=4.0,
        estimated_params_millions=125
    ),

    'medium': TrainingConfig(
        name='Medium (Production)',
        description='Production-quality model similar to GPT-2 medium (345M params).',
        vocab_size=50257,
        max_seq_len=1024,
        batch_size=8,
        learning_rate=5e-5,
        max_epochs=10,
        warmup_steps=1000,
        dataset_name='openwebtext',
        dataset_config=None,
        gradient_accumulation_steps=4,
        precision='16',
        estimated_time_hours=12.0,
        estimated_params_millions=350
    ),

    'large': TrainingConfig(
        name='Large (Research)',
        description='Large-scale model similar to GPT-2 large (774M params). Requires Colab Pro+.',
        vocab_size=50257,
        max_seq_len=1024,
        batch_size=4,
        learning_rate=2e-5,
        max_epochs=20,
        warmup_steps=2000,
        dataset_name='openwebtext',
        dataset_config=None,
        gradient_accumulation_steps=8,
        precision='16',
        estimated_time_hours=48.0,
        estimated_params_millions=774
    ),

    # Task-specific presets
    'code_generation': TrainingConfig(
        name='Code Generation',
        description='Optimized for code generation tasks (Python, JavaScript, etc.).',
        vocab_size=50257,
        max_seq_len=2048,
        batch_size=8,
        learning_rate=1e-4,
        max_epochs=5,
        warmup_steps=500,
        dataset_name='code_search_net',
        dataset_config='python',
        gradient_accumulation_steps=2,
        precision='16',
        estimated_time_hours=8.0,
        estimated_params_millions=125
    ),

    'chat': TrainingConfig(
        name='Chat/Dialogue',
        description='Optimized for conversational AI and dialogue systems.',
        vocab_size=50257,
        max_seq_len=512,
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=10,
        warmup_steps=1000,
        dataset_name='daily_dialog',
        dataset_config=None,
        gradient_accumulation_steps=2,
        precision='16',
        estimated_time_hours=6.0,
        estimated_params_millions=125
    ),

    'summarization': TrainingConfig(
        name='Summarization',
        description='Optimized for text summarization tasks.',
        vocab_size=50257,
        max_seq_len=1024,
        batch_size=8,
        learning_rate=5e-5,
        max_epochs=5,
        warmup_steps=500,
        dataset_name='cnn_dailymail',
        dataset_config='3.0.0',
        gradient_accumulation_steps=4,
        precision='16',
        estimated_time_hours=10.0,
        estimated_params_millions=125
    ),
}


class ConfigPresets:
    """
    Configuration preset manager.

    Provides easy access to predefined training configurations
    and allows customization.

    Example:
        >>> presets = ConfigPresets()
        >>>
        >>> # Get tiny preset
        >>> config = presets.get('tiny')
        >>> print(config.description)
        >>>
        >>> # Customize preset
        >>> custom = presets.customize('small', max_epochs=10, batch_size=32)
        >>>
        >>> # Apply to training
        >>> from utils.training import train_model
        >>> results = train_model(model=my_model, **config.to_dict())
    """

    def __init__(self):
        """Initialize preset manager."""
        self.presets = PRESETS

    def list_presets(self) -> Dict[str, str]:
        """
        List all available presets with descriptions.

        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {
            name: config.description
            for name, config in self.presets.items()
        }

    def get(self, preset_name: str) -> TrainingConfig:
        """
        Get configuration by preset name.

        Args:
            preset_name: Preset identifier

        Returns:
            TrainingConfig object

        Raises:
            KeyError: If preset not found
        """
        if preset_name not in self.presets:
            available = ', '.join(self.presets.keys())
            raise KeyError(
                f"Preset '{preset_name}' not found. "
                f"Available presets: {available}"
            )

        return self.presets[preset_name]

    def customize(self,
                  preset_name: str,
                  **overrides) -> TrainingConfig:
        """
        Get preset with custom overrides.

        Args:
            preset_name: Base preset to customize
            **overrides: Fields to override

        Returns:
            Customized TrainingConfig

        Example:
            >>> config = presets.customize(
            ...     'small',
            ...     max_epochs=10,
            ...     learning_rate=5e-4
            ... )
        """
        base_config = self.get(preset_name)
        config_dict = base_config.to_dict()
        config_dict.update(overrides)

        return TrainingConfig(**config_dict)

    def print_preset(self, preset_name: str):
        """
        Print detailed preset information.

        Args:
            preset_name: Preset to display
        """
        config = self.get(preset_name)

        print(f"\n{'='*80}")
        print(f"Preset: {config.name}")
        print(f"{'='*80}")
        print(f"\n{config.description}\n")

        print("Model Configuration:")
        print(f"  Vocabulary Size: {config.vocab_size:,}")
        print(f"  Max Sequence Length: {config.max_seq_len}")
        print(f"  Est. Parameters: ~{config.estimated_params_millions}M")

        print("\nTraining Parameters:")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Max Epochs: {config.max_epochs}")
        print(f"  Warmup Steps: {config.warmup_steps}")
        print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
        print(f"  Precision: {config.precision}-bit")

        print("\nDataset:")
        print(f"  Name: {config.dataset_name}")
        if config.dataset_config:
            print(f"  Config: {config.dataset_config}")

        print(f"\nEstimated Training Time: ~{config.estimated_time_hours:.1f} hours")
        print(f"{'='*80}\n")

    def print_all_presets(self):
        """Print summary of all available presets."""
        print("\n" + "="*80)
        print("Available Training Presets")
        print("="*80 + "\n")

        for name, config in self.presets.items():
            print(f"📦 {config.name}")
            print(f"   {config.description}")
            print(f"   Time: ~{config.estimated_time_hours:.1f}h | "
                  f"Params: ~{config.estimated_params_millions}M | "
                  f"Dataset: {config.dataset_name}")
            print()

        print("Usage:")
        print("  presets = ConfigPresets()")
        print("  config = presets.get('small')")
        print("  presets.print_preset('small')  # Show details")
        print("="*80 + "\n")

    def get_recommendation(self,
                          goal: Literal['learning', 'production', 'research', 'quick_test'],
                          time_budget_hours: float = None) -> str:
        """
        Get preset recommendation based on goal and constraints.

        Args:
            goal: Training goal
            time_budget_hours: Available training time

        Returns:
            Recommended preset name
        """
        recommendations = {
            'quick_test': 'tiny',
            'learning': 'small',
            'production': 'medium',
            'research': 'large',
        }

        recommended = recommendations.get(goal, 'small')

        # Adjust based on time budget
        if time_budget_hours:
            config = self.get(recommended)
            if config.estimated_time_hours > time_budget_hours:
                # Find fastest preset that fits
                for preset_name in ['tiny', 'small', 'medium', 'large']:
                    preset = self.get(preset_name)
                    if preset.estimated_time_hours <= time_budget_hours:
                        recommended = preset_name
                    else:
                        break

        return recommended
