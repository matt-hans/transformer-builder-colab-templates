"""
Example: Using TrainingConfig for Reproducible Experiments

This example demonstrates the complete workflow for managing training
configurations with version control, validation, and W&B integration.

Key features demonstrated:
- Creating and validating configurations
- Saving configs with timestamps for versioning
- Loading configs to reproduce experiments
- Comparing configs to track changes
- W&B integration for experiment tracking
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training.training_config import TrainingConfig, compare_configs, print_config_diff
from utils.training.seed_manager import set_random_seed

# ==============================================================================
# Example 1: Create and Validate Configuration
# ==============================================================================

print("=" * 70)
print("Example 1: Create and Validate Configuration")
print("=" * 70)

# Create a training configuration with custom hyperparameters
config = TrainingConfig(
    # Hyperparameters
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,

    # Model architecture
    model_name="gpt-50M-wikitext",
    model_type="gpt",
    vocab_size=50257,
    max_seq_len=128,
    d_model=512,
    num_layers=8,
    num_heads=8,

    # Dataset
    dataset_name="wikitext-103-v1",
    validation_split=0.1,

    # Reproducibility
    random_seed=42,
    deterministic=False,  # Fast mode

    # Experiment tracking
    wandb_project="transformer-builder-training",
    run_name="gpt-50M-baseline",

    # Notes
    notes="Baseline experiment with standard hyperparameters",
)

# Validate configuration before training
try:
    config.validate()
    print("✅ Configuration is valid!")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Model: {config.model_type} with {config.num_layers} layers")
    print(f"   Random seed: {config.random_seed}")
except ValueError as e:
    print(f"❌ Configuration invalid:\n{e}")
    exit(1)

print()

# ==============================================================================
# Example 2: Save Configuration with Versioning
# ==============================================================================

print("=" * 70)
print("Example 2: Save Configuration with Versioning")
print("=" * 70)

# Option 1: Auto-generate timestamped filename
config_path_auto = config.save()
print(f"Saved with auto-generated name: {config_path_auto}")

# Option 2: Specify custom path
os.makedirs("experiments", exist_ok=True)  # Create directory if needed
config_path_custom = config.save("experiments/baseline_config.json")
print(f"Saved to custom path: {config_path_custom}")

print()

# ==============================================================================
# Example 3: Use Configuration to Initialize Training
# ==============================================================================

print("=" * 70)
print("Example 3: Use Configuration to Initialize Training")
print("=" * 70)

# Set random seed from config for reproducibility
set_random_seed(config.random_seed, config.deterministic)

# Initialize W&B with config (if W&B is available and configured)
try:
    import wandb

    # Check if W&B is properly configured (offline mode or logged in)
    if os.environ.get("WANDB_MODE") == "offline" or os.path.exists(os.path.expanduser("~/.netrc")):
        # Initialize run with config
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.to_dict(),
            tags=["baseline", "reproducibility-demo"],
        )

        print(f"✅ W&B initialized with project: {config.wandb_project}")
        print(f"   Run name: {config.run_name}")

        # Save config as W&B artifact for versioning
        config_artifact = wandb.Artifact(
            name=f"{wandb.run.name}-config",
            type="config",
            description="Training configuration",
        )
        config_artifact.add_file(config_path_custom)
        wandb.log_artifact(config_artifact)

        print(f"✅ Config saved as W&B artifact")

        # Clean up W&B run (for demo)
        wandb.finish()
    else:
        print("⚠️ W&B not configured - skipping W&B integration")
        print("   (This is OK for testing - config still works without W&B)")

except ImportError:
    print("⚠️ W&B not installed - skipping W&B integration")
except Exception as e:
    print(f"⚠️ W&B error (this is OK for demo): {e}")

print()

# ==============================================================================
# Example 4: Load Configuration to Reproduce Experiment
# ==============================================================================

print("=" * 70)
print("Example 4: Load Configuration to Reproduce Experiment")
print("=" * 70)

# Later: Load config to reproduce exact training setup
loaded_config = TrainingConfig.load(config_path_custom)

print(f"✅ Loaded config from {config_path_custom}")
print(f"   Learning rate: {loaded_config.learning_rate}")
print(f"   Batch size: {loaded_config.batch_size}")
print(f"   Random seed: {loaded_config.random_seed}")
print(f"   Notes: {loaded_config.notes}")

# Verify it matches original
assert loaded_config.learning_rate == config.learning_rate
assert loaded_config.batch_size == config.batch_size
assert loaded_config.random_seed == config.random_seed

print("✅ Loaded config matches original - reproducibility confirmed!")

print()

# ==============================================================================
# Example 5: Compare Configurations Between Experiments
# ==============================================================================

print("=" * 70)
print("Example 5: Compare Configurations Between Experiments")
print("=" * 70)

# Baseline experiment
baseline = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    notes="Baseline with standard settings"
)

# Experiment 1: Increase learning rate and batch size
experiment_1 = TrainingConfig(
    learning_rate=1e-4,  # Doubled
    batch_size=8,        # Doubled
    epochs=10,
    notes="Experiment 1: Higher LR and batch size"
)

# Compare configurations
print("\nComparing baseline vs experiment 1:")
diff = compare_configs(baseline, experiment_1)
print_config_diff(diff)

# You can also programmatically access the differences:
if diff['changed']:
    print("\nProgrammatic access to changes:")
    for field, (old_val, new_val) in diff['changed'].items():
        print(f"  - {field}: {old_val} → {new_val}")

print()

# ==============================================================================
# Example 6: Validate Configuration Catches Errors
# ==============================================================================

print("=" * 70)
print("Example 6: Validation Catches Invalid Configurations")
print("=" * 70)

# Create invalid configuration (negative learning rate)
try:
    invalid_config = TrainingConfig(
        learning_rate=-0.001,  # Invalid!
        batch_size=0,          # Invalid!
        epochs=0,              # Invalid!
    )
    invalid_config.validate()
    print("❌ Validation should have failed!")
except ValueError as e:
    print("✅ Validation correctly caught errors:")
    print(str(e))

print()

# ==============================================================================
# Example 7: Invalid Architecture Configuration
# ==============================================================================

print("=" * 70)
print("Example 7: Validate Transformer Architecture Constraints")
print("=" * 70)

# d_model must be divisible by num_heads
try:
    bad_architecture = TrainingConfig(
        d_model=768,
        num_heads=5,  # 768 % 5 != 0 - invalid!
    )
    bad_architecture.validate()
    print("❌ Validation should have failed!")
except ValueError as e:
    print("✅ Validation correctly caught architecture error:")
    print(str(e))

print()

# ==============================================================================
# Example 8: Complete Training Workflow
# ==============================================================================

print("=" * 70)
print("Example 8: Complete Training Workflow")
print("=" * 70)

# Step 1: Create config
workflow_config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    random_seed=42,
    notes="End-to-end workflow demo"
)

# Step 2: Validate
workflow_config.validate()
print("✅ Step 1: Config created and validated")

# Step 3: Save for reproducibility
workflow_path = workflow_config.save("experiments/workflow_demo.json")
print(f"✅ Step 2: Config saved to {workflow_path}")

# Step 4: Set seed
set_random_seed(workflow_config.random_seed, workflow_config.deterministic)
print(f"✅ Step 3: Random seed set to {workflow_config.random_seed}")

# Step 5: Initialize W&B (if available)
print("✅ Step 4: Ready to initialize W&B and start training")

# Step 6: Start training with config values
print("✅ Step 5: Config ready for training loop")
print(f"   Training for {workflow_config.epochs} epochs")
print(f"   Batch size: {workflow_config.batch_size}")
print(f"   Learning rate: {workflow_config.learning_rate}")

print()
print("=" * 70)
print("Examples Complete!")
print("=" * 70)
print("\nKey Takeaways:")
print("1. Always validate configs before training")
print("2. Save configs with timestamps for version control")
print("3. Use config.to_dict() for W&B integration")
print("4. Load configs to reproduce experiments exactly")
print("5. Compare configs to track experiment changes")
print("6. Validation catches errors early (before training starts)")
