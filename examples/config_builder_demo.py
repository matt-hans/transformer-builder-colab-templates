"""
Config Builder Demo: Comprehensive examples of TrainingConfigBuilder.

This script demonstrates all features of the builder pattern:
- Fluent API with method chaining
- All 5 preset configurations
- Customizing presets
- Progressive validation
- Comparison of different configurations

Run this script to see the builder in action and learn best practices.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training.training_config import (
    TrainingConfig,
    TrainingConfigBuilder,
    compare_configs,
    print_config_diff,
)


def print_section(title: str) -> None:
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_config_summary(config: TrainingConfig, name: str) -> None:
    """Print concise config summary."""
    print(f"\n{name}:")
    print(f"  Model: {config.model_name}")
    print(f"  Architecture: {config.num_layers} layers, {config.d_model} dims, {config.num_heads} heads")
    print(f"  Training: {config.epochs} epochs, LR={config.learning_rate}, batch={config.batch_size}")
    print(f"  Hardware: AMP={config.use_amp}, compile={config.compile_mode}, strategy={config.strategy}")
    print(f"  Export: {config.export_bundle}")


# =============================================================================
# Example 1: Basic Fluent API Usage
# =============================================================================

print_section("Example 1: Basic Fluent API Usage")

print("\nBuilding a custom configuration with method chaining:")
config = (TrainingConfigBuilder()
    .with_model(
        model_name="custom-gpt",
        model_type="gpt",
        d_model=512,
        num_layers=6,
        num_heads=8,
        vocab_size=50257
    )
    .with_training(
        learning_rate=1e-4,
        batch_size=16,
        epochs=15,
        validation_split=0.1
    )
    .with_optimizer(
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=2
    )
    .with_hardware(
        use_amp=True,
        compile_mode="default",
        devices=1
    )
    .with_logging(
        wandb_project="transformer-experiments",
        run_name="custom-config-demo",
        notes="Demonstrating fluent API"
    )
    .build()
)

print_config_summary(config, "Custom Configuration")
print("\nValidation: PASSED")


# =============================================================================
# Example 2: All 5 Presets
# =============================================================================

print_section("Example 2: All 5 Preset Configurations")

# Quick Prototype
quick_config = TrainingConfigBuilder.quick_prototype().build()
print_config_summary(quick_config, "Quick Prototype")
print("  Use case: Fast prototyping, debugging, CI/CD tests")
print("  Runtime: ~5-10 minutes on single GPU")

# Baseline
baseline_config = TrainingConfigBuilder.baseline().build()
print_config_summary(baseline_config, "Baseline")
print("  Use case: Standard starting point, balanced settings")
print("  Runtime: ~2-4 hours on V100/A100")

# Production
production_config = TrainingConfigBuilder.production().build()
print_config_summary(production_config, "Production")
print("  Use case: Final runs, maximum quality, deployment")
print("  Runtime: ~8-12 hours on A100")

# Distributed
distributed_config = TrainingConfigBuilder.distributed().build()
print_config_summary(distributed_config, "Distributed")
print("  Use case: Multi-GPU training, large models")
print("  Runtime: Depends on cluster size")
print("  WARNING: Use CLI, not Jupyter notebooks")

# Low Memory
low_mem_config = TrainingConfigBuilder.low_memory().build()
print_config_summary(low_mem_config, "Low Memory")
print("  Use case: Colab free tier, consumer GPUs, CPU training")
print("  Runtime: ~1-2 hours on Colab T4")


# =============================================================================
# Example 3: Customizing Presets
# =============================================================================

print_section("Example 3: Customizing Presets")

print("\nStarting with baseline preset and customizing:")
custom_baseline = (TrainingConfigBuilder.baseline()
    .with_training(epochs=30, batch_size=8)
    .with_optimizer(gradient_accumulation_steps=4)
    .with_logging(run_name="custom-baseline-v1", notes="Extended training")
    .build()
)

print_config_summary(custom_baseline, "Customized Baseline")
print("\nChanges from baseline preset:")
print("  - Epochs: 10 → 30 (extended training)")
print("  - Batch size: 4 → 8 (higher throughput)")
print("  - Gradient accumulation: 1 → 4 (effective batch = 32)")


# =============================================================================
# Example 4: Progressive Validation
# =============================================================================

print_section("Example 4: Progressive Validation (Error Handling)")

print("\nDemonstrating early error detection:")

# Example 1: Invalid d_model/num_heads
print("\n1. Invalid d_model/num_heads divisibility:")
try:
    TrainingConfigBuilder().with_model(d_model=768, num_heads=5)
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  ✓ Caught error: {str(e)}")

# Example 2: Invalid learning rate
print("\n2. Negative learning rate:")
try:
    TrainingConfigBuilder().with_training(learning_rate=-0.001)
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  ✓ Caught error: {str(e)}")

# Example 3: Invalid validation split
print("\n3. Validation split too large:")
try:
    TrainingConfigBuilder().with_training(validation_split=0.8)
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  ✓ Caught error: {str(e)}")

# Example 4: Invalid compile mode
print("\n4. Invalid compile mode:")
try:
    TrainingConfigBuilder().with_hardware(compile_mode="invalid")
    print("  ERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"  ✓ Caught error: {str(e)}")

print("\nProgressive validation catches errors immediately, before build()!")


# =============================================================================
# Example 5: Comparing Configurations
# =============================================================================

print_section("Example 5: Comparing Configurations")

print("\nComparing Quick Prototype vs Production:")
diff = compare_configs(quick_config, production_config)
print_config_diff(diff)

print("\n\nKey differences explained:")
print("  - Epochs: 3 vs 20 (production trains longer)")
print("  - Deterministic: False vs True (production is reproducible)")
print("  - Export bundle: False vs True (production generates artifacts)")
print("  - Compile mode: None vs 'reduce-overhead' (production optimized)")


# =============================================================================
# Example 6: Common Use Case Patterns
# =============================================================================

print_section("Example 6: Common Use Case Patterns")

# Pattern 1: Hyperparameter Search Baseline
print("\n1. Hyperparameter search (start from baseline):")
hp_search_config = (TrainingConfigBuilder.baseline()
    .with_training(epochs=5, max_train_samples=5000)  # Quick iterations
    .with_logging(run_name="hp-search-lr-1e4")
    .build()
)
print("  ✓ Fast iterations with subset of data")
print(f"  Config: {hp_search_config.epochs} epochs, {hp_search_config.max_train_samples} samples")

# Pattern 2: Multi-GPU Training
print("\n2. Multi-GPU training on 4 GPUs:")
multi_gpu_config = (TrainingConfigBuilder.distributed()
    .with_hardware(devices=4, strategy="ddp")
    .with_model(d_model=768, num_layers=12)  # Reduce model size
    .build()
)
print("  ✓ DDP with 4 GPUs")
print(f"  Effective batch size: {multi_gpu_config.batch_size * 4 * multi_gpu_config.gradient_accumulation_steps}")

# Pattern 3: Colab Free Tier
print("\n3. Colab free tier (T4 GPU, 12GB RAM):")
colab_config = (TrainingConfigBuilder.low_memory()
    .with_training(batch_size=2, max_train_samples=10000)
    .with_optimizer(gradient_accumulation_steps=8)  # Effective batch = 16
    .with_checkpointing(keep_best_only=True)  # Save space
    .build()
)
print("  ✓ Optimized for limited memory")
print(f"  Batch size: {colab_config.batch_size}, accumulation: {colab_config.gradient_accumulation_steps}")

# Pattern 4: Production Deployment
print("\n4. Production deployment with export:")
deploy_config = (TrainingConfigBuilder.production()
    .with_export(
        export_bundle=True,
        export_formats=["onnx", "torchscript"],
        export_dir="./model_exports"
    )
    .with_logging(run_name="final-model-v1.0")
    .build()
)
print("  ✓ Full training with deployment artifacts")
print(f"  Export formats: {', '.join(deploy_config.export_formats)}")

# Pattern 5: Reproducible Research
print("\n5. Reproducible research (deterministic mode):")
research_config = (TrainingConfigBuilder.baseline()
    .with_reproducibility(random_seed=42, deterministic=True)
    .with_logging(run_name="reproducible-exp-1", notes="Deterministic for paper")
    .build()
)
print("  ✓ Bit-exact reproducibility")
print(f"  Deterministic: {research_config.deterministic}, Seed: {research_config.random_seed}")


# =============================================================================
# Example 7: Save and Load with Builder
# =============================================================================

print_section("Example 7: Save and Load Configurations")

print("\nSaving configuration:")
os.makedirs("demo_configs", exist_ok=True)
config_path = production_config.save("demo_configs/production_demo.json")
print(f"  ✓ Saved to: {config_path}")

print("\nLoading configuration:")
loaded_config = TrainingConfig.load(config_path)
print(f"  ✓ Loaded: {loaded_config.model_name}")
print(f"  Validation: {'PASSED' if loaded_config.learning_rate == production_config.learning_rate else 'FAILED'}")


# =============================================================================
# Example 8: Builder Immutability
# =============================================================================

print_section("Example 8: Builder Immutability (Thread-Safe)")

print("\nDemonstrating immutability:")
builder1 = TrainingConfigBuilder().with_training(learning_rate=1e-4)
builder2 = builder1.with_training(batch_size=8)

print(f"  Builder 1 has learning_rate: {'learning_rate' in builder1._config}")
print(f"  Builder 1 has batch_size: {'batch_size' in builder1._config}")
print(f"  Builder 2 has learning_rate: {'learning_rate' in builder2._config}")
print(f"  Builder 2 has batch_size: {'batch_size' in builder2._config}")
print(f"\n  Builder 1 is Builder 2: {builder1 is builder2}")
print("  ✓ Each method call returns a NEW builder (immutable)")


# =============================================================================
# Example 9: Migration from Direct Construction
# =============================================================================

print_section("Example 9: Migration from Direct Construction")

print("\nOLD WAY (direct construction):")
print("""
config = TrainingConfig(
    learning_rate=5e-5,
    batch_size=4,
    epochs=10,
    d_model=768,
    num_layers=12,
    num_heads=12,
    use_amp=True,
    compile_mode="default",
    wandb_project="my-project",
    notes="Old way"
)
""")

print("NEW WAY (builder pattern):")
print("""
config = (TrainingConfigBuilder()
    .with_training(learning_rate=5e-5, batch_size=4, epochs=10)
    .with_model(d_model=768, num_layers=12, num_heads=12)
    .with_hardware(use_amp=True, compile_mode="default")
    .with_logging(wandb_project="my-project", notes="New way with builder")
    .build()
)
""")

print("\nBENEFITS:")
print("  ✓ Organized by concern (model, training, hardware)")
print("  ✓ Progressive validation (errors caught early)")
print("  ✓ Presets for common scenarios")
print("  ✓ Method chaining for readability")
print("  ✓ Thread-safe immutability")


# =============================================================================
# Summary
# =============================================================================

print_section("Summary")

print("""
TrainingConfigBuilder provides a fluent API for creating training configurations:

PRESETS (5 available):
  1. quick_prototype() - Fast iteration, debugging (3 epochs, 12M params)
  2. baseline()        - Standard config, balanced (10 epochs, 125M params)
  3. production()      - High quality, reproducible (20 epochs, export enabled)
  4. distributed()     - Multi-GPU training (DDP/FSDP, 4 GPUs default)
  5. low_memory()      - Colab free tier, small GPUs (2 batch, 8x accumulation)

METHODS (11 categories):
  - with_model()          - Architecture (d_model, layers, heads)
  - with_training()       - Hyperparameters (LR, batch, epochs)
  - with_optimizer()      - Optimizer settings (weight decay, warmup)
  - with_scheduler()      - LR schedule configuration
  - with_hardware()       - GPU, AMP, compilation, distributed
  - with_logging()        - W&B, run names, notes
  - with_checkpointing()  - Save frequency, best-only
  - with_export()         - ONNX, TorchScript export
  - with_reproducibility() - Seed, deterministic mode
  - with_dataset()        - Dataset, task selection
  - build()               - Create final validated config

FEATURES:
  ✓ Progressive validation - errors caught early
  ✓ Immutable builder - thread-safe
  ✓ Method chaining - readable code
  ✓ Preset customization - start from template
  ✓ Backward compatible - TrainingConfig() still works

NEXT STEPS:
  1. Choose a preset or start from scratch
  2. Customize with .with_*() methods
  3. Call .build() to create config
  4. Use config with test_fine_tuning() or run_training()

See USAGE_GUIDE_COLAB_AND_CLI.md for more examples!
""")

print("=" * 80)
print("Demo complete! All configurations validated successfully.")
print("=" * 80)
