"""
Gradient Accumulation Example

Demonstrates how to use GradientAccumulator for efficient training with limited GPU memory.

This example shows:
1. Basic manual accumulation
2. Integration with MetricsTracker for W&B logging
3. PyTorch Lightning integration (commented out)
4. Checkpointing and resume

Run:
    python examples/gradient_accumulation_example.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import GradientAccumulator
from utils.training.engine import GradientAccumulator
from utils.training.metrics_tracker import MetricsTracker


# Example 1: Basic Manual Accumulation
def example_basic_accumulation():
    """
    Example: Train with gradient accumulation to simulate larger batch size.

    Setup:
    - Physical batch size: 8
    - Accumulation steps: 4
    - Effective batch size: 32 (8 * 4)
    """
    print("\n" + "="*70)
    print("Example 1: Basic Manual Accumulation")
    print("="*70)

    # Create simple model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize gradient accumulator
    accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=4,      # Accumulate over 4 batches
        max_grad_norm=1.0,         # Gradient clipping
        batch_size=8               # Physical batch size
    )

    print(f"Physical batch size: 8")
    print(f"Accumulation steps: 4")
    print(f"Effective batch size: {accumulator.effective_batch_size}")
    print()

    # Create dummy dataset
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)

            # Accumulate gradients
            should_step = accumulator.accumulate(
                loss=loss,
                model=model,
                is_final_batch=(batch_idx == len(dataloader) - 1)
            )

            # Log when optimizer steps
            if should_step:
                stats = accumulator.stats
                print(
                    f"  Batch {batch_idx:2d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Grad norm: {stats.last_grad_norm:.2e} | "
                    f"Effective step: {accumulator.effective_step}"
                )

        # Reset for next epoch
        accumulator.reset_epoch()

    print(f"\nFinal effective steps: {accumulator.effective_step}")
    print("✅ Basic accumulation complete!")


# Example 2: Integration with MetricsTracker
def example_with_metrics_tracker():
    """
    Example: Use GradientAccumulator with MetricsTracker for W&B logging.

    Setup:
    - MetricsTracker logs metrics at effective steps
    - W&B commits reduced by 75% (only at accumulation boundaries)
    """
    print("\n" + "="*70)
    print("Example 2: Integration with MetricsTracker")
    print("="*70)

    # Create simple model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize tracker and accumulator
    tracker = MetricsTracker(
        use_wandb=False,  # Set to True for actual W&B logging
        gradient_accumulation_steps=4
    )

    accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=4,
        max_grad_norm=1.0,
        batch_size=8
    )

    print(f"MetricsTracker initialized (W&B disabled for demo)")
    print(f"Accumulation steps: 4 → 75% log volume reduction")
    print()

    # Create dummy dataset
    dataset = TensorDataset(torch.randn(32, 10), torch.randn(32, 10))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training loop
    model.train()
    print("Training with metrics logging:")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)

        # Accumulate gradients
        should_step = accumulator.accumulate(
            loss=loss,
            model=model,
            is_final_batch=(batch_idx == len(dataloader) - 1)
        )

        # Log metrics at batch level
        # MetricsTracker calculates effective_step internally
        tracker.log_scalar('train/batch_loss', loss.item(), step=batch_idx)

        if should_step:
            # Log additional metrics at optimizer step
            tracker.log_scalar(
                'train/grad_norm',
                accumulator.stats.last_grad_norm,
                step=batch_idx
            )
            print(
                f"  Batch {batch_idx} | "
                f"Loss: {loss.item():.4f} | "
                f"Effective step: {accumulator.effective_step} | "
                f"✓ W&B commit"
            )
        else:
            print(
                f"  Batch {batch_idx} | "
                f"Loss: {loss.item():.4f} | "
                f"Effective step: {accumulator.effective_step} | "
                f"  (accumulating)"
            )

    # Retrieve metrics
    step_metrics = tracker.get_step_metrics()
    print(f"\nTotal metrics logged: {len(step_metrics)}")
    print(f"Unique effective steps: {step_metrics['effective_step'].nunique()}")
    print("✅ Metrics integration complete!")


# Example 3: Checkpointing and Resume
def example_checkpointing():
    """
    Example: Save and load accumulator state for training resume.

    Setup:
    - Train for 1 epoch
    - Save checkpoint with accumulator state
    - Resume training from checkpoint
    """
    print("\n" + "="*70)
    print("Example 3: Checkpointing and Resume")
    print("="*70)

    # Create simple model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize accumulator
    accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=4,
        max_grad_norm=1.0,
        batch_size=8
    )

    # Create dummy dataset
    dataset = TensorDataset(torch.randn(24, 10), torch.randn(24, 10))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Train for 1 epoch
    print("Training initial epoch:")
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        accumulator.accumulate(loss, model, is_final_batch=(batch_idx == len(dataloader) - 1))

    print(f"Effective steps after epoch 1: {accumulator.effective_step}")

    # Save checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accumulator': accumulator.state_dict(),
        'epoch': 1
    }
    checkpoint_path = '/tmp/gradient_accumulator_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"\n✓ Checkpoint saved to {checkpoint_path}")

    # Create new model and accumulator (simulate restart)
    print("\nSimulating training resume...")
    model_new = nn.Linear(10, 10)
    optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-4)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_new.load_state_dict(checkpoint['model'])
    optimizer_new.load_state_dict(checkpoint['optimizer'])

    # Create new accumulator and load state
    accumulator_new = GradientAccumulator(
        optimizer=optimizer_new,
        accumulation_steps=4,
        max_grad_norm=1.0,
        batch_size=8
    )
    accumulator_new.load_state_dict(checkpoint['accumulator'])

    print(f"✓ Loaded accumulator state")
    print(f"Resumed from effective step: {accumulator_new.effective_step}")

    # Continue training
    print("\nContinuing training:")
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model_new(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        accumulator_new.accumulate(loss, model_new, is_final_batch=(batch_idx == len(dataloader) - 1))

    print(f"Effective steps after epoch 2: {accumulator_new.effective_step}")
    print("✅ Checkpoint resume complete!")


# Example 4: PyTorch Lightning Integration (commented out)
def example_lightning_integration():
    """
    Example: Use GradientAccumulator with PyTorch Lightning.

    Note: This requires pytorch_lightning to be installed.
    Uncomment the code below to run with Lightning.
    """
    print("\n" + "="*70)
    print("Example 4: PyTorch Lightning Integration")
    print("="*70)

    print("⚠️  This example requires pytorch_lightning to be installed.")
    print("Install: pip install pytorch-lightning")
    print("\nExample code (commented out):")
    print("""
# import pytorch_lightning as pl

# Create Lightning trainer with accumulation
# trainer = pl.Trainer(
#     accelerator='gpu',
#     devices=1,
#     accumulate_grad_batches=4,  # Lightning manages accumulation
#     max_epochs=10
# )

# Create model and optimizer
# model = nn.Linear(10, 10)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# GradientAccumulator detects Lightning and delegates
# accumulator = GradientAccumulator(
#     optimizer=optimizer,
#     accumulation_steps=1,       # Disabled (Lightning controls)
#     trainer=trainer             # Automatic detection
# )

# Check detection
# assert accumulator.is_lightning_managed
# print(f"✓ Lightning-managed accumulation detected")
# print(f"Effective batch size: {accumulator.effective_batch_size}")

# In training loop, always returns True (Lightning controls)
# for batch in dataloader:
#     loss = model(batch)
#     should_step = accumulator.accumulate(loss, model)  # Always True
#     print(f"Effective step: {accumulator.effective_step}")
    """)

    print("\n✅ See docs/gradient_accumulation_guide.md for complete Lightning example")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Gradient Accumulation Examples")
    print("="*70)
    print("\nDemonstrating GradientAccumulator usage for efficient training")

    # Run examples
    example_basic_accumulation()
    example_with_metrics_tracker()
    example_checkpointing()
    example_lightning_integration()

    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nFor more information:")
    print("  - User guide: docs/gradient_accumulation_guide.md")
    print("  - Implementation: utils/training/engine/gradient_accumulator.py")
    print("  - Tests: tests/training/engine/test_gradient_accumulator.py")
    print()


if __name__ == '__main__':
    main()
