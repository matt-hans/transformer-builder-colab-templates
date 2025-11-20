"""
Training Loop Example - P1-4

Demonstrates usage of TrainingLoop and ValidationLoop with Phase 0 components.

Example usage:
    python examples/training_loop_example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from utils.training.engine import (
    TrainingLoop,
    ValidationLoop,
    LanguageModelingLoss,
    GradientAccumulator,
    GradientMonitor
)


class SimpleLM(nn.Module):
    """Simple transformer language model."""

    def __init__(self, vocab_size=100, d_model=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.head(x)


def create_synthetic_data(num_samples=1000, seq_len=64, vocab_size=100, batch_size=32):
    """Create synthetic language modeling data."""
    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(data)

    # Split train/val (80/20)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def main():
    """Run training loop example."""
    print("=" * 80)
    print("Training Loop Example - P1-4")
    print("=" * 80)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    print("\nCreating model...")
    vocab_size = 100
    model = SimpleLM(vocab_size=vocab_size, d_model=128, num_layers=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data
    print("\nCreating synthetic data...")
    train_loader, val_loader = create_synthetic_data(
        num_samples=1000,
        seq_len=64,
        vocab_size=vocab_size,
        batch_size=32
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Setup optimizer and scheduler
    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * 5  # 5 epochs
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=total_steps)

    # Setup Phase 0 components
    print("\nSetting up training components...")
    loss_strategy = LanguageModelingLoss()
    gradient_accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=2,  # Effective batch size = 32 * 2 = 64
        max_grad_norm=1.0
    )
    gradient_monitor = GradientMonitor(
        vanishing_threshold=1e-7,
        explosion_threshold=10.0,
        max_consecutive_failures=3
    )

    # Create training and validation loops
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        gradient_monitor=gradient_monitor,
        use_amp=device == 'cuda',  # Enable AMP on GPU
        device=device,
        progress_bar=True
    )

    val_loop = ValidationLoop(
        loss_strategy=loss_strategy,
        device=device,
        progress_bar=True
    )

    print(f"Gradient accumulation steps: {gradient_accumulator.accumulation_steps}")
    print(f"Effective batch size: {gradient_accumulator.effective_batch_size}")
    print(f"Mixed precision: {train_loop.use_amp}")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    num_epochs = 5
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 80)

        # Training
        train_result = train_loop.train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch
        )

        # Validation
        val_result = val_loop.validate_epoch(
            model=model,
            dataloader=val_loader,
            epoch=epoch
        )

        # Print results
        print(f"\n{'Training:':<15} {train_result}")
        print(f"{'Validation:':<15} {val_result}")
        print(f"{'Perplexity:':<15} train={train_result.metrics['train/perplexity']:.2f} | "
              f"val={val_result.metrics['val/perplexity']:.2f}")
        print(f"{'Grad norms:':<15} min={train_result.metrics['train/grad_norm_min']:.4f} | "
              f"max={train_result.metrics['train/grad_norm_max']:.4f}")

        # Track best model
        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            print(f"\n✅ New best validation loss: {best_val_loss:.4f}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final learning rate: {scheduler.get_last_lr()[0]:.2e}")
    print(f"Total optimizer steps: {gradient_accumulator.effective_step}")

    # Serialize results for logging
    print("\nExample result serialization:")
    print("-" * 80)
    result_dict = train_result.to_dict()
    for key, value in list(result_dict.items())[:5]:
        if isinstance(value, float):
            print(f"{key:<25}: {value:.4f}")
        else:
            print(f"{key:<25}: {value}")


if __name__ == '__main__':
    main()
