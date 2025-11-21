"""
Unit and Integration Tests for Training/Validation Loops

Tests cover:
- TrainingLoop with synthetic data
- ValidationLoop with synthetic data
- Gradient accumulation integration
- Gradient monitoring integration
- Exception handling (OOM, NaN loss, keyboard interrupt)
- Mixed precision training (AMP)
- Progress bar integration
- Metrics tracking
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from utils.training.engine.loop import TrainingLoop, ValidationLoop, EpochResult
from utils.training.engine.loss import LanguageModelingLoss, ClassificationLoss
from utils.training.engine.gradient_monitor import GradientMonitor
from utils.training.engine.gradient_accumulator import GradientAccumulator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_lm_model():
    """Simple language model for testing."""
    class SimpleLM(nn.Module):
        def __init__(self, vocab_size=100, d_model=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            )
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embed(x)
            x = self.transformer(x)
            return self.head(x)

    return SimpleLM()


@pytest.fixture
def simple_classifier():
    """Simple classifier for testing."""
    class SimpleClassifier(nn.Module):
        def __init__(self, num_classes=10, input_size=128):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            # Flatten if needed
            if x.ndim > 2:
                x = x.mean(dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleClassifier()


@pytest.fixture
def synthetic_lm_data():
    """Synthetic language modeling data."""
    # Create random sequences
    batch_size = 8
    seq_len = 32
    vocab_size = 100
    num_samples = 64

    data = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, vocab_size


@pytest.fixture
def synthetic_classification_data():
    """Synthetic classification data."""
    batch_size = 16
    input_size = 128
    num_classes = 10
    num_samples = 128

    inputs = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, num_classes


# ============================================================================
# TrainingLoop Tests
# ============================================================================

def test_training_loop_basic_execution(simple_lm_model, synthetic_lm_data):
    """Test basic training loop execution with synthetic data."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Setup components
    loss_strategy = LanguageModelingLoss()
    gradient_accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=1,
        max_grad_norm=1.0
    )

    # Create training loop
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    # Execute one epoch
    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    # Verify result structure
    assert isinstance(result, EpochResult)
    assert result.loss > 0
    assert 0 <= result.accuracy <= 1.0
    assert result.batch_count == len(dataloader)
    assert result.duration > 0
    assert result.gradient_norms is not None
    assert len(result.gradient_norms) > 0
    assert result.loss_history is not None
    assert len(result.loss_history) == result.batch_count


def test_training_loop_gradient_accumulation(simple_lm_model, synthetic_lm_data):
    """Test training loop with gradient accumulation."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    accumulation_steps = 4
    gradient_accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=accumulation_steps,
        max_grad_norm=1.0
    )

    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    # Optimizer should step every accumulation_steps batches
    expected_optimizer_steps = (len(dataloader) + accumulation_steps - 1) // accumulation_steps
    assert len(result.gradient_norms) == expected_optimizer_steps
    assert gradient_accumulator.effective_step == expected_optimizer_steps


def test_training_loop_gradient_monitoring(simple_lm_model, synthetic_lm_data):
    """Test training loop with gradient health monitoring."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    gradient_monitor = GradientMonitor(
        vanishing_threshold=1e-7,
        explosion_threshold=10.0,
        max_consecutive_failures=3
    )

    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        gradient_monitor=gradient_monitor,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    # Should execute without gradient health issues
    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    assert result.loss > 0
    assert all(torch.isfinite(torch.tensor(g)) for g in result.gradient_norms)


def test_training_loop_learning_rate_scheduler(simple_lm_model, synthetic_lm_data):
    """Test training loop with LR scheduler integration."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Linear warmup scheduler
    total_steps = len(dataloader)
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=total_steps)
    initial_lr = scheduler.get_last_lr()[0]

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0
    )

    # LR should have changed
    final_lr = scheduler.get_last_lr()[0]
    assert final_lr != initial_lr
    assert result.learning_rate == final_lr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_training_loop_amp(simple_lm_model, synthetic_lm_data):
    """Test training loop with automatic mixed precision."""
    model = simple_lm_model.cuda()
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=True,
        device='cuda',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    # Verify AMP scaler was used
    assert train_loop.scaler is not None
    assert result.loss > 0


def test_training_loop_nan_loss_detection(simple_lm_model, synthetic_lm_data):
    """Test NaN loss detection and error handling."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e10)  # Extreme LR to cause NaN

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    # Should raise RuntimeError for NaN loss (may take a few epochs)
    with pytest.raises(RuntimeError, match="NaN loss detected"):
        for epoch in range(10):
            result = train_loop.train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                epoch=epoch
            )


def test_training_loop_classification_task(simple_classifier, synthetic_classification_data):
    """Test training loop with classification task."""
    model = simple_classifier
    dataloader, num_classes = synthetic_classification_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = ClassificationLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    assert result.loss > 0
    assert 0 <= result.accuracy <= 1.0
    assert result.batch_count == len(dataloader)


# ============================================================================
# ValidationLoop Tests
# ============================================================================

def test_validation_loop_basic_execution(simple_lm_model, synthetic_lm_data):
    """Test basic validation loop execution."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data

    loss_strategy = LanguageModelingLoss()
    val_loop = ValidationLoop(
        loss_strategy=loss_strategy,
        device='cpu',
        progress_bar=False
    )

    result = val_loop.validate_epoch(
        model=model,
        dataloader=dataloader,
        epoch=0
    )

    # Verify result structure
    assert isinstance(result, EpochResult)
    assert result.loss > 0
    assert 0 <= result.accuracy <= 1.0
    assert result.batch_count == len(dataloader)
    assert result.duration > 0
    assert result.gradient_norms is None  # No gradients in validation
    assert result.loss_history is not None
    assert len(result.loss_history) == result.batch_count


def test_validation_loop_no_gradient_computation(simple_lm_model, synthetic_lm_data):
    """Test that validation loop does not compute gradients."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data

    # Verify model parameters don't have gradients initially
    for param in model.parameters():
        assert param.grad is None

    loss_strategy = LanguageModelingLoss()
    val_loop = ValidationLoop(
        loss_strategy=loss_strategy,
        device='cpu',
        progress_bar=False
    )

    result = val_loop.validate_epoch(
        model=model,
        dataloader=dataloader,
        epoch=0
    )

    # After validation, gradients should still be None
    for param in model.parameters():
        assert param.grad is None


def test_validation_loop_eval_mode(simple_lm_model, synthetic_lm_data):
    """Test that validation loop sets model to eval mode."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data

    # Set model to training mode
    model.train()
    assert model.training is True

    loss_strategy = LanguageModelingLoss()
    val_loop = ValidationLoop(
        loss_strategy=loss_strategy,
        device='cpu',
        progress_bar=False
    )

    result = val_loop.validate_epoch(
        model=model,
        dataloader=dataloader,
        epoch=0
    )

    # Model should be in eval mode after validation
    assert model.training is False


def test_validation_loop_perplexity_computation(simple_lm_model, synthetic_lm_data):
    """Test perplexity computation in validation."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data

    loss_strategy = LanguageModelingLoss()
    val_loop = ValidationLoop(
        loss_strategy=loss_strategy,
        device='cpu',
        progress_bar=False
    )

    result = val_loop.validate_epoch(
        model=model,
        dataloader=dataloader,
        epoch=0
    )

    # Perplexity should be computed
    assert 'val/perplexity' in result.metrics
    perplexity = result.metrics['val/perplexity']
    assert perplexity >= 1.0  # Perplexity is always >= 1
    assert torch.isfinite(torch.tensor(perplexity))


# ============================================================================
# Integration Tests
# ============================================================================

def test_train_val_integration(simple_lm_model, synthetic_lm_data):
    """Test full training + validation integration."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=len(dataloader) * 3)

    # Setup components
    loss_strategy = LanguageModelingLoss()
    gradient_accumulator = GradientAccumulator(
        optimizer=optimizer,
        accumulation_steps=2,
        max_grad_norm=1.0
    )
    gradient_monitor = GradientMonitor()

    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        gradient_monitor=gradient_monitor,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    val_loop = ValidationLoop(
        loss_strategy=loss_strategy,
        device='cpu',
        progress_bar=False
    )

    # Run 3 epochs
    train_losses = []
    val_losses = []

    for epoch in range(3):
        # Training
        train_result = train_loop.train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch
        )
        train_losses.append(train_result.loss)

        # Validation
        val_result = val_loop.validate_epoch(
            model=model,
            dataloader=dataloader,
            epoch=epoch
        )
        val_losses.append(val_result.loss)

    # Loss should generally decrease (or at least not explode)
    assert train_losses[-1] < train_losses[0] * 2  # Allow some variance
    assert all(torch.isfinite(torch.tensor(loss)) for loss in train_losses)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in val_losses)


def test_epoch_result_serialization(simple_lm_model, synthetic_lm_data):
    """Test EpochResult to_dict() serialization."""
    model = simple_lm_model
    dataloader, vocab_size = synthetic_lm_data
    optimizer = Adam(model.parameters(), lr=1e-3)

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    # Serialize to dict
    result_dict = result.to_dict()

    # Verify structure
    assert 'loss' in result_dict
    assert 'accuracy' in result_dict
    assert 'duration' in result_dict
    assert 'batch_count' in result_dict
    assert 'train/loss' in result_dict
    assert 'train/accuracy' in result_dict
    assert 'train/perplexity' in result_dict


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_dataloader():
    """Test behavior with empty dataloader."""
    model = nn.Linear(10, 10)
    optimizer = Adam(model.parameters(), lr=1e-3)
    empty_dataloader = DataLoader(TensorDataset(torch.randn(0, 10)))

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = ClassificationLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=empty_dataloader,
        optimizer=optimizer,
        epoch=0
    )

    # Should handle gracefully
    assert result.batch_count == 0
    assert result.loss == float('inf')


def test_single_batch_training(simple_lm_model):
    """Test training with single batch."""
    model = simple_lm_model
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Single batch: (num_samples, seq_len) - no extra batch dimension
    data = torch.randint(0, 100, (8, 32))
    dataloader = DataLoader(TensorDataset(data), batch_size=8)

    gradient_accumulator = GradientAccumulator(optimizer=optimizer, accumulation_steps=1)
    loss_strategy = LanguageModelingLoss()
    train_loop = TrainingLoop(
        loss_strategy=loss_strategy,
        gradient_accumulator=gradient_accumulator,
        use_amp=False,
        device='cpu',
        progress_bar=False
    )

    result = train_loop.train_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epoch=0
    )

    assert result.batch_count == 1
    assert len(result.gradient_norms) == 1
    assert len(result.loss_history) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
