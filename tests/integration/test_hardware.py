"""
Integration tests for hardware-specific features.

Tests GPU training, mixed precision, torch.compile, Flash Attention, and distributed training.
"""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import time

from utils.training.engine.trainer import Trainer
from utils.training.training_config import TrainingConfig
from utils.training.task_spec import TaskSpec
from utils.adapters.model_adapter import UniversalModelAdapter
from utils.training.flash_attention_wrapper import FlashAttentionWrapper


# ============================================================================
# Test 1: GPU Training (if CUDA available)
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_training(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    cuda_device
):
    """Test training on GPU with CUDA."""
    # Arrange
    model = tiny_transformer_model.to(cuda_device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Act - Train on GPU
    trainer = Trainer(
        model=adapter,
        config=basic_training_config,
        task_spec=lm_task_spec
    )

    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    gpu_time = time.time() - start_time

    # Assert
    assert results is not None
    assert results['final_loss'] > 0
    # Verify model is on GPU
    assert next(model.parameters()).device.type == 'cuda'
    print(f"GPU training time: {gpu_time:.2f}s")


# ============================================================================
# Test 2: Mixed Precision (AMP) Training
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
def test_mixed_precision_training(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    cuda_device
):
    """Test automatic mixed precision (AMP) training."""
    # Arrange
    model = tiny_transformer_model.to(cuda_device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Enable AMP
    config_with_amp = TrainingConfig(
        **basic_training_config.to_dict(),
        use_amp=True,
        amp_dtype='float16'  # or 'bfloat16' if supported
    )

    # Act
    trainer = Trainer(
        model=adapter,
        config=config_with_amp,
        task_spec=lm_task_spec
    )

    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    amp_time = time.time() - start_time

    # Assert
    assert results is not None
    assert results['final_loss'] > 0
    print(f"AMP training time: {amp_time:.2f}s")

    # Verify AMP was used (check for GradScaler in trainer)
    # Note: This is implementation-specific
    if hasattr(trainer, 'scaler'):
        assert trainer.scaler is not None


# ============================================================================
# Test 3: torch.compile Integration
# ============================================================================

@pytest.mark.integration
@pytest.mark.skipif(
    not hasattr(torch, 'compile'),
    reason="torch.compile requires PyTorch 2.0+"
)
def test_torch_compile_integration(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    device
):
    """Test torch.compile for accelerated training."""
    # Arrange
    model = tiny_transformer_model.to(device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Enable torch.compile
    config_with_compile = TrainingConfig(
        **basic_training_config.to_dict(),
        compile_mode="default",  # or "reduce-overhead", "max-autotune"
        compile_fullgraph=False,
        compile_dynamic=True
    )

    # Act
    trainer = Trainer(
        model=adapter,
        config=config_with_compile,
        task_spec=lm_task_spec
    )

    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    compile_time = time.time() - start_time

    # Assert
    assert results is not None
    assert results['final_loss'] > 0
    print(f"Compiled training time: {compile_time:.2f}s")


# ============================================================================
# Test 4: Flash Attention Integration
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(
    not hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
    reason="Flash Attention (SDPA) requires PyTorch 2.0+"
)
def test_flash_attention_integration(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    cuda_device
):
    """Test Flash Attention (SDPA) integration."""
    # Arrange
    model = tiny_transformer_model.to(cuda_device)

    # Wrap model with Flash Attention
    flash_wrapper = FlashAttentionWrapper()
    num_patched = flash_wrapper.wrap_model(model)

    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Act
    trainer = Trainer(
        model=adapter,
        config=basic_training_config,
        task_spec=lm_task_spec
    )

    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    flash_time = time.time() - start_time

    # Assert
    assert results is not None
    assert results['final_loss'] > 0
    assert num_patched > 0, "At least one attention layer should be patched"
    print(f"Flash Attention training time: {flash_time:.2f}s")
    print(f"Flash Attention patched {num_patched} layers")


# ============================================================================
# Test 5: Multi-GPU Training (DDP)
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Multi-GPU test requires 2+ GPUs"
)
def test_distributed_data_parallel_training(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset
):
    """Test distributed data parallel (DDP) training on multiple GPUs."""
    # Note: This test is challenging in pytest context
    # Real DDP requires torch.distributed.launch or multiprocessing
    # This test verifies configuration compatibility

    config_with_ddp = TrainingConfig(
        **basic_training_config.to_dict(),
        strategy='ddp',
        devices=2
    )

    # Verify config is valid
    assert config_with_ddp.strategy == 'ddp'
    assert config_with_ddp.devices == 2

    # For actual DDP testing, use CLI: python cli/run_training.py --strategy ddp
    # Integration test verifies configuration correctness
    pytest.skip("DDP requires multiprocessing context, test via CLI")


# ============================================================================
# Test 6: CPU vs GPU Performance Comparison
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
def test_cpu_vs_gpu_performance(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset
):
    """Compare CPU vs GPU training performance."""
    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # CPU training
    model_cpu = tiny_transformer_model.to('cpu')
    adapter_cpu = UniversalModelAdapter(model_cpu, tiny_config, lm_task_spec)
    trainer_cpu = Trainer(model=adapter_cpu, config=basic_training_config, task_spec=lm_task_spec)

    start_cpu = time.time()
    results_cpu = trainer_cpu.train(train_loader=train_loader, val_loader=val_loader)
    cpu_time = time.time() - start_cpu

    print(f"CPU training time: {cpu_time:.2f}s")

    # GPU training (if available)
    if torch.cuda.is_available():
        model_gpu = tiny_transformer_model.to('cuda')
        adapter_gpu = UniversalModelAdapter(model_gpu, tiny_config, lm_task_spec)
        trainer_gpu = Trainer(model=adapter_gpu, config=basic_training_config, task_spec=lm_task_spec)

        start_gpu = time.time()
        results_gpu = trainer_gpu.train(train_loader=train_loader, val_loader=val_loader)
        gpu_time = time.time() - start_gpu

        print(f"GPU training time: {gpu_time:.2f}s")
        print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")

        assert results_gpu is not None
        # GPU should be faster (but not always guaranteed for tiny models)
    else:
        pytest.skip("CUDA not available for GPU comparison")


# ============================================================================
# Test 7: Memory Profiling (GPU)
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_memory_profiling(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    cuda_device
):
    """Test GPU memory usage during training."""
    # Arrange
    torch.cuda.reset_peak_memory_stats(cuda_device)
    torch.cuda.empty_cache()

    model = tiny_transformer_model.to(cuda_device)
    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Act
    trainer = Trainer(model=adapter, config=basic_training_config, task_spec=lm_task_spec)

    initial_memory = torch.cuda.memory_allocated(cuda_device) / 1024**2  # MB

    results = trainer.train(train_loader=train_loader, val_loader=val_loader)

    peak_memory = torch.cuda.max_memory_allocated(cuda_device) / 1024**2  # MB
    final_memory = torch.cuda.memory_allocated(cuda_device) / 1024**2  # MB

    # Assert
    assert results is not None
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    print(f"Peak GPU memory: {peak_memory:.2f} MB")
    print(f"Final GPU memory: {final_memory:.2f} MB")
    print(f"Memory increase: {peak_memory - initial_memory:.2f} MB")

    # Verify reasonable memory usage (tiny model should use < 500MB)
    assert peak_memory < 500, "Tiny model should not exceed 500MB GPU memory"


# ============================================================================
# Test 8: Gradient Checkpointing (Memory Optimization)
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
def test_gradient_checkpointing(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    cuda_device
):
    """Test gradient checkpointing for memory efficiency."""
    # Arrange
    model = tiny_transformer_model.to(cuda_device)

    # Enable gradient checkpointing (if model supports)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=basic_training_config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=basic_training_config.batch_size
    )

    # Act
    torch.cuda.reset_peak_memory_stats(cuda_device)
    trainer = Trainer(model=adapter, config=basic_training_config, task_spec=lm_task_spec)
    results = trainer.train(train_loader=train_loader, val_loader=val_loader)
    peak_memory_with_checkpointing = torch.cuda.max_memory_allocated(cuda_device) / 1024**2

    # Assert
    assert results is not None
    print(f"Peak memory with checkpointing: {peak_memory_with_checkpointing:.2f} MB")


# ============================================================================
# Test 9: Inference Mode (torch.inference_mode)
# ============================================================================

@pytest.mark.integration
def test_inference_mode_optimization(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    synthetic_text_dataset,
    device
):
    """Test inference optimization with torch.inference_mode."""
    # Arrange
    model = tiny_transformer_model.to(device)
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        synthetic_text_dataset, batch_size=8, shuffle=False
    )

    # Act - Inference without optimization
    start_normal = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            _ = model(input_ids)
            break  # Single batch for speed
    normal_time = time.time() - start_normal

    # Act - Inference with torch.inference_mode
    start_inference = time.time()
    with torch.inference_mode():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            _ = model(input_ids)
            break
    inference_time = time.time() - start_inference

    print(f"Normal inference: {normal_time*1000:.2f}ms")
    print(f"Inference mode: {inference_time*1000:.2f}ms")
    print(f"Speedup: {normal_time / inference_time:.2f}x")


# ============================================================================
# Test 10: Batch Size Scaling (Find Optimal Batch Size)
# ============================================================================

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
def test_batch_size_scaling(
    tiny_transformer_model,
    tiny_config,
    lm_task_spec,
    basic_training_config,
    synthetic_text_dataset,
    cuda_device
):
    """Test training with different batch sizes to find optimal."""
    train_size = int(0.8 * len(synthetic_text_dataset))
    val_size = len(synthetic_text_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        synthetic_text_dataset, [train_size, val_size]
    )

    batch_sizes = [2, 4, 8, 16]
    results_by_batch_size = {}

    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()

            model = tiny_transformer_model.to(cuda_device)
            adapter = UniversalModelAdapter(model, tiny_config, lm_task_spec)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )

            config = TrainingConfig(**basic_training_config.to_dict(), batch_size=batch_size)
            trainer = Trainer(model=adapter, config=config, task_spec=lm_task_spec)

            start_time = time.time()
            results = trainer.train(train_loader=train_loader, val_loader=val_loader)
            train_time = time.time() - start_time

            peak_memory = torch.cuda.max_memory_allocated(cuda_device) / 1024**2

            results_by_batch_size[batch_size] = {
                'time': train_time,
                'memory': peak_memory,
                'final_loss': results['final_loss']
            }

            print(f"Batch size {batch_size}: {train_time:.2f}s, {peak_memory:.2f}MB, loss={results['final_loss']:.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OOM")
                break
            else:
                raise

    # Verify we tested at least 2 batch sizes
    assert len(results_by_batch_size) >= 2
