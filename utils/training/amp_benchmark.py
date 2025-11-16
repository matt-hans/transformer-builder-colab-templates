"""
AMP (Automatic Mixed Precision) Benchmarking Utilities.

Provides functions to benchmark AMP speedup by comparing FP32 vs FP16 training time,
memory usage, and accuracy metrics.
"""

import copy
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


def test_amp_speedup_benchmark(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    use_wandb: bool = False
) -> Dict[str, Any]:
    """
    Benchmark AMP speedup by comparing FP32 vs FP16 training time.

    Runs identical training twice (FP32 and FP16) and measures:
    - Training time
    - Throughput (samples/sec)
    - Memory usage
    - Final validation loss and accuracy
    - Speedup ratio

    Args:
        model: The transformer model to benchmark
        config: Model configuration
        train_data: List of input_ids tensors (if None, generates synthetic data)
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size for training
        use_wandb: Whether to log metrics to W&B

    Returns:
        Dictionary with benchmark results comparing FP32 vs FP16
    """
    # Import here to avoid circular dependency
    from utils.tier3_training_utilities import test_fine_tuning

    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, AMP benchmark requires GPU")
        return {
            "error": "CUDA not available",
            "fp32_results": None,
            "fp16_results": None,
            "speedup": None
        }

    print("=" * 60)
    print("AMP SPEEDUP BENCHMARK")
    print("=" * 60)
    print("Running identical training twice:")
    print("  1. FP32 baseline (standard precision)")
    print("  2. FP16 with AMP (mixed precision)")
    print("-" * 60)

    # Store initial model state
    initial_state = copy.deepcopy(model.state_dict())

    # Measure GPU memory before training
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Run FP32 baseline
    print("\n[1/2] Running FP32 baseline...")
    model.load_state_dict(initial_state)  # Reset model
    fp32_results = test_fine_tuning(
        model=model,
        config=config,
        train_data=train_data,
        val_data=None,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        use_amp=False
    )
    fp32_time = fp32_results['training_time_seconds']
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    fp32_final_val_loss = fp32_results['metrics_summary']['val/loss'].iloc[-1]
    fp32_final_val_acc = fp32_results['metrics_summary']['val/accuracy'].iloc[-1]

    # Reset for FP16
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Run FP16 with AMP
    print("\n[2/2] Running FP16 with AMP...")
    model.load_state_dict(initial_state)  # Reset model
    fp16_results = test_fine_tuning(
        model=model,
        config=config,
        train_data=train_data,
        val_data=None,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        use_amp=True
    )
    fp16_time = fp16_results['training_time_seconds']
    fp16_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    fp16_final_val_loss = fp16_results['metrics_summary']['val/loss'].iloc[-1]
    fp16_final_val_acc = fp16_results['metrics_summary']['val/accuracy'].iloc[-1]

    # Calculate metrics
    speedup = fp32_time / fp16_time
    memory_reduction = ((fp32_memory - fp16_memory) / fp32_memory) * 100
    accuracy_diff = abs(fp32_final_val_acc - fp16_final_val_acc)
    loss_diff = abs(fp32_final_val_loss - fp16_final_val_loss)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"FP32 Baseline:")
    print(f"  Training time: {fp32_time:.2f}s")
    print(f"  GPU memory: {fp32_memory:.1f} MB")
    print(f"  Final val loss: {fp32_final_val_loss:.4f}")
    print(f"  Final val acc: {fp32_final_val_acc:.4f}")
    print("-" * 60)
    print(f"FP16 with AMP:")
    print(f"  Training time: {fp16_time:.2f}s")
    print(f"  GPU memory: {fp16_memory:.1f} MB")
    print(f"  Final val loss: {fp16_final_val_loss:.4f}")
    print(f"  Final val acc: {fp16_final_val_acc:.4f}")
    print("-" * 60)
    print(f"Performance Gains:")
    print(f"  ⚡ Speedup: {speedup:.2f}x")
    print(f"  💾 Memory reduction: {memory_reduction:.1f}%")
    print(f"  📊 Accuracy difference: {accuracy_diff:.4f}")
    print(f"  📉 Loss difference: {loss_diff:.4f}")
    print("=" * 60)

    # Verify requirements (updated threshold to 40%)
    print("\nRequirement Verification:")
    if speedup >= 1.5:
        print(f"  ✅ Speedup target met: {speedup:.2f}x >= 1.5x")
    else:
        print(f"  ⚠️ Speedup below target: {speedup:.2f}x < 1.5x")

    if memory_reduction >= 40:
        print(f"  ✅ Memory reduction target met: {memory_reduction:.1f}% >= 40%")
    else:
        print(f"  ⚠️ Memory reduction below target: {memory_reduction:.1f}% < 40%")

    if accuracy_diff < 0.01:
        print(f"  ✅ No accuracy degradation: {accuracy_diff:.4f} < 0.01")
    else:
        print(f"  ⚠️ Accuracy difference: {accuracy_diff:.4f} >= 0.01")

    # Log to W&B if enabled
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'amp_benchmark/speedup': speedup,
                    'amp_benchmark/memory_reduction_percent': memory_reduction,
                    'amp_benchmark/accuracy_diff': accuracy_diff,
                    'amp_benchmark/loss_diff': loss_diff,
                    'amp_benchmark/fp32_time': fp32_time,
                    'amp_benchmark/fp16_time': fp16_time,
                    'amp_benchmark/fp32_memory_mb': fp32_memory,
                    'amp_benchmark/fp16_memory_mb': fp16_memory
                })
                print("  📊 Benchmark metrics logged to W&B")
        except Exception as e:
            import logging
            logging.warning(f"Failed to log benchmark to W&B: {e}")
            print(f"  ⚠️ Failed to log benchmark to W&B: {e}")

    return {
        "fp32_results": fp32_results,
        "fp16_results": fp16_results,
        "speedup": speedup,
        "memory_reduction_percent": memory_reduction,
        "accuracy_difference": accuracy_diff,
        "loss_difference": loss_diff,
        "fp32_time_seconds": fp32_time,
        "fp16_time_seconds": fp16_time,
        "fp32_memory_mb": fp32_memory,
        "fp16_memory_mb": fp16_memory,
        "requirements_met": {
            "speedup_1.5x": speedup >= 1.5,
            "memory_reduction_40pct": memory_reduction >= 40,  # Updated from 30%
            "accuracy_stable": accuracy_diff < 0.01
        }
    }
