"""
Benchmark Utilities for Model Comparison.

Provides helper functions for benchmarking model inference speed,
computing perplexity, loading baseline models, and creating visualizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import time
import numpy as np


def load_baseline_model(baseline_model_name: str, device: torch.device) -> Optional[nn.Module]:
    """Load HuggingFace baseline model with error handling."""
    try:
        from transformers import AutoModelForCausalLM
        baseline = AutoModelForCausalLM.from_pretrained(baseline_model_name).to(device)
        baseline.eval()
        return baseline
    except Exception as e:
        print(f"❌ Failed to load baseline: {str(e)}")
        return None


def benchmark_inference_speed(
    model: nn.Module,
    test_data: List[torch.Tensor],
    device: torch.device,
    warmup_runs: int = 5
) -> List[float]:
    """
    Benchmark model inference speed with warmup using CUDA events for efficient timing.

    Returns:
        List of inference times in seconds
    """
    model.eval()

    # Warmup (synchronize only once at end)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(test_data[0].unsqueeze(0))
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Single sync after warmup

    # Benchmark using CUDA events for accurate timing without excessive syncs
    times = []
    if device.type == 'cuda':
        # Use CUDA events for GPU timing
        for sample in test_data:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            with torch.no_grad():
                _ = model(sample.unsqueeze(0))
            end_event.record()

            # Store events for later synchronization
            times.append((start_event, end_event))

        # Single synchronization at the end
        torch.cuda.synchronize()

        # Convert events to elapsed times
        times = [start.elapsed_time(end) / 1000.0 for start, end in times]  # Convert ms to seconds
    else:
        # CPU timing - use time.perf_counter()
        for sample in test_data:
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(sample.unsqueeze(0))
            times.append(time.perf_counter() - start)

    return times


def compute_model_perplexity(
    model: nn.Module,
    test_data: List[torch.Tensor],
    vocab_size: int,
    is_baseline: bool = False,
    safe_get_model_output=None
) -> float:
    """
    Compute perplexity for a model on test data.

    Args:
        model: Model to evaluate
        test_data: List of test samples
        vocab_size: Vocabulary size
        is_baseline: Whether this is a HuggingFace baseline model
        safe_get_model_output: Function to safely extract model output

    Returns:
        Perplexity value
    """
    losses = []

    for sample in test_data:
        input_ids = sample.unsqueeze(0)

        with torch.no_grad():
            if is_baseline:
                logits = model(input_ids).logits
            else:
                if safe_get_model_output is None:
                    raise ValueError("safe_get_model_output function must be provided for non-baseline models")
                logits = safe_get_model_output(model, input_ids)

            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            losses.append(loss.item())

    return np.exp(np.mean(losses))


def create_benchmark_visualization(
    custom_params: int,
    baseline_params: int,
    custom_speed_ms: float,
    baseline_speed_ms: float,
    custom_ppl: float,
    baseline_ppl: float
):
    """Create benchmark comparison visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Parameter comparison
    axes[0].bar(['Custom', 'Baseline'],
               [custom_params, baseline_params],
               edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Parameter Count')
    axes[0].set_title('Model Size')
    axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Speed comparison
    axes[1].bar(['Custom', 'Baseline'],
               [custom_speed_ms, baseline_speed_ms],
               edgecolor='black', linewidth=2, color=['green', 'blue'])
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Inference Speed')

    # Perplexity comparison
    axes[2].bar(['Custom', 'Baseline'],
               [custom_ppl, baseline_ppl],
               edgecolor='black', linewidth=2, color=['orange', 'red'])
    axes[2].set_ylabel('Perplexity (lower is better)')
    axes[2].set_title('Language Modeling Quality')

    plt.tight_layout()
    plt.show()
