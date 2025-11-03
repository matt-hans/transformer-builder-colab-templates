"""
Tier 1: Critical Validation Tests

This module contains essential validation tests that verify core model functionality:
- Shape robustness across diverse input dimensions
- Gradient flow through all layers
- Output stability and numerical health
- Parameter initialization quality
- Memory footprint profiling
- Inference speed benchmarking

These tests should pass before proceeding to advanced analysis or training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import time


def test_shape_robustness(model: nn.Module, config: Any) -> Any:
    """
    Validate model across diverse input shapes.

    Tests edge cases like single token, max length, max batch size, etc.
    """
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    vocab_size = getattr(config, 'vocab_size', 50257)
    max_seq_len = getattr(config, 'max_seq_len', 512)
    max_batch_size = getattr(config, 'max_batch_size', 16)

    test_cases = [
        {"batch": 1, "seq_len": 1, "desc": "Single token"},
        {"batch": 1, "seq_len": max_seq_len, "desc": f"Max length ({max_seq_len})"},
        {"batch": max_batch_size, "seq_len": 32, "desc": f"Max batch ({max_batch_size})"},
        {"batch": 4, "seq_len": 64, "desc": "Typical case"},
        {"batch": 2, "seq_len": 128, "desc": "Medium case"},
    ]

    device = next(model.parameters()).device
    results = []

    for case in test_cases:
        try:
            input_ids = torch.randint(0, vocab_size, (case["batch"], case["seq_len"])).to(device)

            with torch.no_grad():
                output = model(input_ids)

            expected_shape = (case["batch"], case["seq_len"], vocab_size)
            actual_shape = tuple(output.shape)

            if actual_shape == expected_shape:
                status = "✅ PASS"
            else:
                status = f"❌ FAIL: Expected {expected_shape}, got {actual_shape}"

            results.append({
                "case": case["desc"],
                "input_shape": f"({case['batch']}, {case['seq_len']})",
                "output_shape": str(actual_shape),
                "status": status
            })
        except Exception as e:
            results.append({
                "case": case["desc"],
                "input_shape": f"({case['batch']}, {case['seq_len']})",
                "output_shape": "N/A",
                "status": f"❌ ERROR: {str(e)[:50]}"
            })

    if pd is not None:
        return pd.DataFrame(results)
    return results


def test_gradient_flow(model: nn.Module, config: Any) -> Any:
    """
    Verify gradients propagate through all layers.

    Detects vanishing gradients, exploding gradients, and unused parameters.
    """
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    vocab_size = getattr(config, 'vocab_size', 50257)
    device = next(model.parameters()).device

    original_mode = model.training  # Preserve original training mode
    model.train()

    # Forward + backward
    input_ids = torch.randint(0, vocab_size, (2, 32)).to(device)
    logits = model(input_ids)

    # Use cross-entropy loss
    labels = torch.randint(0, vocab_size, (2, 32)).to(device)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1)
    )
    loss.backward()

    # Collect gradient statistics
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()

            # Detect issues
            issues = []
            if grad_norm < 1e-7:
                issues.append("⚠️ Near-zero")
            if grad_norm > 1000:
                issues.append("⚠️ Exploding")
            if torch.isnan(param.grad).any():
                issues.append("❌ NaN")

            grad_stats.append({
                "parameter": name,
                "grad_norm": f"{grad_norm:.2e}",
                "grad_mean": f"{grad_mean:.2e}",
                "grad_std": f"{grad_std:.2e}",
                "status": " | ".join(issues) if issues else "✅"
            })
        else:
            grad_stats.append({
                "parameter": name,
                "grad_norm": "N/A",
                "grad_mean": "N/A",
                "grad_std": "N/A",
                "status": "❌ No gradient (unused)"
            })

    # Visualization
    if plt is not None:
        norms = [float(g["grad_norm"]) for g in grad_stats if g["grad_norm"] != "N/A"]
        if norms:
            plt.figure(figsize=(12, 4))
            plt.bar(range(len(norms)), norms)
            plt.yscale('log')
            plt.xlabel('Parameter Index')
            plt.ylabel('Gradient Norm (log scale)')
            plt.title('Gradient Flow Across Layers')
            plt.axhline(y=1e-7, color='r', linestyle='--', label='Near-zero threshold')
            plt.axhline(y=1000, color='orange', linestyle='--', label='Exploding threshold')
            plt.legend()
            plt.tight_layout()
            plt.show()

    model.train(original_mode)  # Restore original training mode

    if pd is not None:
        return pd.DataFrame(grad_stats)
    return grad_stats


def test_output_stability(model: nn.Module, config: Any, n_samples: int = 100) -> Dict[str, Any]:
    """
    Analyze output distribution for numerical issues.

    Tests for NaN, Inf, collapsed outputs, and excessive variance.
    """
    try:
        from scipy.stats import shapiro
    except ImportError:
        print("⚠️ scipy not installed, skipping normality test")
        shapiro = None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    vocab_size = getattr(config, 'vocab_size', 50257)
    device = next(model.parameters()).device

    model.eval()

    outputs = []
    with torch.no_grad():
        for _ in range(n_samples):
            input_ids = torch.randint(0, vocab_size, (1, 32)).to(device)
            logits = model(input_ids)
            outputs.append(logits.cpu())

    outputs = torch.cat(outputs, dim=0)

    # Statistical analysis
    stats = {
        "mean": outputs.mean().item(),
        "std": outputs.std().item(),
        "min": outputs.min().item(),
        "max": outputs.max().item(),
        "has_nan": torch.isnan(outputs).any().item(),
        "has_inf": torch.isinf(outputs).any().item(),
        "dynamic_range": (outputs.max() - outputs.min()).item(),
    }

    # Check for issues
    issues = []
    if stats["has_nan"]:
        issues.append("❌ NaN values detected")
    if stats["has_inf"]:
        issues.append("❌ Inf values detected")
    if stats["std"] < 0.01:
        issues.append("⚠️ Very low variance (collapsed outputs)")
    if stats["std"] > 100:
        issues.append("⚠️ Very high variance (unstable)")
    if abs(stats["mean"]) > 10:
        issues.append("⚠️ Large mean bias")

    # Normality test (sample 1000 values)
    if shapiro is not None:
        sample = outputs.flatten()[:min(1000, outputs.numel())].numpy()
        if len(sample) >= 20:
            _, p_value = shapiro(sample)
        else:
            p_value = None
            print("⚠️ Insufficient samples for normality test")
    else:
        p_value = None

    print("=" * 60)
    print("OUTPUT STABILITY ANALYSIS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    if p_value is not None:
        print(f"{'Shapiro-Wilk p':20s}: {p_value:.4f}")
    print(f"\nStatus: {issues[0] if issues else '✅ PASS'}")
    if len(issues) > 1:
        for issue in issues[1:]:
            print(f"        {issue}")
    print("=" * 60)

    # Histogram
    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.hist(outputs.flatten().numpy(), bins=50, density=True, alpha=0.7, edgecolor='black')
        plt.xlabel('Logit Value')
        plt.ylabel('Density')
        plt.title(f'Output Distribution (n={n_samples} samples)')
        plt.axvline(stats["mean"], color='r', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return stats


def test_parameter_initialization(model: nn.Module) -> Any:
    """
    Verify parameter initialization is reasonable.

    Checks for common issues like all-zeros, excessive variance, or high mean bias.
    """
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    param_stats = []

    for name, param in model.named_parameters():
        stats = {
            "parameter": name,
            "shape": str(tuple(param.shape)),
            "mean": f"{param.mean().item():.4f}",
            "std": f"{param.std().item():.4f}",
            "min": f"{param.min().item():.4f}",
            "max": f"{param.max().item():.4f}",
        }

        # Heuristic checks
        issues = []
        mean_val = abs(param.mean().item())
        std_val = param.std().item()

        if mean_val > 0.1:
            issues.append("⚠️ High mean bias")
        if std_val < 0.001:
            issues.append("⚠️ Very small std")
        if std_val > 2.0:
            issues.append("⚠️ Very large std")
        if (param == 0).all():
            issues.append("❌ All zeros")

        stats["status"] = " | ".join(issues) if issues else "✅"
        param_stats.append(stats)

    # Plot distribution of stds
    if plt is not None:
        stds = [float(s["std"]) for s in param_stats]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(stds)), stds, edgecolor='black')
        plt.xlabel('Parameter Index')
        plt.ylabel('Standard Deviation')
        plt.title('Parameter Initialization Spread')
        plt.axhline(y=0.02, color='g', linestyle='--', linewidth=2, label='Typical lower bound')
        plt.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Typical upper bound')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    if pd is not None:
        return pd.DataFrame(param_stats)
    return param_stats


def test_memory_footprint(model: nn.Module, config: Any) -> Any:
    """
    Measure memory usage across batch sizes.

    Helps identify memory scaling issues and OOM thresholds.
    """
    import gc

    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    vocab_size = getattr(config, 'vocab_size', 50257)
    device = next(model.parameters()).device

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    results = []
    batch_sizes = [1, 2, 4, 8, 16]

    for batch_size in batch_sizes:
        try:
            # Measure before
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                try:
                    import psutil
                    process = psutil.Process()
                    mem_before = process.memory_info().rss / 1024**2
                except ImportError:
                    print("⚠️ psutil not installed, skipping CPU memory tracking")
                    mem_before = 0

            # Forward pass
            input_ids = torch.randint(0, vocab_size, (batch_size, 64)).to(device)

            with torch.no_grad():
                output = model(input_ids)

            # Measure after
            if device.type == 'cuda':
                torch.cuda.synchronize()
                mem_after = torch.cuda.max_memory_allocated() / 1024**2
            else:
                try:
                    mem_after = process.memory_info().rss / 1024**2
                except:
                    mem_after = 0

            mem_used = mem_after - mem_before

            results.append({
                "batch_size": batch_size,
                "memory_mb": f"{mem_used:.2f}",
                "per_sample_mb": f"{mem_used/batch_size:.2f}",
                "status": "✅"
            })

            # Clean up
            del input_ids, output
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results.append({
                    "batch_size": batch_size,
                    "memory_mb": "OOM",
                    "per_sample_mb": "N/A",
                    "status": "❌ Out of Memory"
                })
                break
            else:
                raise

    # Plot memory scaling
    if plt is not None and len(results) > 1:
        valid_results = [r for r in results if r["status"] == "✅"]
        if valid_results:
            batch_sizes_ok = [r["batch_size"] for r in valid_results]
            mem_values = [float(r["memory_mb"]) for r in valid_results]

            plt.figure(figsize=(8, 5))
            plt.plot(batch_sizes_ok, mem_values, marker='o', linewidth=2, markersize=8)
            plt.xlabel('Batch Size')
            plt.ylabel('Memory (MB)')
            plt.title('Memory Scaling by Batch Size')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    if pd is not None:
        return pd.DataFrame(results)
    return results


def test_inference_speed(model: nn.Module, config: Any, n_trials: int = 50) -> Dict[str, float]:
    """
    Benchmark inference latency and throughput.

    Measures P50, P95, P99 latencies and samples/second throughput.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    vocab_size = getattr(config, 'vocab_size', 50257)
    device = next(model.parameters()).device

    model.eval()

    # Warmup
    for _ in range(5):
        input_ids = torch.randint(0, vocab_size, (1, 64)).to(device)
        with torch.no_grad():
            _ = model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(n_trials):
        input_ids = torch.randint(0, vocab_size, (1, 64)).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            output = model(input_ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)

    stats = {
        "mean_ms": latencies.mean(),
        "std_ms": latencies.std(),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "throughput_samples_per_sec": 1000 / latencies.mean(),
    }

    print("=" * 60)
    print("INFERENCE SPEED BENCHMARK")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Trials: {n_trials}")
    print(f"Input shape: (1, 64)")
    print("-" * 60)
    for key, value in stats.items():
        print(f"{key:30s}: {value:.2f}")
    print("=" * 60)

    # Latency distribution
    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.hist(latencies, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(stats["mean_ms"], color='r', linestyle='--', linewidth=2, label=f'Mean: {stats["mean_ms"]:.2f}ms')
        plt.axvline(stats["p95_ms"], color='orange', linestyle='--', linewidth=2, label=f'P95: {stats["p95_ms"]:.2f}ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Latency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    return stats
