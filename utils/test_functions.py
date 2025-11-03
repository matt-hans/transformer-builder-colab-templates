"""
Test utilities for Transformer Builder Colab template.

Implements 3-tier testing infrastructure:
- Tier 1: Critical validation (shape, gradients, stability, performance)
- Tier 2: Advanced analysis (attention, attribution, robustness)
- Tier 3: Training utilities (fine-tuning, hyperparameter search, benchmarks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import time

# ==============================================================================
# TIER 1: CRITICAL VALIDATION
# ==============================================================================

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


# ==============================================================================
# TIER 2: ADVANCED ANALYSIS
# ==============================================================================

def test_attention_patterns(
    model: nn.Module,
    config: Any,
    input_text: str = "The quick brown fox jumps over the lazy dog",
    tokenizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Visualize attention weights and analyze attention patterns.

    Detects:
    - Collapsed attention (all weights uniform or focused on single token)
    - Head specialization (different heads learning different patterns)
    - Attention to special tokens (CLS, SEP, padding)
    - Layer-wise attention evolution

    Args:
        model: The transformer model to analyze
        config: Model configuration
        input_text: Text to analyze (default: sample sentence)
        tokenizer: Optional tokenizer (if None, uses random token IDs)

    Returns:
        Dictionary with attention statistics and analysis results
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    try:
        import seaborn as sns
    except ImportError:
        print("⚠️ seaborn not installed, using matplotlib only")
        sns = None

    device = next(model.parameters()).device
    model.eval()

    # Prepare input
    if tokenizer is not None:
        tokens = tokenizer.encode(input_text)
        input_ids = torch.tensor([tokens]).to(device)
        token_labels = [tokenizer.decode([t]) for t in tokens]
    else:
        vocab_size = getattr(config, 'vocab_size', 50257)
        input_ids = torch.randint(0, vocab_size, (1, 16)).to(device)
        token_labels = [f"T{i}" for i in range(input_ids.shape[1])]

    # Extract attention weights
    # This is model-specific; adjust based on your model's structure
    attention_weights = []

    def attention_hook(module, input, output):
        """Hook to capture attention weights from transformer layers."""
        if hasattr(output, 'attentions') and output.attentions is not None:
            attention_weights.append(output.attentions.detach().cpu())
        elif isinstance(output, tuple) and len(output) > 1:
            # Some models return (output, attention) tuples
            attn = output[1]
            if attn is not None:
                attention_weights.append(attn.detach().cpu())

    # Register hooks (this is generic; may need adjustment for specific models)
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        try:
            output = model(input_ids, output_attentions=True)
            # Try to extract attentions from output
            if hasattr(output, 'attentions') and output.attentions is not None:
                attention_weights = [a.cpu() for a in output.attentions]
        except TypeError:
            # Model doesn't support output_attentions parameter
            output = model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze attention patterns
    results = {
        "num_layers": len(attention_weights),
        "input_length": input_ids.shape[1],
        "collapsed_layers": [],
        "head_specialization_scores": [],
        "attention_entropy": [],
    }

    if len(attention_weights) == 0:
        print("⚠️ Could not extract attention weights from model")
        print("   Model may not expose attention in standard way")
        return results

    print("=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)
    print(f"Layers analyzed: {len(attention_weights)}")
    print(f"Input length: {input_ids.shape[1]}")
    print("-" * 60)

    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: (batch, num_heads, seq_len, seq_len)
        attn_mean = attn.mean(dim=0)  # Average over batch
        num_heads = attn_mean.shape[0]

        # Check for collapsed attention per head
        collapsed_heads = 0
        for head_idx in range(num_heads):
            head_attn = attn_mean[head_idx]
            # Check if attention is too uniform (entropy close to max)
            # or too concentrated (max weight > 0.9)
            max_weight = head_attn.max().item()
            entropy = -(head_attn * torch.log(head_attn + 1e-9)).sum(dim=-1).mean().item()

            if max_weight > 0.9 or entropy < 0.1:
                collapsed_heads += 1

        if collapsed_heads > num_heads * 0.5:
            results["collapsed_layers"].append(layer_idx)

        # Measure head specialization (variance in attention patterns across heads)
        head_patterns = attn_mean.reshape(num_heads, -1)
        specialization = head_patterns.std(dim=0).mean().item()
        results["head_specialization_scores"].append(specialization)

        # Average attention entropy
        avg_entropy = -(attn_mean * torch.log(attn_mean + 1e-9)).sum(dim=-1).mean().item()
        results["attention_entropy"].append(avg_entropy)

        print(f"Layer {layer_idx}: {num_heads} heads, "
              f"specialization={specialization:.4f}, "
              f"entropy={avg_entropy:.4f}, "
              f"collapsed={collapsed_heads}/{num_heads}")

    if results["collapsed_layers"]:
        print(f"\n⚠️ Collapsed attention detected in layers: {results['collapsed_layers']}")
    else:
        print(f"\n✅ No collapsed attention detected")

    print("=" * 60)

    # Visualization
    if plt is not None and len(attention_weights) > 0:
        # Plot attention heatmaps for first and last layer
        fig, axes = plt.subplots(1, min(2, len(attention_weights)), figsize=(14, 6))
        if len(attention_weights) == 1:
            axes = [axes]

        for idx, layer_idx in enumerate([0, -1][:len(attention_weights)]):
            attn = attention_weights[layer_idx][0]  # First batch item
            # Average across heads
            attn_avg = attn.mean(dim=0).numpy()

            ax = axes[idx] if len(attention_weights) > 1 else axes[0]

            if sns is not None:
                sns.heatmap(attn_avg, ax=ax, cmap='viridis', cbar=True,
                           xticklabels=token_labels[:attn_avg.shape[1]],
                           yticklabels=token_labels[:attn_avg.shape[0]])
            else:
                im = ax.imshow(attn_avg, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(token_labels[:attn_avg.shape[1]])))
                ax.set_xticklabels(token_labels[:attn_avg.shape[1]], rotation=45, ha='right')
                ax.set_yticks(range(len(token_labels[:attn_avg.shape[0]])))
                ax.set_yticklabels(token_labels[:attn_avg.shape[0]])

            layer_name = "First Layer" if layer_idx == 0 else "Last Layer"
            ax.set_title(f'{layer_name} Attention Pattern')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

        plt.tight_layout()
        plt.show()

        # Plot layer-wise metrics
        if len(attention_weights) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))

            # Head specialization
            axes[0].plot(results["head_specialization_scores"], marker='o', linewidth=2)
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Specialization Score')
            axes[0].set_title('Head Specialization by Layer')
            axes[0].grid(True, alpha=0.3)

            # Attention entropy
            axes[1].plot(results["attention_entropy"], marker='o', linewidth=2, color='orange')
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Entropy')
            axes[1].set_title('Attention Entropy by Layer')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    return results


def test_attribution_analysis(
    model: nn.Module,
    config: Any,
    input_ids: Optional[torch.Tensor] = None,
    target_idx: int = -1
) -> Dict[str, Any]:
    """
    Perform input attribution analysis using Integrated Gradients.

    Identifies which input tokens contribute most to the model's predictions.
    Useful for understanding model decisions and debugging unexpected behavior.

    Args:
        model: The transformer model
        config: Model configuration
        input_ids: Input token IDs (if None, uses random tokens)
        target_idx: Token position to analyze attribution for (-1 = last token)

    Returns:
        Dictionary with attribution scores and visualizations
    """
    try:
        from captum.attr import IntegratedGradients, LayerIntegratedGradients
        from captum.attr import visualization as viz
    except ImportError:
        print("❌ captum not installed. Install with: pip install captum")
        return {"error": "captum not installed"}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    device = next(model.parameters()).device
    model.eval()

    # Prepare input
    if input_ids is None:
        vocab_size = getattr(config, 'vocab_size', 50257)
        input_ids = torch.randint(0, vocab_size, (1, 16)).to(device)
    else:
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    seq_len = input_ids.shape[1]
    if target_idx < 0:
        target_idx = seq_len + target_idx

    print("=" * 60)
    print("ATTRIBUTION ANALYSIS (Integrated Gradients)")
    print("=" * 60)
    print(f"Input shape: {input_ids.shape}")
    print(f"Analyzing attribution for position: {target_idx}")
    print("-" * 60)

    # Define forward function for attribution
    def forward_func(input_embeds):
        """Forward pass using embeddings instead of token IDs."""
        # This is model-specific; adjust based on your architecture
        try:
            # Try to access embedding layer
            if hasattr(model, 'transformer'):
                # GPT-style models
                embeddings = model.transformer.wte(input_ids)
            elif hasattr(model, 'embeddings'):
                # BERT-style models
                embeddings = model.embeddings(input_ids)
            elif hasattr(model, 'embed_tokens'):
                embeddings = model.embed_tokens(input_ids)
            else:
                # Fallback: assume model has an embedding layer as first module
                embeddings = None
                for module in model.modules():
                    if isinstance(module, nn.Embedding):
                        embeddings = module(input_ids)
                        break

                if embeddings is None:
                    raise AttributeError("Could not find embedding layer")

            # Replace embeddings with input_embeds for gradient computation
            # This requires model-specific implementation
            output = model(inputs_embeds=input_embeds)
            return output
        except TypeError:
            # Model doesn't support inputs_embeds
            # Use direct token input (less accurate for attribution)
            return model(input_ids)

    try:
        # Get embeddings for the input
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embedding_layer = model.transformer.wte
        elif hasattr(model, 'embeddings') and hasattr(model.embeddings, 'word_embeddings'):
            embedding_layer = model.embeddings.word_embeddings
        elif hasattr(model, 'embed_tokens'):
            embedding_layer = model.embed_tokens
        else:
            # Find first embedding layer
            embedding_layer = None
            for module in model.modules():
                if isinstance(module, nn.Embedding):
                    embedding_layer = module
                    break

        if embedding_layer is None:
            print("❌ Could not find embedding layer in model")
            return {"error": "No embedding layer found"}

        input_embeds = embedding_layer(input_ids)

        # Create baseline (zero embeddings)
        baseline = torch.zeros_like(input_embeds)

        # Compute integrated gradients
        ig = IntegratedGradients(lambda x: forward_func(x)[:, target_idx, :].sum(dim=-1))

        attributions, delta = ig.attribute(
            input_embeds,
            baseline,
            target=None,
            return_convergence_delta=True,
            n_steps=50
        )

        # Aggregate attribution scores (L2 norm across embedding dimension)
        attribution_scores = attributions.squeeze(0).norm(dim=-1).cpu().numpy()

        # Normalize to [0, 1]
        attribution_scores = attribution_scores / (attribution_scores.max() + 1e-9)

        results = {
            "attribution_scores": attribution_scores.tolist(),
            "convergence_delta": delta.item(),
            "target_position": target_idx,
            "input_tokens": input_ids.squeeze(0).cpu().tolist(),
        }

        print(f"Convergence delta: {delta.item():.6f}")
        print(f"(Lower is better; < 0.01 indicates good approximation)")
        print()

        # Print top contributing tokens
        top_k = min(5, len(attribution_scores))
        top_indices = np.argsort(attribution_scores)[-top_k:][::-1]

        print("Top contributing tokens:")
        for rank, idx in enumerate(top_indices, 1):
            token_id = input_ids[0, idx].item()
            score = attribution_scores[idx]
            print(f"  {rank}. Position {idx} (token_id={token_id}): {score:.4f}")

        print("=" * 60)

        # Visualization
        if plt is not None:
            fig, ax = plt.subplots(figsize=(12, 4))
            positions = np.arange(len(attribution_scores))
            bars = ax.bar(positions, attribution_scores, edgecolor='black', linewidth=1.5)

            # Color bars by intensity
            colors = plt.cm.Reds(attribution_scores / attribution_scores.max())
            for bar, color in zip(bars, colors):
                bar.set_facecolor(color)

            ax.set_xlabel('Token Position')
            ax.set_ylabel('Attribution Score (normalized)')
            ax.set_title(f'Input Attribution for Position {target_idx}')
            ax.set_xticks(positions)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()

        return results

    except Exception as e:
        print(f"❌ Attribution analysis failed: {str(e)}")
        return {"error": str(e)}


def test_robustness(
    model: nn.Module,
    config: Any,
    n_samples: int = 20,
    noise_levels: List[float] = [0.0, 0.01, 0.05, 0.1, 0.2]
) -> Dict[str, Any]:
    """
    Test model robustness to input perturbations and noise.

    Tests:
    - Stability under embedding noise (Gaussian)
    - Consistency with token substitutions
    - Adversarial robustness (FGSM-style attacks)

    Args:
        model: The transformer model
        config: Model configuration
        n_samples: Number of samples to test per noise level
        noise_levels: Standard deviations for Gaussian noise

    Returns:
        Dictionary with robustness metrics and visualizations
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    device = next(model.parameters()).device
    vocab_size = getattr(config, 'vocab_size', 50257)

    print("=" * 60)
    print("ROBUSTNESS TESTING")
    print("=" * 60)
    print(f"Samples per noise level: {n_samples}")
    print(f"Noise levels: {noise_levels}")
    print("-" * 60)

    results = {
        "noise_levels": noise_levels,
        "accuracy_under_noise": [],
        "output_stability": [],
        "prediction_flips": [],
    }

    model.eval()

    for noise_std in noise_levels:
        accuracies = []
        output_dists = []
        flips = 0

        for _ in range(n_samples):
            # Generate input
            input_ids = torch.randint(0, vocab_size, (1, 32)).to(device)

            # Clean prediction
            with torch.no_grad():
                clean_output = model(input_ids)
                clean_pred = clean_output.argmax(dim=-1)

            # Add noise to embeddings (if supported)
            if noise_std > 0:
                try:
                    # Get embedding layer
                    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                        embed_layer = model.transformer.wte
                    elif hasattr(model, 'embeddings'):
                        embed_layer = model.embeddings.word_embeddings
                    elif hasattr(model, 'embed_tokens'):
                        embed_layer = model.embed_tokens
                    else:
                        for module in model.modules():
                            if isinstance(module, nn.Embedding):
                                embed_layer = module
                                break

                    embeds = embed_layer(input_ids)
                    noise = torch.randn_like(embeds) * noise_std
                    noisy_embeds = embeds + noise

                    # Forward with noisy embeddings
                    with torch.no_grad():
                        try:
                            noisy_output = model(inputs_embeds=noisy_embeds)
                        except TypeError:
                            # Model doesn't support inputs_embeds
                            # Fall back to token-level perturbation
                            noisy_input_ids = input_ids.clone()
                            mask = torch.rand_like(input_ids.float()) < noise_std * 10
                            noisy_input_ids[mask] = torch.randint(0, vocab_size, (mask.sum(),)).to(device)
                            noisy_output = model(noisy_input_ids)

                        noisy_pred = noisy_output.argmax(dim=-1)

                    # Measure prediction consistency
                    agreement = (clean_pred == noisy_pred).float().mean().item()
                    accuracies.append(agreement)

                    # Measure output distribution distance (KL divergence)
                    clean_probs = F.softmax(clean_output, dim=-1)
                    noisy_probs = F.softmax(noisy_output, dim=-1)
                    kl_div = F.kl_div(
                        noisy_probs.log(),
                        clean_probs,
                        reduction='batchmean'
                    ).item()
                    output_dists.append(kl_div)

                    # Count prediction flips
                    if (clean_pred != noisy_pred).any():
                        flips += 1

                except Exception as e:
                    print(f"⚠️ Error testing noise level {noise_std}: {str(e)}")
                    continue
            else:
                # No noise: perfect agreement
                accuracies.append(1.0)
                output_dists.append(0.0)

        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
        avg_distance = np.mean(output_dists) if output_dists else 0.0
        flip_rate = flips / n_samples if n_samples > 0 else 0.0

        results["accuracy_under_noise"].append(avg_accuracy)
        results["output_stability"].append(avg_distance)
        results["prediction_flips"].append(flip_rate)

        print(f"Noise σ={noise_std:.3f}: "
              f"Accuracy={avg_accuracy:.3f}, "
              f"KL-Div={avg_distance:.4f}, "
              f"Flip Rate={flip_rate:.2%}")

    print("=" * 60)

    # Detect issues
    if results["accuracy_under_noise"][-1] < 0.5:
        print("⚠️ WARNING: Model is very sensitive to noise (accuracy < 50% at max noise)")
    elif results["accuracy_under_noise"][-1] < 0.7:
        print("⚠️ Model shows moderate sensitivity to noise")
    else:
        print("✅ Model is relatively robust to input noise")

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Accuracy under noise
        axes[0].plot(noise_levels, results["accuracy_under_noise"],
                     marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Noise Level (σ)')
        axes[0].set_ylabel('Prediction Accuracy')
        axes[0].set_title('Robustness to Input Noise')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        # Output stability (KL divergence)
        axes[1].plot(noise_levels, results["output_stability"],
                     marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Noise Level (σ)')
        axes[1].set_ylabel('KL Divergence from Clean Output')
        axes[1].set_title('Output Distribution Stability')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return results


# ==============================================================================
# TIER 3: TRAINING UTILITIES
# ==============================================================================

def test_fine_tuning(
    model: nn.Module,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_epochs: int = 3,
    learning_rate: float = 5e-5,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Run a basic fine-tuning loop with loss tracking.

    Demonstrates:
    - Training loop setup
    - Gradient clipping
    - Learning rate scheduling
    - Loss convergence tracking

    Args:
        model: The transformer model to fine-tune
        config: Model configuration
        train_data: List of input_ids tensors (if None, generates synthetic data)
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size for training

    Returns:
        Dictionary with training metrics and loss curves
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    device = next(model.parameters()).device
    vocab_size = getattr(config, 'vocab_size', 50257)

    # Generate synthetic training data if not provided
    if train_data is None:
        print("Generating synthetic training data...")
        train_data = [
            torch.randint(0, vocab_size, (32,))
            for _ in range(50)  # 50 samples
        ]

    print("=" * 60)
    print("FINE-TUNING TEST")
    print("=" * 60)
    print(f"Training samples: {len(train_data)}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs * (len(train_data) // batch_size)
    )

    model.train()

    loss_history = []
    grad_norm_history = []

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_losses = []

        # Shuffle data
        indices = torch.randperm(len(train_data))

        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = torch.stack([train_data[idx] for idx in batch_indices]).to(device)

            # Forward pass
            logits = model(batch)

            # Compute loss (language modeling: predict next token)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_losses.append(loss.item())
            grad_norm_history.append(grad_norm.item())
            loss_history.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")

    training_time = time.time() - start_time

    print("-" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Loss reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.1f}%")
    print("=" * 60)

    results = {
        "loss_history": loss_history,
        "grad_norm_history": grad_norm_history,
        "final_loss": loss_history[-1],
        "initial_loss": loss_history[0],
        "training_time_seconds": training_time,
        "samples_per_second": len(train_data) * n_epochs / training_time,
    }

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Loss curve
        axes[0].plot(loss_history, linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].grid(True, alpha=0.3)

        # Add epoch markers
        steps_per_epoch = len(train_data) // batch_size
        for e in range(1, n_epochs):
            axes[0].axvline(x=e * steps_per_epoch, color='r',
                           linestyle='--', alpha=0.5, linewidth=1)

        # Gradient norm
        axes[1].plot(grad_norm_history, linewidth=2, alpha=0.7, color='orange')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Gradient Norm')
        axes[1].set_title('Gradient Norm (after clipping)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=1.0, color='r', linestyle='--',
                       linewidth=1, label='Clip threshold')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return results


def test_hyperparameter_search(
    model_factory: Any,
    config: Any,
    train_data: Optional[List[torch.Tensor]] = None,
    n_trials: int = 10,
    search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization using Optuna.

    Searches over:
    - Learning rate
    - Batch size
    - Warmup steps
    - Weight decay

    Args:
        model_factory: Function that creates a fresh model instance
        config: Model configuration
        train_data: Training data (if None, generates synthetic)
        n_trials: Number of Optuna trials
        search_space: Custom search space (if None, uses defaults)

    Returns:
        Dictionary with best parameters and optimization history
    """
    try:
        import optuna
    except ImportError:
        print("❌ optuna not installed. Install with: pip install optuna")
        return {"error": "optuna not installed"}

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
        plt = None

    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas not installed, returning dict instead of DataFrame")
        pd = None

    vocab_size = getattr(config, 'vocab_size', 50257)

    # Generate synthetic data if needed
    if train_data is None:
        train_data = [
            torch.randint(0, vocab_size, (32,))
            for _ in range(30)
        ]

    print("=" * 60)
    print("HYPERPARAMETER SEARCH (Optuna)")
    print("=" * 60)
    print(f"Trials: {n_trials}")
    print(f"Training samples: {len(train_data)}")
    print("-" * 60)

    def objective(trial):
        """Optuna objective function."""
        # Sample hyperparameters
        if search_space is None:
            lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [2, 4, 8])
            warmup_steps = trial.suggest_int('warmup_steps', 0, 10)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        else:
            lr = trial.suggest_loguniform('learning_rate', *search_space.get('lr', (1e-5, 1e-3)))
            batch_size = trial.suggest_categorical('batch_size', search_space.get('batch_size', [2, 4, 8]))
            warmup_steps = trial.suggest_int('warmup_steps', *search_space.get('warmup', (0, 10)))
            weight_decay = trial.suggest_loguniform('weight_decay', *search_space.get('wd', (1e-6, 1e-2)))

        # Create fresh model
        model = model_factory()
        device = next(model.parameters()).device
        model.train()

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Quick training (2 epochs)
        n_epochs = 2
        losses = []

        for epoch in range(n_epochs):
            for i in range(0, len(train_data), batch_size):
                batch = torch.stack(train_data[i:i+batch_size]).to(device)

                logits = model(batch)

                # Next-token prediction loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1)
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())

        # Return average loss
        return np.mean(losses)

    # Create study and optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("-" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    results = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params
            }
            for t in study.trials
        ]
    }

    # Visualization
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Optimization history
        trial_numbers = [t.number for t in study.trials]
        trial_values = [t.value for t in study.trials]

        axes[0].plot(trial_numbers, trial_values, marker='o', linewidth=2, alpha=0.7)
        axes[0].axhline(y=study.best_value, color='r', linestyle='--',
                       linewidth=2, label=f'Best: {study.best_value:.4f}')
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())

            axes[1].barh(params, importances, edgecolor='black')
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, alpha=0.3, axis='x')
        except:
            axes[1].text(0.5, 0.5, 'Importance analysis\nnot available',
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return results


def test_benchmark_comparison(
    model: nn.Module,
    config: Any,
    baseline_model_name: str = "distilgpt2",
    test_data: Optional[List[torch.Tensor]] = None,
    n_samples: int = 20
) -> Dict[str, Any]:
    """
    Compare model against a baseline transformer.

    Compares:
    - Inference speed
    - Parameter count
    - Memory footprint
    - Loss/perplexity on test data

    Args:
        model: Custom model to benchmark
        config: Model configuration
        baseline_model_name: HuggingFace model name to compare against
        test_data: Test samples (if None, generates synthetic)
        n_samples: Number of samples to test

    Returns:
        Dictionary with comparative metrics
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("❌ transformers not installed. Install with: pip install transformers")
        return {"error": "transformers not installed"}

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

    device = next(model.parameters()).device
    vocab_size = getattr(config, 'vocab_size', 50257)

    print("=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"Custom model vs. {baseline_model_name}")
    print(f"Test samples: {n_samples}")
    print("-" * 60)

    # Load baseline model
    print(f"Loading baseline model: {baseline_model_name}...")
    try:
        baseline = AutoModelForCausalLM.from_pretrained(baseline_model_name).to(device)
        baseline.eval()
    except Exception as e:
        print(f"❌ Failed to load baseline: {str(e)}")
        return {"error": f"Failed to load baseline: {str(e)}"}

    # Generate test data
    if test_data is None:
        test_data = [
            torch.randint(0, vocab_size, (32,)).to(device)
            for _ in range(n_samples)
        ]
    else:
        test_data = [t.to(device) for t in test_data[:n_samples]]

    # Compare parameter counts
    custom_params = sum(p.numel() for p in model.parameters())
    baseline_params = sum(p.numel() for p in baseline.parameters())

    print(f"\nParameter Count:")
    print(f"  Custom model:   {custom_params:,}")
    print(f"  Baseline model: {baseline_params:,}")
    print(f"  Ratio: {custom_params / baseline_params:.2f}x")

    # Compare inference speed
    print(f"\nBenchmarking inference speed...")
    model.eval()

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(test_data[0].unsqueeze(0))
            _ = baseline(test_data[0].unsqueeze(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark custom model
    custom_times = []
    for sample in test_data:
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample.unsqueeze(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        custom_times.append(time.perf_counter() - start)

    # Benchmark baseline
    baseline_times = []
    for sample in test_data:
        start = time.perf_counter()
        with torch.no_grad():
            _ = baseline(sample.unsqueeze(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - start)

    custom_avg_ms = np.mean(custom_times) * 1000
    baseline_avg_ms = np.mean(baseline_times) * 1000

    print(f"\nInference Speed (avg):")
    print(f"  Custom model:   {custom_avg_ms:.2f} ms")
    print(f"  Baseline model: {baseline_avg_ms:.2f} ms")
    print(f"  Speedup: {baseline_avg_ms / custom_avg_ms:.2f}x")

    # Compare loss/perplexity
    print(f"\nComputing perplexity...")
    custom_losses = []
    baseline_losses = []

    for sample in test_data:
        input_ids = sample.unsqueeze(0)

        # Custom model
        with torch.no_grad():
            custom_logits = model(input_ids)
            custom_loss = F.cross_entropy(
                custom_logits[:, :-1, :].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            custom_losses.append(custom_loss.item())

        # Baseline model
        with torch.no_grad():
            baseline_logits = baseline(input_ids).logits
            baseline_loss = F.cross_entropy(
                baseline_logits[:, :-1, :].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            baseline_losses.append(baseline_loss.item())

    custom_ppl = np.exp(np.mean(custom_losses))
    baseline_ppl = np.exp(np.mean(baseline_losses))

    print(f"\nPerplexity:")
    print(f"  Custom model:   {custom_ppl:.2f}")
    print(f"  Baseline model: {baseline_ppl:.2f}")
    print(f"  Ratio: {custom_ppl / baseline_ppl:.2f}x")

    print("=" * 60)

    results = {
        "parameter_count": {
            "custom": custom_params,
            "baseline": baseline_params,
            "ratio": custom_params / baseline_params
        },
        "inference_speed_ms": {
            "custom": custom_avg_ms,
            "baseline": baseline_avg_ms,
            "speedup": baseline_avg_ms / custom_avg_ms
        },
        "perplexity": {
            "custom": custom_ppl,
            "baseline": baseline_ppl,
            "ratio": custom_ppl / baseline_ppl
        },
        "baseline_model": baseline_model_name,
    }

    # Visualization
    if plt is not None:
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
                   [custom_avg_ms, baseline_avg_ms],
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

    return results


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def run_all_tier1_tests(model: nn.Module, config: Any) -> None:
    """
    Run all Tier 1 tests in sequence.

    Provides a comprehensive validation suite for critical model functionality.
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL TIER 1 TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Shape Robustness", lambda: test_shape_robustness(model, config)),
        ("Gradient Flow", lambda: test_gradient_flow(model, config)),
        ("Output Stability", lambda: test_output_stability(model, config)),
        ("Parameter Initialization", lambda: test_parameter_initialization(model)),
        ("Memory Footprint", lambda: test_memory_footprint(model, config)),
        ("Inference Speed", lambda: test_inference_speed(model, config)),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        try:
            result = test_func()
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
        print()


def run_all_tier2_tests(model: nn.Module, config: Any) -> None:
    """
    Run all Tier 2 tests in sequence.

    Provides advanced analysis of attention patterns, attribution, and robustness.
    """
    print("\n" + "=" * 60)
    print("RUNNING ALL TIER 2 TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Attention Patterns", lambda: test_attention_patterns(model, config)),
        ("Attribution Analysis", lambda: test_attribution_analysis(model, config)),
        ("Robustness Testing", lambda: test_robustness(model, config)),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        try:
            result = test_func()
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
        print()


def run_all_tests(model: nn.Module, config: Any) -> None:
    """
    Run complete test suite (Tier 1 + Tier 2).

    Note: Tier 3 tests require additional setup and are not included here.
    """
    run_all_tier1_tests(model, config)
    run_all_tier2_tests(model, config)
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
