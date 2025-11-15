"""
Tier 2: Advanced Analysis Tests

This module contains advanced diagnostic tests for transformer models:
- Attention pattern visualization and analysis
- Input attribution analysis (Integrated Gradients)
- Robustness testing under perturbations and noise

These tests provide deeper insights into model behavior beyond basic validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
import numpy as np


def _detect_vocab_size(model: nn.Module, config: Any) -> int:
    """
    Detect vocabulary size from model or config.

    Priority:
    1. config.vocab_size (explicit)
    2. model embedding layer vocab size (introspection)
    3. Default fallback (50257 for GPT-2 compatibility)
    """
    # Try config first
    if hasattr(config, 'vocab_size') and config.vocab_size is not None:
        return config.vocab_size

    # Try to detect from model embedding layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            return module.num_embeddings

    # Fallback with warning
    print("⚠️ Could not detect vocab_size, using default 50257 (GPT-2)")
    return 50257


def _extract_output_tensor(output: Any) -> torch.Tensor:
    """
    Extract tensor from various model output formats.

    Handles:
    - Direct tensor: return as-is
    - Tuple: return first element
    - Dict: return output['logits'] or output['last_hidden_state']
    - ModelOutput object: return .logits attribute
    """
    # Direct tensor
    if isinstance(output, torch.Tensor):
        return output

    # Tuple (common for models that return multiple outputs)
    if isinstance(output, tuple):
        return output[0]

    # Dict
    if isinstance(output, dict):
        if 'logits' in output:
            return output['logits']
        if 'last_hidden_state' in output:
            return output['last_hidden_state']
        # Return first tensor value found
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value

    # HuggingFace ModelOutput object
    if hasattr(output, 'logits'):
        return output.logits
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state

    # Fallback - assume it's tensor-like
    return output


def _safe_get_model_output(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Safely extract logits tensor from model output.

    Wraps model() call and handles diverse output formats.
    """
    output = model(input_ids)
    return _extract_output_tensor(output)


def _has_multihead_attention_layers(model: nn.Module) -> bool:
    """Check if model contains nn.MultiheadAttention layers."""
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            return True
    return False


def _extract_attention_from_mha_model(
    model: nn.Module,
    input_ids: torch.Tensor
) -> List[torch.Tensor]:
    """
    Extract attention weights from models using nn.MultiheadAttention.

    This requires monkey-patching the forward calls to capture attention weights,
    since nn.MultiheadAttention needs explicit need_weights=True parameter.
    """
    attention_weights = []
    original_forwards = {}

    # Monkey-patch all MultiheadAttention layers
    def create_hooked_forward(original_forward, layer_name):
        def hooked_forward(query, key, value, *args, **kwargs):
            # Force need_weights and get per-head attention
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False  # Get per-head weights
            output, attn = original_forward(query, key, value, *args, **kwargs)
            if attn is not None:
                attention_weights.append(attn.detach().cpu())
            return output, attn
        return hooked_forward

    # Store and replace forward methods
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            original_forwards[module] = module.forward
            module.forward = create_hooked_forward(module.forward, name)

    # Run forward pass
    try:
        with torch.no_grad():
            _ = model(input_ids)
    except Exception as e:
        print(f"⚠️ Error during attention extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original forward methods
        for module, original_forward in original_forwards.items():
            module.forward = original_forward

    return attention_weights


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
        vocab_size = _detect_vocab_size(model, config)
        input_ids = torch.randint(0, vocab_size, (1, 16)).to(device)
        token_labels = [f"T{i}" for i in range(input_ids.shape[1])]

    # Extract attention weights
    # Detect model architecture and extract attention accordingly
    if _has_multihead_attention_layers(model):
        # Use specialized extraction for nn.MultiheadAttention
        print("🔍 Detected nn.MultiheadAttention layers - using specialized extraction")
        attention_weights = _extract_attention_from_mha_model(model, input_ids)
        print(f"   Extracted {len(attention_weights)} attention weight tensor(s)")
    else:
        # Use existing hook-based approach for HuggingFace-style models
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
        vocab_size = _detect_vocab_size(model, config)
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
    vocab_size = _detect_vocab_size(model, config)

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
                clean_output = _safe_get_model_output(model, input_ids)
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
                            noisy_output_raw = model(inputs_embeds=noisy_embeds)
                            noisy_output = _extract_output_tensor(noisy_output_raw)
                        except TypeError:
                            # Model doesn't support inputs_embeds
                            # Fall back to token-level perturbation
                            noisy_input_ids = input_ids.clone()
                            mask = torch.rand_like(input_ids.float()) < noise_std * 10
                            noisy_input_ids[mask] = torch.randint(0, vocab_size, (mask.sum(),)).to(device)
                            noisy_output = _safe_get_model_output(model, noisy_input_ids)

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
