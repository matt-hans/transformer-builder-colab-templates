"""
Weights & Biases configuration and initialization helpers for training notebooks.

Reduces cyclomatic complexity by extracting W&B setup logic into focused functions.
"""

import torch
import torch.nn as nn
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Any, Optional, Literal


def detect_model_type(model: nn.Module) -> Literal['gpt', 'bert', 't5', 'custom']:
    """
    Detect transformer architecture type from model structure.

    Inspects class name and module structure to infer architecture type.

    Args:
        model: PyTorch model to analyze

    Returns:
        One of: 'gpt', 'bert', 't5', or 'custom'

    Examples:
        >>> model_type = detect_model_type(gpt_model)
        >>> print(model_type)  # 'gpt'
    """
    model_class = model.__class__.__name__.lower()

    # Check class name first
    if _is_gpt_style(model_class):
        return 'gpt'
    elif _is_bert_style(model_class):
        return 'bert'
    elif _is_t5_style(model_class):
        return 't5'

    # Inspect module structure
    module_names = [name for name, _ in model.named_modules()]
    architecture = _infer_from_modules(module_names)

    return architecture


def _is_gpt_style(class_name: str) -> bool:
    """Check if class name suggests GPT architecture."""
    return 'gpt' in class_name or 'decoder' in class_name


def _is_bert_style(class_name: str) -> bool:
    """Check if class name suggests BERT architecture."""
    return 'bert' in class_name or 'encoder' in class_name


def _is_t5_style(class_name: str) -> bool:
    """Check if class name suggests T5 architecture."""
    return 't5' in class_name or 'encoderdecoder' in class_name


def _infer_from_modules(module_names: list) -> Literal['gpt', 'bert', 't5', 'custom']:
    """Infer architecture from module structure."""
    has_decoder = any('decoder' in name.lower() for name in module_names)
    has_encoder = any('encoder' in name.lower() for name in module_names)

    if has_decoder and not has_encoder:
        return 'gpt'
    elif has_encoder and not has_decoder:
        return 'bert'
    elif has_encoder and has_decoder:
        return 't5'

    return 'custom'


def build_wandb_config(
    model: nn.Module,
    config: SimpleNamespace,
    hyperparameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build W&B config dictionary with hyperparameters, model metadata, and environment info.

    Args:
        model: PyTorch model
        config: Model configuration object (must have vocab_size, max_seq_len)
        hyperparameters: Optional training hyperparameters dict

    Returns:
        Complete W&B config dictionary ready for wandb.init(config=...)

    Examples:
        >>> config_dict = build_wandb_config(model, config, {
        ...     'learning_rate': 5e-5,
        ...     'batch_size': 4
        ... })
        >>> run = wandb.init(config=config_dict)
    """
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = _get_default_hyperparameters()

    # Calculate model metadata
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    model_type = detect_model_type(model)
    device_str = str(next(model.parameters()).device)

    # Build config dictionary
    wandb_config = {
        # Hyperparameters
        "learning_rate": hyperparameters.get('learning_rate', 5e-5),
        "batch_size": hyperparameters.get('batch_size', 2),
        "epochs": hyperparameters.get('epochs', 3),
        "warmup_ratio": hyperparameters.get('warmup_ratio', 0.1),
        "weight_decay": hyperparameters.get('weight_decay', 0.01),
        "max_grad_norm": hyperparameters.get('max_grad_norm', 1.0),
        # Reproducibility
        "random_seed": hyperparameters.get('random_seed', None),
        "deterministic_mode": hyperparameters.get('deterministic', False),

        # Model architecture
        "model_type": model_type,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_millions": round(total_params / 1e6, 2),

        # Environment
        "device": device_str,
        "mixed_precision": hyperparameters.get('use_amp', True),
        "gradient_accumulation_steps": hyperparameters.get('grad_accum_steps', 1),
    }

    return wandb_config


def _get_default_hyperparameters() -> Dict[str, Any]:
    """Get default hyperparameters for training."""
    return {
        'learning_rate': 5e-5,
        'batch_size': 2,
        'epochs': 3,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'grad_accum_steps': 1
    }


def initialize_wandb_run(
    model: nn.Module,
    config: SimpleNamespace,
    project_name: str = "transformer-builder-training",
    hyperparameters: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None
):
    """
    Initialize W&B run with automatic config generation.

    Args:
        model: PyTorch model
        config: Model configuration object
        project_name: W&B project name
        hyperparameters: Optional training hyperparameters
        tags: Optional list of tags (default: [model_type, "v1", "tier3"])

    Returns:
        wandb.Run object

    Requires:
        wandb package must be imported and authenticated

    Examples:
        >>> import wandb
        >>> run = initialize_wandb_run(model, config)
        >>> print(run.get_url())
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb package required - install with: pip install wandb")

    # Detect model type for tags and run name
    model_type = detect_model_type(model)

    # Generate run name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{model_type}_{timestamp}"

    # Default tags
    if tags is None:
        tags = [model_type, "v1", "tier3"]

    # Build config
    wandb_config = build_wandb_config(model, config, hyperparameters)

    # Initialize run
    run = wandb.init(
        project=project_name,
        name=run_name,
        tags=tags,
        config=wandb_config
    )

    return run


def print_wandb_summary(run, model: nn.Module, hyperparameters: Dict[str, Any]) -> None:
    """
    Print W&B run summary with formatted output.

    Args:
        run: wandb.Run object
        model: PyTorch model
        hyperparameters: Training hyperparameters dict

    Examples:
        >>> run = wandb.init(...)
        >>> print_wandb_summary(run, model, {'learning_rate': 5e-5})
    """
    model_type = detect_model_type(model)
    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 80)
    print("📊 W&B TRACKING INITIALIZED")
    print("=" * 80)
    print()
    print(f"🎯 Project: {run.project}")
    print(f"🏷️  Run name: {run.name}")
    print(f"🔗 Dashboard: {run.get_url()}")
    print()
    print(f"📋 Logged config:")
    print(f"   • Model: {model_type} ({round(total_params/1e6, 2)}M params)")
    print(f"   • Learning rate: {hyperparameters.get('learning_rate', 'N/A')}")
    print(f"   • Batch size: {hyperparameters.get('batch_size', 'N/A')}")
    print(f"   • Epochs: {hyperparameters.get('epochs', 'N/A')}")
    print()
