"""
Model instantiation and configuration helpers for training notebooks.

Reduces cyclomatic complexity by extracting model setup logic into focused functions.
"""

import inspect
import json
import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import Any, Dict, Optional, Type


def find_model_class(
    globals_dict: Dict[str, Any],
    model_name: Optional[str] = None
) -> Optional[Type[nn.Module]]:
    """
    Find the model class from globals by name or by inheritance.

    Args:
        globals_dict: Global namespace dictionary (typically from globals())
        model_name: Optional specific model class name to find

    Returns:
        Model class or None if not found

    Examples:
        >>> model_class = find_model_class(globals(), "CustomTransformer")
        >>> model_class = find_model_class(globals())  # Auto-detect
    """
    # First pass: Try to find by exact name
    if model_name:
        for name, obj in globals_dict.items():
            if not _is_model_class(obj):
                continue
            if name == model_name:
                return obj

    # Second pass: Return first nn.Module subclass found
    for name, obj in globals_dict.items():
        if _is_model_class(obj):
            return obj

    return None


def _is_model_class(obj: Any) -> bool:
    """Check if object is a valid model class (nn.Module subclass)."""
    return (
        isinstance(obj, type) and
        issubclass(obj, nn.Module) and
        obj is not nn.Module
    )


def instantiate_model(
    model_class: Type[nn.Module],
    config_dict: Dict[str, Any]
) -> nn.Module:
    """
    Instantiate model with config, handling both no-arg and config-based constructors.

    Args:
        model_class: The model class to instantiate
        config_dict: Configuration dictionary to pass to constructor

    Returns:
        Instantiated model in eval mode

    Raises:
        ValueError: If model instantiation fails

    Examples:
        >>> model = instantiate_model(GPTModel, {"vocab_size": 50257})
        >>> model = instantiate_model(BERTModel, {})  # No-arg constructor
    """
    sig = inspect.signature(model_class.__init__)
    params_list = [p for p in sig.parameters.values() if p.name != 'self']

    # No-argument constructor
    if len(params_list) == 0:
        model = model_class()
    else:
        # Pass full config dict
        model = model_class(**config_dict)

    model.eval()
    return model


def create_model_config(config_dict: Dict[str, Any]) -> SimpleNamespace:
    """
    Create unified config object from Transformer Builder config JSON.

    Extracts vocab_size, max_seq_len from nested node structure and
    creates a SimpleNamespace with standardized attributes.

    Args:
        config_dict: Raw config dictionary from config.json

    Returns:
        SimpleNamespace with standardized attributes:
            - vocab_size (default: 50257)
            - max_seq_len (default: 512)
            - max_batch_size (default: 8)
            - All other top-level config keys

    Examples:
        >>> config = create_model_config({"nodes": [{"params": {"vocab_size": 32000}}]})
        >>> print(config.vocab_size)  # 32000
    """
    config = SimpleNamespace(
        vocab_size=50257,
        max_seq_len=512,
        max_batch_size=8
    )

    # Extract from nested node structure (Transformer Builder format)
    if 'nodes' in config_dict:
        _extract_from_nodes(config, config_dict['nodes'])

    # Copy all other top-level keys (excluding metadata)
    _copy_top_level_keys(config, config_dict)

    return config


def _extract_from_nodes(config: SimpleNamespace, nodes: list) -> None:
    """Extract vocab_size and max_seq_len from nodes array."""
    for node in nodes:
        node_params = node.get('params', {})

        if 'vocab_size' in node_params:
            config.vocab_size = node_params['vocab_size']

        if 'max_seq_len' in node_params:
            config.max_seq_len = node_params['max_seq_len']
        elif 'seq_length' in node_params:
            config.max_seq_len = node_params['seq_length']


def _copy_top_level_keys(config: SimpleNamespace, config_dict: Dict[str, Any]) -> None:
    """Copy non-metadata keys to config object."""
    metadata_keys = {'nodes', 'version', 'model_name'}

    for key, value in config_dict.items():
        if key not in metadata_keys:
            setattr(config, key, value)


def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device the model is currently on.

    Args:
        model: PyTorch model

    Returns:
        Device object (cuda:0, cpu, etc.)

    Examples:
        >>> device = get_model_device(model)
        >>> print(device)  # cuda:0
    """
    return next(model.parameters()).device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts

    Examples:
        >>> counts = count_parameters(model)
        >>> print(f"Total: {counts['total']:,}")  # Total: 124,439,808
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return {
        'total': total_params,
        'trainable': trainable_params
    }


def setup_model_from_gist(
    model_code_path: str,
    config_json_path: str,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None
) -> tuple[nn.Module, SimpleNamespace, Dict[str, int]]:
    """
    Complete model setup from Gist files (high-level orchestrator).

    This is the main entry point that combines all helper functions to:
    1. Execute model code
    2. Load config
    3. Find model class
    4. Instantiate model
    5. Move to device
    6. Return model, config, and parameter counts

    Args:
        model_code_path: Path to model.py (e.g., "custom_transformer.py")
        config_json_path: Path to config.json
        model_name: Optional model class name to find
        device: Optional device to move model to (default: auto-detect GPU/CPU)

    Returns:
        Tuple of (model, config, param_counts)
        - model: Instantiated nn.Module
        - config: SimpleNamespace with standardized attributes
        - param_counts: Dict with 'total' and 'trainable' counts

    Raises:
        RuntimeError: If model class not found or instantiation fails
        FileNotFoundError: If model files don't exist

    Examples:
        >>> model, config, params = setup_model_from_gist(
        ...     "custom_transformer.py",
        ...     "config.json",
        ...     model_name="GPT2Custom"
        ... )
        >>> print(f"Loaded {params['total']:,} parameters")
    """
    # Execute model code to register classes
    exec(open(model_code_path).read(), globals())

    # Load config
    with open(config_json_path) as f:
        config_dict = json.load(f)

    # Find model class
    model_class = find_model_class(globals(), model_name)
    if model_class is None:
        raise RuntimeError(
            f"Could not find model class '{model_name or 'any'}' in {model_code_path}"
        )

    # Instantiate model
    model = instantiate_model(model_class, config_dict)

    # Count parameters
    param_counts = count_parameters(model)

    # Move to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create config object
    config = create_model_config(config_dict)

    return model, config, param_counts
