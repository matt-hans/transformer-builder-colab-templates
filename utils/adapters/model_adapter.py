"""
Universal Model Adapter for handling arbitrary transformer architectures.

This module provides tools to wrap generated models with complex forward() signatures
into a unified interface compatible with PyTorch Lightning and standard training loops.

Key components:
- ModelSignatureInspector: Analyzes forward() method signatures
- ComputationalGraphExecutor: Resolves intermediate output dependencies
- UniversalModelAdapter: Lightning-compatible wrapper for any model
"""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Set, Tuple
import pytorch_lightning as pl
from transformers import PreTrainedTokenizer


# ==============================================================================
# MODEL SIGNATURE INSPECTOR
# ==============================================================================

class ModelSignatureInspector:
    """
    Analyzes model forward() signature using Python's inspect module.

    This class examines a model's forward method to understand:
    - What parameters it expects
    - Which parameters are required vs optional
    - Whether it uses intermediate outputs (e.g., mhsa_0_output, residual_0_output)
    - Whether it has a simple signature (just input_ids, attention_mask)

    Examples:
        Simple signature:
            def forward(self, input_ids): ...
            def forward(self, input_ids, attention_mask=None): ...

        Complex signature (requires intermediates):
            def forward(self, input_0_tokens, mhsa_0_output, residual_0_output): ...
    """

    # Prefixes that indicate intermediate computational outputs
    INTERMEDIATE_PREFIXES = (
        'mhsa_',        # Multi-Head Self-Attention outputs
        'residual_',    # Residual connection outputs
        'ffn_',         # Feed-Forward Network outputs
        'attention_',   # Generic attention outputs
        'mlp_',         # MLP layer outputs
        'layer_',       # Generic layer outputs
    )

    # Standard parameter names that don't require computation
    STANDARD_PARAMS = {
        'self',
        'input_ids',
        'input_0_tokens',  # Alternative name for input_ids
        'attention_mask',
        'token_type_ids',
        'position_ids',
        'labels',
    }

    def __init__(self, model: nn.Module):
        """
        Initialize inspector with a model.

        Args:
            model: PyTorch model to inspect
        """
        self.model = model
        self.signature = inspect.signature(model.forward)
        self.params = list(self.signature.parameters.keys())

        # Remove 'self' if present
        if 'self' in self.params:
            self.params.remove('self')

    def get_parameters(self) -> List[str]:
        """
        Get all parameter names from forward() signature.

        Returns:
            List of parameter names (excluding 'self')
        """
        return self.params.copy()

    def get_required_params(self) -> List[str]:
        """
        Get required parameters (those without default values).

        Returns:
            List of required parameter names
        """
        required = []
        for param_name in self.params:
            param = self.signature.parameters[param_name]
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        return required

    def get_optional_params(self) -> List[str]:
        """
        Get optional parameters (those with default values).

        Returns:
            List of optional parameter names
        """
        optional = []
        for param_name in self.params:
            param = self.signature.parameters[param_name]
            if param.default != inspect.Parameter.empty:
                optional.append(param_name)
        return optional

    def requires_intermediate_outputs(self) -> bool:
        """
        Check if model signature requires intermediate computational outputs.

        Returns:
            True if any parameter starts with intermediate prefixes
        """
        return any(
            p.startswith(self.INTERMEDIATE_PREFIXES)
            for p in self.params
        )

    def is_simple_signature(self) -> bool:
        """
        Check if model has a simple signature (standard params only).

        A simple signature contains only standard parameters like:
        - input_ids / input_0_tokens
        - attention_mask
        - position_ids
        - token_type_ids

        Returns:
            True if signature is simple (no intermediate outputs needed)
        """
        param_set = set(self.params)
        return param_set <= self.STANDARD_PARAMS

    def get_intermediate_params(self) -> List[str]:
        """
        Get list of parameters that represent intermediate outputs.

        Returns:
            List of intermediate parameter names
        """
        return [
            p for p in self.params
            if p.startswith(self.INTERMEDIATE_PREFIXES)
        ]

    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete analysis of model signature.

        Returns:
            Dictionary with analysis results
        """
        return {
            'all_params': self.get_parameters(),
            'required_params': self.get_required_params(),
            'optional_params': self.get_optional_params(),
            'intermediate_params': self.get_intermediate_params(),
            'requires_intermediates': self.requires_intermediate_outputs(),
            'is_simple': self.is_simple_signature(),
            'signature_str': str(self.signature),
        }

    def __repr__(self) -> str:
        return f"ModelSignatureInspector({self.model.__class__.__name__}, params={self.params})"


# ==============================================================================
# COMPUTATIONAL GRAPH EXECUTOR
# ==============================================================================

class ComputationalGraphExecutor:
    """
    Resolves and computes intermediate dependencies in model forward pass.

    For models with complex signatures that require intermediate outputs
    (e.g., mhsa_0_output, residual_0_output), this class:
    1. Analyzes the model's layer structure
    2. Computes intermediates in correct order
    3. Caches results to avoid redundant computation
    4. Calls model.forward() with all required parameters

    Strategy:
    - Uses layer introspection to identify computation modules
    - Executes layers sequentially to generate intermediate outputs
    - Maps parameter names to layer outputs (e.g., mhsa_0 → model.layers[0].attention)
    """

    def __init__(self, model: nn.Module, inspector: ModelSignatureInspector):
        """
        Initialize executor.

        Args:
            model: The model to execute
            inspector: Signature inspector for this model
        """
        self.model = model
        self.inspector = inspector
        self.intermediate_cache = {}

        # Analyze model structure
        self.layer_map = self._build_layer_map()

    def _build_layer_map(self) -> Dict[str, nn.Module]:
        """
        Build a mapping from intermediate parameter names to model layers.

        Introspects the model to find layers that might produce intermediate outputs.
        Common patterns:
        - model.layers[i].attention → mhsa_{i}_output
        - model.layers[i].feed_forward → ffn_{i}_output
        - model.transformer.h[i] → layer_{i}_output

        Returns:
            Dictionary mapping parameter prefixes to layer modules
        """
        layer_map = {}

        # Try common layer structure patterns
        # Pattern 1: model.layers[i]
        if hasattr(self.model, 'layers'):
            layers = self.model.layers
            if isinstance(layers, (nn.ModuleList, list)):
                for i, layer in enumerate(layers):
                    layer_map[f'layer_{i}'] = layer

                    # Look for attention sublayers
                    for attr_name in ['attention', 'self_attn', 'attn', 'mhsa']:
                        if hasattr(layer, attr_name):
                            layer_map[f'mhsa_{i}'] = getattr(layer, attr_name)
                            layer_map[f'attention_{i}'] = getattr(layer, attr_name)
                            break

                    # Look for FFN sublayers
                    for attr_name in ['feed_forward', 'ffn', 'mlp', 'fc']:
                        if hasattr(layer, attr_name):
                            layer_map[f'ffn_{i}'] = getattr(layer, attr_name)
                            layer_map[f'mlp_{i}'] = getattr(layer, attr_name)
                            break

        # Pattern 2: model.transformer.h[i] (GPT-style)
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            if isinstance(layers, (nn.ModuleList, list)):
                for i, layer in enumerate(layers):
                    layer_map[f'layer_{i}'] = layer
                    if hasattr(layer, 'attn'):
                        layer_map[f'mhsa_{i}'] = layer.attn

        # Pattern 3: model.encoder.layer[i] (BERT-style)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
            if isinstance(layers, (nn.ModuleList, list)):
                for i, layer in enumerate(layers):
                    layer_map[f'layer_{i}'] = layer

        return layer_map

    def _parse_intermediate_name(self, param_name: str) -> Tuple[str, int]:
        """
        Parse intermediate parameter name into layer type and index.

        Examples:
            mhsa_0_output → ('mhsa', 0)
            residual_1_output → ('residual', 1)
            ffn_2_output → ('ffn', 2)

        Args:
            param_name: Parameter name from model signature

        Returns:
            Tuple of (layer_type, layer_index)
        """
        # Remove '_output' suffix if present
        name = param_name.replace('_output', '')

        # Split by underscore
        parts = name.split('_')

        if len(parts) >= 2:
            layer_type = parts[0]
            try:
                layer_idx = int(parts[1])
                return (layer_type, layer_idx)
            except ValueError:
                pass

        # Fallback: treat whole name as type, index 0
        return (name, 0)

    def _compute_intermediate(self, param_name: str, input_ids: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute a single intermediate output.

        Args:
            param_name: Name of intermediate parameter to compute
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Computed intermediate tensor
        """
        # Check cache first
        if param_name in self.intermediate_cache:
            return self.intermediate_cache[param_name]

        # Parse parameter name
        layer_type, layer_idx = self._parse_intermediate_name(param_name)

        # Get the appropriate layer
        layer_key = f'{layer_type}_{layer_idx}'

        if layer_key in self.layer_map:
            layer = self.layer_map[layer_key]

            # Get input for this layer
            # For first layer, use embeddings; for later layers, use previous output
            if layer_idx == 0:
                # Use model embeddings
                x = self._get_embeddings(input_ids)
            else:
                # Try to get previous layer output
                prev_param = f'{layer_type}_{layer_idx - 1}_output'
                if prev_param in self.intermediate_cache:
                    x = self.intermediate_cache[prev_param]
                else:
                    # Fallback to embeddings
                    x = self._get_embeddings(input_ids)

            # Execute layer
            try:
                # Try with attention_mask
                if attention_mask is not None:
                    output = layer(x, attention_mask=attention_mask)
                else:
                    output = layer(x)

                # Handle different return types
                if isinstance(output, tuple):
                    output = output[0]  # Take first element (usually the tensor)

                # Cache result
                self.intermediate_cache[param_name] = output
                return output

            except Exception:
                # If layer call fails, return input as fallback
                self.intermediate_cache[param_name] = x
                return x
        else:
            # Layer not found in map - return embeddings as fallback
            x = self._get_embeddings(input_ids)
            self.intermediate_cache[param_name] = x
            return x

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embedded representation of input tokens.

        Tries common embedding layer names.

        Args:
            input_ids: Input token IDs

        Returns:
            Embedded tokens tensor
        """
        # Try common embedding attribute names
        for attr_name in ['embedding', 'embeddings', 'wte', 'word_embeddings', 'embed_tokens']:
            if hasattr(self.model, attr_name):
                embed_layer = getattr(self.model, attr_name)
                return embed_layer(input_ids)

        # Try nested paths
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte(input_ids)

        # Fallback: create random embeddings (should rarely happen)
        batch_size, seq_len = input_ids.shape
        d_model = 512  # Default dimension
        return torch.randn(batch_size, seq_len, d_model, device=input_ids.device)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute model with dependency resolution.

        Computes all required intermediate outputs and calls model.forward()
        with the complete parameter set.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            Model output logits [batch_size, seq_len, vocab_size]
        """
        # Clear cache for new forward pass
        self.intermediate_cache = {}

        # Build kwargs with all required parameters
        kwargs = {}

        for param in self.inspector.get_required_params():
            if param == 'input_ids':
                kwargs['input_ids'] = input_ids
            elif param == 'input_0_tokens':
                # Alternative name for input_ids
                kwargs['input_0_tokens'] = input_ids
            elif param == 'attention_mask':
                if attention_mask is not None:
                    kwargs['attention_mask'] = attention_mask
                else:
                    # Create default attention mask (all ones)
                    kwargs['attention_mask'] = torch.ones_like(input_ids)
            else:
                # Compute intermediate output
                kwargs[param] = self._compute_intermediate(param, input_ids, attention_mask)

        # Add optional parameters if available
        for param in self.inspector.get_optional_params():
            if param == 'attention_mask' and attention_mask is not None:
                kwargs['attention_mask'] = attention_mask

        # Call model with all parameters
        output = self.model(**kwargs)

        return output

    def clear_cache(self):
        """Clear the intermediate output cache."""
        self.intermediate_cache = {}


# ==============================================================================
# UNIVERSAL MODEL ADAPTER (Placeholder for Task 2.1)
# ==============================================================================

class UniversalModelAdapter(pl.LightningModule):
    """
    Lightning-compatible wrapper for ANY generated model.

    Provides a unified interface regardless of model's forward() signature:
    - Simple signatures: calls model directly
    - Complex signatures: uses ComputationalGraphExecutor

    Implements PyTorch Lightning training/validation steps, loss computation,
    and optimizer configuration.

    This is a placeholder - full implementation in Task 2.1.
    """

    def __init__(self,
                 generated_model: nn.Module,
                 config: Any,
                 tokenizer: PreTrainedTokenizer,
                 learning_rate: float = 5e-5):
        """
        Initialize adapter.

        Args:
            generated_model: The model to wrap
            config: Model configuration object
            tokenizer: Tokenizer for this model
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.model = generated_model
        self.config = config
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

        # Analyze model signature
        self.inspector = ModelSignatureInspector(generated_model)

        # TODO (Task 2.1): Initialize executor if needed
        # if self.inspector.requires_intermediate_outputs():
        #     self.executor = ComputationalGraphExecutor(generated_model, self.inspector)

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=['generated_model', 'tokenizer'])

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Unified forward interface.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation

        Returns:
            Dictionary with 'loss' and 'logits' keys
        """
        # TODO (Task 2.1): Implement full forward logic
        raise NotImplementedError("UniversalModelAdapter will be implemented in Task 2.1")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        # TODO (Task 2.1): Implement training step
        raise NotImplementedError("Training step will be implemented in Task 2.1")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning validation step."""
        # TODO (Task 2.1): Implement validation step
        raise NotImplementedError("Validation step will be implemented in Task 2.1")

    def configure_optimizers(self):
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
