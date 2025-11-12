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
# COMPUTATIONAL GRAPH EXECUTOR (Placeholder for Task 1.4)
# ==============================================================================

class ComputationalGraphExecutor:
    """
    Resolves and computes intermediate dependencies in model forward pass.

    For models with complex signatures that require intermediate outputs
    (e.g., mhsa_0_output, residual_0_output), this class:
    1. Builds a dependency graph
    2. Computes intermediates in correct order
    3. Caches results to avoid redundant computation
    4. Calls model.forward() with all required parameters

    This is a placeholder - full implementation in Task 1.4.
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

        # TODO (Task 1.4): Build dependency graph
        # self.dependency_graph = self._build_dependency_graph()

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Execute model with dependency resolution.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            Model output logits
        """
        # TODO (Task 1.4): Implement full dependency resolution
        # For now, just call model directly
        raise NotImplementedError("ComputationalGraphExecutor will be implemented in Task 1.4")


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
