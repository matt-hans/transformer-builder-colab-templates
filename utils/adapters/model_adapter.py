"""
Universal Model Adapter utilities and training adapters.

This module provides two layers of abstraction:
- Signature/execution helpers for arbitrarily generated models with complex
  forward() signatures (ModelSignatureInspector, ComputationalGraphExecutor,
  UniversalModelAdapter for Lightning integration).
- A family of lightweight, task-aware ModelAdapter classes used by the
  validation tiers and training/eval loops to interact with arbitrary
  architectures through a consistent API.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from utils.training.task_spec import TaskSpec

# Optional dependency - only needed for Tier 3 (UniversalModelAdapter)
try:
    import pytorch_lightning as pl
    HAS_LIGHTNING = True
except ImportError:
    pl = None
    HAS_LIGHTNING = False

# Optional dependency - only needed for tokenization
try:
    from transformers import PreTrainedTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    PreTrainedTokenizer = None
HAS_TRANSFORMERS = False


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
# TASK-AWARE MODEL ADAPTERS (Workstream B)
# ==============================================================================

class ModelAdapter(ABC):
    """Adapter interface between raw model and task/validation code."""

    @abstractmethod
    def prepare_inputs(self, batch: Dict[str, Any], task: "TaskSpec") -> Dict[str, Any]:
        ...

    @abstractmethod
    def forward_for_loss(
        self,
        model: Any,
        batch: Dict[str, Any],
        task: "TaskSpec",
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run forward pass and return (loss, outputs_dict)."""
        ...

    @abstractmethod
    def get_logits(self, outputs: Dict[str, Any], task: "TaskSpec"):
        ...

    @abstractmethod
    def predict(self, outputs: Dict[str, Any], task: "TaskSpec"):
        ...

    def get_attention_maps(self, outputs: Dict[str, Any], task: "TaskSpec"):
        """Optional: return attention maps for interpretability (Tier 2)."""
        return None


def _extract_logits_generic(output: Any) -> torch.Tensor:
    """Best-effort extraction of logits tensor from common output types."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and len(output) > 0:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    if isinstance(output, dict):
        if 'logits' in output and isinstance(output['logits'], torch.Tensor):
            return output['logits']
        if 'last_hidden_state' in output and isinstance(output['last_hidden_state'], torch.Tensor):
            return output['last_hidden_state']
        # First tensor value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(output, 'logits') and isinstance(output.logits, torch.Tensor):
        return output.logits
    if hasattr(output, 'last_hidden_state') and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    # Fallthrough: return as-is; callers may fail fast
    return output


def get_adapter_for_task(task: "TaskSpec") -> ModelAdapter:
    """
    Factory helper to select a task-aware ModelAdapter based on TaskSpec.

    This keeps adapter selection logic centralized so that new modalities
    (e.g. vision) can plug into existing training/eval workflows without
    changing call sites.
    """
    task_type = getattr(task, "task_type", None)
    modality = getattr(task, "modality", "text")

    if task_type == "lm":
        return DecoderOnlyLMAdapter()
    if task_type == "classification" or task_type == "text_classification":
        return EncoderOnlyClassificationAdapter()
    if task_type == "seq2seq":
        return EncoderDecoderSeq2SeqAdapter()
    if task_type == "vision_classification" and modality == "vision":
        return VisionClassificationAdapter()

    raise ValueError(f"Unsupported task_type/modality combination: task_type={task_type}, modality={modality}")


class DecoderOnlyLMAdapter(ModelAdapter):
    """Adapter for decoder-only language models (LM)."""

    def prepare_inputs(self, batch: Dict[str, Any], task: "TaskSpec") -> Dict[str, Any]:
        prepared = {
            'input_ids': batch.get('input_ids'),
        }
        if 'attention_mask' in batch:
            prepared['attention_mask'] = batch['attention_mask']
        if 'labels' in batch:
            prepared['labels'] = batch['labels']
        return prepared

    def forward_for_loss(
        self,
        model: Any,
        batch: Dict[str, Any],
        task: "TaskSpec",
    ) -> Tuple[Any, Dict[str, Any]]:
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')

        if attention_mask is not None:
            output = model(input_ids, attention_mask=attention_mask)
        else:
            output = model(input_ids)

        logits = _extract_logits_generic(output)
        outputs: Dict[str, Any] = {'logits': logits}

        # Compute language modeling loss if labels are provided
        loss = None
        if labels is not None:
            shift = bool(task.additional_config.get('shift_labels', True))
            pad_id = int(task.special_tokens.get('pad_token_id', -100))
            if shift:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=pad_id,
                )
            else:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=pad_id,
                )

        return loss, outputs

    def get_logits(self, outputs: Dict[str, Any], task: "TaskSpec"):
        return outputs.get('logits')

    def predict(self, outputs: Dict[str, Any], task: "TaskSpec"):
        logits = self.get_logits(outputs, task)
        return logits.argmax(dim=-1)


class EncoderOnlyClassificationAdapter(ModelAdapter):
    """Adapter for encoder-only classification models."""

    def prepare_inputs(self, batch: Dict[str, Any], task: "TaskSpec") -> Dict[str, Any]:
        prepared = {
            'input_ids': batch.get('input_ids'),
        }
        if 'attention_mask' in batch:
            prepared['attention_mask'] = batch['attention_mask']
        if 'labels' in batch:
            prepared['labels'] = batch['labels']
        return prepared

    def _pool_logits(self, logits: torch.Tensor, attention_mask: Optional[torch.Tensor], num_classes: Optional[int]) -> torch.Tensor:
        # If already [B, C], return as-is
        if logits.dim() == 2:
            return logits
        # If [B, T, C], pool over T
        if logits.dim() == 3:
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                summed = (logits * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1e-6)
                return summed / denom
            return logits.mean(dim=1)
        # Otherwise, try flatten last dim to num_classes if known
        if num_classes is not None and logits.size(-1) == num_classes:
            return logits.view(logits.size(0), -1, num_classes).mean(dim=1)
        return logits.squeeze()

    def forward_for_loss(
        self,
        model: Any,
        batch: Dict[str, Any],
        task: "TaskSpec",
    ) -> Tuple[Any, Dict[str, Any]]:
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')

        if attention_mask is not None:
            output = model(input_ids, attention_mask=attention_mask)
        else:
            output = model(input_ids)

        raw_logits = _extract_logits_generic(output)
        num_classes = task.additional_config.get('num_classes')
        pooled = self._pool_logits(raw_logits, attention_mask, num_classes)
        outputs: Dict[str, Any] = {'logits': pooled}

        loss = None
        if labels is not None:
            loss = F.cross_entropy(pooled, labels.long())

        return loss, outputs

    def get_logits(self, outputs: Dict[str, Any], task: "TaskSpec"):
        return outputs.get('logits')

    def predict(self, outputs: Dict[str, Any], task: "TaskSpec"):
        logits = self.get_logits(outputs, task)
        return logits.argmax(dim=-1)


class VisionClassificationAdapter(ModelAdapter):
    """
    Adapter for vision classification models.

    Expects batches with:
        - pixel_values: Tensor of shape [batch_size, channels, height, width]
        - labels: LongTensor of shape [batch_size]
    """

    task_type: str = "vision_classification"

    def prepare_inputs(self, batch: Dict[str, Any], task: "TaskSpec") -> Dict[str, Any]:
        """
        Prepare inputs for vision classification models.

        Args:
            batch: Dictionary containing at least 'pixel_values', optionally 'labels'.
            task: TaskSpec describing the task (unused here but kept for symmetry).

        Returns:
            Dictionary with keys:
                - 'pixel_values'
                - 'labels' (if present in the input batch)
        """
        if "pixel_values" not in batch:
            raise KeyError(f"Expected 'pixel_values' in batch, found: {list(batch.keys())}")

        prepared: Dict[str, Any] = {"pixel_values": batch["pixel_values"]}
        if "labels" in batch:
            prepared["labels"] = batch["labels"]
        return prepared

    def forward_for_loss(
        self,
        model: Any,
        batch: Dict[str, Any],
        task: "TaskSpec",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run forward pass and compute loss for vision classification.

        Args:
            model: Vision model expecting [B, C, H, W] input.
            batch: Prepared batch with 'pixel_values' and optional 'labels'.
            task: TaskSpec describing the task (may carry num_classes in output_schema/additional_config).

        Returns:
            Tuple of (loss, outputs_dict) where:
                - loss is a scalar tensor or None if labels are missing
                - outputs_dict contains 'logits' with shape [B, num_classes]
        """
        pixel_values = batch["pixel_values"]
        labels = batch.get("labels")

        logits = model(pixel_values)
        logits = _extract_logits_generic(logits)
        outputs: Dict[str, Any] = {"logits": logits}

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return loss, outputs

    def get_logits(self, outputs: Dict[str, Any], task: "TaskSpec") -> torch.Tensor:
        """Extract logits tensor from adapter outputs."""
        logits = outputs.get("logits")
        if logits is None:
            raise KeyError("Expected 'logits' key in outputs for VisionClassificationAdapter.")
        return logits

    def predict(self, outputs: Dict[str, Any], task: "TaskSpec") -> torch.Tensor:
        """
        Compute hard predictions (argmax over class dimension).

        Args:
            outputs: Adapter outputs containing 'logits'.
            task: TaskSpec describing the task.

        Returns:
            LongTensor of shape [batch_size] with predicted class indices.
        """
        logits = self.get_logits(outputs, task)
        return logits.argmax(dim=-1)


class EncoderDecoderSeq2SeqAdapter(ModelAdapter):
    """Adapter for encoder–decoder seq2seq models."""

    def prepare_inputs(self, batch: Dict[str, Any], task: "TaskSpec") -> Dict[str, Any]:
        prepared = {
            'input_ids': batch.get('input_ids'),
            'decoder_input_ids': batch.get('decoder_input_ids'),
        }
        if 'attention_mask' in batch:
            prepared['attention_mask'] = batch['attention_mask']
        if 'labels' in batch:
            prepared['labels'] = batch['labels']
        return prepared

    def forward_for_loss(
        self,
        model: Any,
        batch: Dict[str, Any],
        task: "TaskSpec",
    ) -> Tuple[Any, Dict[str, Any]]:
        kwargs: Dict[str, Any] = {
            'input_ids': batch.get('input_ids'),
            'decoder_input_ids': batch.get('decoder_input_ids'),
        }
        if batch.get('attention_mask') is not None:
            kwargs['attention_mask'] = batch['attention_mask']

        output = model(**kwargs) if hasattr(model, 'forward') else model(kwargs)
        logits = _extract_logits_generic(output)
        outputs: Dict[str, Any] = {'logits': logits}

        labels = batch.get('labels')
        loss = None
        if labels is not None:
            ignore_index = int(task.special_tokens.get('ignore_index', -100))
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=ignore_index,
            )

        # Try to expose attention maps if present
        if isinstance(output, dict) and 'attentions' in output:
            outputs['attentions'] = output['attentions']

        return loss, outputs

    def get_logits(self, outputs: Dict[str, Any], task: "TaskSpec"):
        return outputs.get('logits')

    def predict(self, outputs: Dict[str, Any], task: "TaskSpec"):
        logits = self.get_logits(outputs, task)
        return logits.argmax(dim=-1)

    def get_attention_maps(self, outputs: Dict[str, Any], task: "TaskSpec"):
        return outputs.get('attentions')


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
# FLASH ATTENTION WRAPPER (v3.6.0)
# ==============================================================================

import logging
logger = logging.getLogger(__name__)


class FlashAttentionWrapper:
    """
    Wrapper to enable Flash Attention (SDPA) for compatible PyTorch models.

    PyTorch 2.0+ nn.MultiheadAttention automatically uses SDPA when:
    - PyTorch >= 2.0
    - CUDA available
    - fast_path conditions met

    This wrapper validates compatibility and logs enabled layers.
    No actual patching needed - PyTorch handles it internally via fast_path.
    """

    def __init__(self, model: nn.Module, enable: bool = True):
        """
        Initialize flash attention wrapper.

        Args:
            model: PyTorch model to wrap
            enable: Whether to enable flash attention (default: True)
        """
        self.model = model
        self.enable = enable
        self.patched_layers: List[str] = []
        self.sdpa_available = False

        if enable:
            self.sdpa_available = self._check_sdpa_availability()
            if self.sdpa_available:
                self._detect_attention_layers()

    @staticmethod
    def _check_sdpa_availability() -> bool:
        """
        Check if SDPA is available in current environment.

        Requirements:
            - PyTorch >= 2.0
            - CUDA available (SDPA flash attention kernel requires GPU)
            - F.scaled_dot_product_attention function exists

        Returns:
            bool: True if SDPA can be used
        """
        # Check PyTorch version >= 2.0
        version_parts = torch.__version__.split('.')
        try:
            major_version = int(version_parts[0])
        except (ValueError, IndexError):
            logger.debug(
                f"Unable to parse PyTorch version '{torch.__version__}'. "
                "Flash attention disabled."
            )
            return False

        if major_version < 2:
            logger.debug(
                f"SDPA requires PyTorch >= 2.0, found {torch.__version__}. "
                "Flash attention disabled."
            )
            return False

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.debug("CUDA not available. Flash attention disabled.")
            return False

        # Check if SDPA function exists
        if not hasattr(F, 'scaled_dot_product_attention'):
            logger.warning(
                "torch.nn.functional.scaled_dot_product_attention not found. "
                "This is unexpected for PyTorch 2.0+. Flash attention disabled."
            )
            return False

        return True

    def _detect_attention_layers(self) -> None:
        """
        Detect nn.MultiheadAttention layers in model.

        PyTorch 2.0+ MultiheadAttention automatically uses SDPA fast path when:
        - fast_path=True (default)
        - No attention mask or boolean mask
        - _qkv_same_embed_dim=True (default for most models)

        This method logs layers that will benefit from SDPA.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Check if module meets SDPA fast path requirements
                if hasattr(module, '_qkv_same_embed_dim') and module._qkv_same_embed_dim:
                    self.patched_layers.append(name)
                    logger.debug(f"✓ SDPA-compatible attention layer detected: {name}")
                else:
                    logger.debug(
                        f"⚠ Attention layer {name} not SDPA-compatible "
                        "(qkv_same_embed_dim=False)"
                    )

        if self.patched_layers:
            # Format layer names for concise logging
            layer_summary = ', '.join(self.patched_layers[:3])
            if len(self.patched_layers) > 3:
                layer_summary += f" and {len(self.patched_layers) - 3} more"

            logger.info(
                f"✅ Flash Attention (SDPA) enabled for {len(self.patched_layers)} "
                f"attention layer(s): {layer_summary}"
            )
        else:
            logger.info(
                "ℹ️  No nn.MultiheadAttention layers found. Flash attention not applicable "
                "for this model architecture."
            )


# ==============================================================================
# UNIVERSAL MODEL ADAPTER
# ==============================================================================

# Only define if pytorch_lightning is available (Tier 3 only)
if HAS_LIGHTNING:
    class UniversalModelAdapter(pl.LightningModule):
        """
        Lightning-compatible wrapper for ANY generated model.

        Provides a unified interface regardless of model's forward() signature:
    - Simple signatures: calls model directly
    - Complex signatures: uses ComputationalGraphExecutor

    Implements PyTorch Lightning training/validation steps, loss computation,
    and optimizer configuration.

    Example:
        >>> model = YourGeneratedModel(**config_dict)
        >>> adapter = UniversalModelAdapter(model, config, tokenizer)
        >>> trainer = pl.Trainer(max_epochs=3)
        >>> trainer.fit(adapter, datamodule)
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
            config: Model configuration object with vocab_size attribute
            tokenizer: Tokenizer for this model
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

        # Analyze model signature BEFORE compilation (important!)
        self.inspector = ModelSignatureInspector(generated_model)

        # === NEW: Flash Attention (v3.6) ===
        # Apply flash attention wrapper BEFORE compilation
        # (SDPA + torch.compile = additive speedup)
        self.flash_wrapper = FlashAttentionWrapper(generated_model, enable=True)
        if self.flash_wrapper.sdpa_available and self.flash_wrapper.patched_layers:
            logger.info(
                f"🚀 Flash Attention (SDPA) enabled - expect 2-4x attention speedup "
                f"on {len(self.flash_wrapper.patched_layers)} layers"
            )

        # === Existing: torch.compile (v3.5) ===
        # Apply torch.compile if configured (v3.5.0)
        # Compile AFTER flash attention wrapper and signature inspection
        if hasattr(config, 'compile_mode') and config.compile_mode is not None:
            compiled_model = self._compile_model(
                generated_model,
                mode=config.compile_mode,
                fullgraph=getattr(config, 'compile_fullgraph', False),
                dynamic=getattr(config, 'compile_dynamic', True)
            )
            self.model = compiled_model
        else:
            self.model = generated_model

        # Initialize executor if model has complex signature
        self.executor = None
        if self.inspector.requires_intermediate_outputs():
            self.executor = ComputationalGraphExecutor(generated_model, self.inspector)

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=['generated_model', 'tokenizer', 'config'])

    def _compile_model(self, model: nn.Module, mode: str, fullgraph: bool, dynamic: bool) -> nn.Module:
        """
        Apply torch.compile with error handling and fallback.

        Args:
            model: Model to compile
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            fullgraph: If True, require single graph (stricter, may fail)
            dynamic: If True, support dynamic shapes (safer for variable seq lengths)

        Returns:
            Compiled model, or original model if compilation fails
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Check if torch.compile is available (PyTorch >= 2.0)
            if not hasattr(torch, 'compile'):
                logger.warning(
                    "torch.compile not available (PyTorch < 2.0). "
                    "Skipping compilation. Upgrade to PyTorch 2.0+ for compilation support."
                )
                return model

            logger.info(f"Compiling model with mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
            compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
            logger.info("✅ Model compilation successful")
            return compiled

        except Exception as e:
            logger.warning(
                f"⚠️  Model compilation failed: {e}. "
                f"Continuing with uncompiled model. "
                f"This is expected for models with exotic operations or dynamic control flow."
            )
            return model

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Unified forward interface.

        Automatically handles both simple and complex model signatures.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for loss computation [batch_size, seq_len]

        Returns:
            Dictionary with keys:
                - 'logits': Model output logits [batch_size, seq_len, vocab_size]
                - 'loss': Cross-entropy loss (if labels provided)
        """
        # Get logits using appropriate method
        if self.executor is not None:
            # Complex signature - use executor
            logits = self.executor.forward(input_ids, attention_mask)
        else:
            # Simple signature - call model directly
            params = self.inspector.get_parameters()

            if 'attention_mask' in params and attention_mask is not None:
                logits = self.model(input_ids, attention_mask=attention_mask)
            else:
                logits = self.model(input_ids)

        # Handle tuple returns (some models return (logits, hidden_states, ...))
        if isinstance(logits, tuple):
            logits = logits[0]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Get vocab size from config or infer from logits
            vocab_size = getattr(self.config, 'vocab_size', logits.shape[-1])

            # Cross-entropy loss (language modeling)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=getattr(self.tokenizer, 'pad_token_id', -100)
            )

        return {'logits': logits, 'loss': loss}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Lightning training step.

        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', 'labels'
            batch_idx: Batch index

        Returns:
            Training loss
        """
        output = self(
            batch['input_ids'],
            batch.get('attention_mask'),
            batch.get('labels')
        )

        loss = output['loss']

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        # Train perplexity (epoch-level), with numerical stability clamp
        try:
            ppl = torch.exp(torch.clamp(loss.detach(), max=torch.tensor(20.0, device=loss.device)))
            self.log('train_perplexity', ppl, prog_bar=True, on_step=False, on_epoch=True)
        except Exception:
            pass

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Lightning validation step.

        Args:
            batch: Dictionary with 'input_ids', 'attention_mask', 'labels'
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        output = self(
            batch['input_ids'],
            batch.get('attention_mask'),
            batch.get('labels')
        )

        loss = output['loss']

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        # Compute perplexity with numerical stability clamp
        perplexity = torch.exp(torch.clamp(loss.detach(), max=torch.tensor(20.0, device=loss.device)))
        self.log('val_perplexity', perplexity, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure AdamW optimizer.

        Returns:
            AdamW optimizer with configured learning rate
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.model.eval()

        generated = input_ids

        for _ in range(max_new_tokens):
            # Get logits for next token
            with torch.no_grad():
                output = self(generated)
                logits = output['logits']

            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated
else:
    # Stub class when pytorch_lightning is not available
    class UniversalModelAdapter:
        """
        Stub class for UniversalModelAdapter when pytorch_lightning is not installed.

        This class is only used for Tier 3 tests. If you see this error, run the
        Tier 3 installation cell to install pytorch_lightning.
        """
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "UniversalModelAdapter requires pytorch_lightning. "
                "This is only needed for Tier 3 tests. "
                "Please run the Tier 3 installation cell before using this feature."
            )
