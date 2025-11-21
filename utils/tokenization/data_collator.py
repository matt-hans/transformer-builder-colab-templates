"""
Data collators for variable-length sequences.

Provides a lightweight LanguageModelingDataCollator that performs dynamic
padding, builds attention masks, and supports causal (GPT) and masked (BERT)
objectives without requiring transformers at import time.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch


class LanguageModelingDataCollator:
    """Custom data collator for transformer language modeling.

    - Causal LM (default): labels = input_ids (model performs shift internally)
    - Masked LM: masks tokens with given probability when tokenizer supports it
    - Dynamic padding with optional left/right side
    """

    def __init__(self,
                 tokenizer: Any,
                 mlm: bool = False,
                 mlm_probability: float = 0.15,
                 padding_side: str = 'right'):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.padding_side = padding_side

    def _safe_copy(self, data: Any) -> Any:
        """
        Copy data safely, handling both tensors and lists/numpy arrays.

        Args:
            data: torch.Tensor, list, or numpy.ndarray

        Returns:
            Copy of data (same type as input)
        """
        if torch.is_tensor(data):
            return data.clone()  # PyTorch tensors use .clone()
        elif hasattr(data, 'copy'):
            return data.copy()   # Lists and numpy arrays use .copy()
        else:
            # Fallback: convert to list
            return list(data)

    def _ensure_tensors(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert all list values in batch to PyTorch tensors.

        Handles both list of tensors (from HuggingFace tokenizer) and list of lists.
        Required because loss functions expect tensors with .ndim/.shape attributes.

        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels' as lists or tensors

        Returns:
            Dict with all values as PyTorch tensors
        """
        result = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                # Already a tensor, keep as-is
                result[key] = value
            elif isinstance(value, list):
                if len(value) == 0:
                    # Empty list, convert to empty tensor
                    result[key] = torch.tensor(value)
                elif torch.is_tensor(value[0]):
                    # List of tensors (e.g., from tokenizer.pad with return_tensors=None)
                    result[key] = torch.stack(value)
                else:
                    # List of lists/ints (e.g., from _pad_examples)
                    result[key] = torch.tensor(value)
            else:
                # Other types (int, str, etc.), keep as-is
                result[key] = value
        return result

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Use tokenizer.pad when available
        batch = None
        if hasattr(self.tokenizer, 'pad'):
            # Temporarily set padding_side if supported
            original_side = getattr(self.tokenizer, 'padding_side', None)
            try:
                if original_side is not None:
                    self.tokenizer.padding_side = self.padding_side
                batch = self.tokenizer.pad(
                    examples,
                    return_tensors=None,  # leave as lists; downstream will cast to torch if available
                    padding=True,
                )
            finally:
                if original_side is not None:
                    self.tokenizer.padding_side = original_side
        else:
            batch = self._pad_examples(examples)

        # Ensure attention_mask exists
        if 'attention_mask' not in batch:
            batch['attention_mask'] = self._build_attention_mask(batch['input_ids'])

        if not self.mlm:
            # labels same as input_ids for causal LM
            # (model performs shifting internally)
            batch['labels'] = [self._safe_copy(seq) for seq in batch['input_ids']]
        else:
            input_ids = batch['input_ids']
            labels, masked_inputs = self._mask_tokens(input_ids)
            batch['labels'] = labels
            batch['input_ids'] = masked_inputs

        # Convert BatchEncoding to plain dict (HuggingFace #23138 workaround)
        # tokenizer.pad() returns BatchEncoding which breaks ** unpacking in trainer
        batch = dict(batch)

        # Convert lists to tensors (required for loss computation)
        # tokenizer.pad() with return_tensors=None returns lists, but trainer expects tensors
        batch = self._ensure_tensors(batch)

        return batch

    def _pad_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Determine pad token id
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        max_len = 0
        for ex in examples:
            max_len = max(max_len, len(ex['input_ids']))

        result_ids: List[List[int]] = []
        for ex in examples:
            ids = list(ex['input_ids'])
            pad_len = max_len - len(ids)
            if self.padding_side == 'left':
                padded = [pad_id] * pad_len + ids
            else:
                padded = ids + [pad_id] * pad_len
            result_ids.append(padded)

        return {'input_ids': result_ids}

    def _build_attention_mask(self, input_ids: List[List[int]]) -> List[List[int]]:
        masks: List[List[int]] = []
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        for seq in input_ids:
            masks.append([0 if tok == pad_id else 1 for tok in seq])
        return masks

    def _mask_tokens(self, input_ids: List[List[int]]) -> (List[List[int]], List[List[int]]):
        # Simple masking strategy if tokenizer supports mask_token_id
        import random
        mask_id = getattr(self.tokenizer, 'mask_token_id', None)
        if mask_id is None:
            # fallback: no masking; labels = input_ids
            return ([self._safe_copy(s) for s in input_ids], [self._safe_copy(s) for s in input_ids])

        labels: List[List[int]] = []
        masked_inputs: List[List[int]] = []
        special_ids = set(getattr(self.tokenizer, 'all_special_ids', []) or [])
        for seq in input_ids:
            lbl = self._safe_copy(seq)
            inp = self._safe_copy(seq)
            for i, tok in enumerate(seq):
                if tok in special_ids:
                    continue
                if random.random() < self.mlm_probability:
                    inp[i] = mask_id
                else:
                    lbl[i] = -100  # ignore index for loss
            labels.append(lbl)
            masked_inputs.append(inp)
        return labels, masked_inputs


class VisionDataCollator:
    """Data collator for vision tasks (classification, multilabel, etc.).

    Handles batching pixel_values tensors and applies normalization in the
    collate_fn for improved performance compared to per-sample normalization
    in Dataset.__getitem__.

    Features:
    - Stacks pixel_values tensors along batch dimension
    - Per-channel normalization with configurable mean/std
    - Supports both RGB (3-channel) and grayscale (1-channel) images
    - Handles optional labels (supports inference mode)

    Args:
        normalize: Whether to apply normalization. Defaults to True.
        mean: Per-channel mean for normalization. Defaults to ImageNet mean.
        std: Per-channel std for normalization. Defaults to ImageNet std.
    """

    def __init__(
        self,
        normalize: bool = True,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ):
        """Initialize vision data collator.

        Args:
            normalize: Whether to apply normalization (default: True)
            mean: Per-channel mean values. Defaults to ImageNet: (0.485, 0.456, 0.406)
            std: Per-channel std values. Defaults to ImageNet: (0.229, 0.224, 0.225)

        Raises:
            ValueError: If mean and std have different lengths
        """
        self.normalize = normalize

        # Default to ImageNet normalization
        self.mean = mean or (0.485, 0.456, 0.406)
        self.std = std or (0.229, 0.224, 0.225)

        if len(self.mean) != len(self.std):
            raise ValueError(
                f"mean and std must have same length, got {len(self.mean)} vs {len(self.std)}"
            )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of vision samples.

        Args:
            batch: List of sample dicts, each containing:
                - 'pixel_values': Tensor of shape (C, H, W)
                - 'labels': Optional label (int or tensor)

        Returns:
            Dictionary with:
                - 'pixel_values': Tensor of shape (B, C, H, W)
                - 'labels': Optional tensor of shape (B,) if labels present

        Raises:
            ValueError: If pixel_values have inconsistent shapes across batch
        """
        # Stack pixel values
        pixel_values_list = [item['pixel_values'] for item in batch]

        # Validate shapes are consistent
        if len(pixel_values_list) > 1:
            first_shape = pixel_values_list[0].shape
            for i, pv in enumerate(pixel_values_list[1:], 1):
                if pv.shape != first_shape:
                    raise ValueError(
                        f"Inconsistent pixel_values shapes in batch: "
                        f"item 0 has shape {first_shape}, item {i} has shape {pv.shape}"
                    )

        pixel_values = torch.stack(pixel_values_list)

        # Apply normalization if enabled
        if self.normalize:
            pixel_values = self._normalize(pixel_values)

        collated = {'pixel_values': pixel_values}

        # Stack labels if present (optional for inference mode)
        if 'labels' in batch[0]:
            labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
            collated['labels'] = labels

        return collated

    def _normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Apply per-channel normalization to pixel values.

        Formula: normalized = (pixel_values - mean) / std

        Args:
            pixel_values: Tensor of shape (B, C, H, W)

        Returns:
            Normalized tensor of same shape
        """
        # pixel_values: [B, C, H, W]
        # mean/std: broadcast to shape (1, C, 1, 1)
        num_channels = pixel_values.shape[1]

        # Handle both RGB and grayscale
        if num_channels == 1:
            # Grayscale: use first channel of mean/std
            mean_tensor = torch.tensor([self.mean[0]], device=pixel_values.device, dtype=pixel_values.dtype)
            std_tensor = torch.tensor([self.std[0]], device=pixel_values.device, dtype=pixel_values.dtype)
        else:
            # RGB or multi-channel: use full mean/std
            if len(self.mean) != num_channels:
                raise ValueError(
                    f"Number of channels ({num_channels}) doesn't match "
                    f"mean length ({len(self.mean)}). Ensure mean/std match image channels."
                )
            mean_tensor = torch.tensor(self.mean, device=pixel_values.device, dtype=pixel_values.dtype)
            std_tensor = torch.tensor(self.std, device=pixel_values.device, dtype=pixel_values.dtype)

        # Reshape for broadcasting: (1, C, 1, 1)
        mean_tensor = mean_tensor.view(1, -1, 1, 1)
        std_tensor = std_tensor.view(1, -1, 1, 1)

        return (pixel_values - mean_tensor) / std_tensor

